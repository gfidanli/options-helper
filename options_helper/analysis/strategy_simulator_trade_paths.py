from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from options_helper.analysis.strategy_simulator_trade_utils import (
    apply_stop_move_rules,
    build_closed_trade,
    evaluation_end_index,
    finalize_target_exit_or_reject,
    intrabar_hits_stop,
    intrabar_hits_target,
    open_hits_stop,
    open_hits_target,
    rejected_trade,
    target_price,
    timestamp_from_ns,
    update_excursions,
)
from options_helper.schemas.strategy_modeling_contracts import (
    StrategySignalEvent,
    StrategyTradeSimulation,
    TradeExitReason,
)
from options_helper.schemas.strategy_modeling_policy import GapFillPolicy, MaxHoldUnit, StopMoveRule

if TYPE_CHECKING:
    from options_helper.analysis.strategy_simulator import (
        StrategyRTarget,
        _EntryDecision,
        _PreparedIntraday,
    )


_EPSILON = 1e-12


@dataclass(frozen=True)
class _TargetSimulationInputs:
    direction: str
    initial_stop_price: float
    entry_row_index: int
    entry_ts: pd.Timestamp
    entry_price: float
    initial_risk: float
    target_price: float


@dataclass
class _TargetLoopState:
    stop_price: float
    stop_move_index: int = 0
    mae_r: float = 0.0
    mfe_r: float = 0.0
    gap_fill_applied: bool = False
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: TradeExitReason | None = None
    holding_bars: int = 0


def simulate_one_target(
    *,
    event: StrategySignalEvent,
    prepared: _PreparedIntraday | None,
    entry_decision: _EntryDecision,
    target: StrategyRTarget,
    max_hold_bars: int | None,
    max_hold_timeframe: tuple[MaxHoldUnit, int],
    gap_fill_policy: GapFillPolicy,
    stop_move_rules: tuple[StopMoveRule, ...],
) -> StrategyTradeSimulation:
    if entry_decision.reject_code is not None:
        return rejected_trade(
            event=event,
            target=target,
            reject_code=entry_decision.reject_code,
            entry_ts=entry_decision.entry_ts,
            entry_price=entry_decision.entry_price,
            stop_price=entry_decision.stop_price,
            initial_risk=entry_decision.initial_risk,
        )

    inputs = _target_inputs_from_entry_decision(
        entry_decision=entry_decision,
        target=target,
        prepared=prepared,
    )
    if inputs is None:
        return rejected_trade(event=event, target=target, reject_code="invalid_signal")

    eval_start, eval_end, has_full_hold_coverage = _resolve_target_eval_window(
        prepared=prepared,
        entry_row_index=inputs.entry_row_index,
        entry_ts=inputs.entry_ts,
        max_hold_bars=max_hold_bars,
        max_hold_timeframe=max_hold_timeframe,
    )
    empty_window_reject = _reject_for_empty_eval_window(
        event=event,
        target=target,
        inputs=inputs,
        eval_start=eval_start,
        eval_end=eval_end,
    )
    if empty_window_reject is not None:
        return empty_window_reject

    eval_len = int(eval_end - eval_start)
    state = _simulate_target_path_loop(
        prepared=prepared,
        inputs=inputs,
        eval_start=eval_start,
        eval_end=eval_end,
        gap_fill_policy=gap_fill_policy,
        stop_move_rules=stop_move_rules,
    )
    rejected = finalize_target_exit_or_reject(
        event=event,
        target=target,
        prepared=prepared,
        inputs=inputs,
        state=state,
        eval_end=eval_end,
        eval_len=eval_len,
        max_hold_bars=max_hold_bars,
        has_full_hold_coverage=has_full_hold_coverage,
    )
    if rejected is not None:
        return rejected
    return build_closed_trade(event=event, target=target, inputs=inputs, state=state)


def _target_inputs_from_entry_decision(
    *,
    entry_decision: _EntryDecision,
    target: StrategyRTarget,
    prepared: _PreparedIntraday | None,
) -> _TargetSimulationInputs | None:
    direction = entry_decision.direction
    initial_stop_price = entry_decision.stop_price
    entry_row_index = entry_decision.entry_row_index
    entry_ts = entry_decision.entry_ts
    entry_price = entry_decision.entry_price
    initial_risk = entry_decision.initial_risk
    if (
        direction is None
        or initial_stop_price is None
        or entry_row_index is None
        or entry_ts is None
        or entry_price is None
        or initial_risk is None
        or prepared is None
    ):
        return None
    return _TargetSimulationInputs(
        direction=direction,
        initial_stop_price=float(initial_stop_price),
        entry_row_index=int(entry_row_index),
        entry_ts=entry_ts,
        entry_price=float(entry_price),
        initial_risk=float(initial_risk),
        target_price=target_price(
            direction=direction,
            entry_price=float(entry_price),
            risk=float(initial_risk),
            target_r=target.target_r,
        ),
    )


def _resolve_target_eval_window(
    *,
    prepared: _PreparedIntraday,
    entry_row_index: int,
    entry_ts: pd.Timestamp,
    max_hold_bars: int | None,
    max_hold_timeframe: tuple[MaxHoldUnit, int],
) -> tuple[int, int, bool]:
    eval_start = int(entry_row_index)
    eval_end, has_full_hold_coverage = evaluation_end_index(
        prepared=prepared,
        entry_row_index=entry_row_index,
        entry_ts=entry_ts,
        max_hold_bars=max_hold_bars,
        max_hold_timeframe=max_hold_timeframe,
    )
    return eval_start, eval_end, has_full_hold_coverage


def _reject_for_empty_eval_window(
    *,
    event: StrategySignalEvent,
    target: StrategyRTarget,
    inputs: _TargetSimulationInputs,
    eval_start: int,
    eval_end: int,
) -> StrategyTradeSimulation | None:
    eval_len = int(eval_end - eval_start)
    if eval_len > 0:
        return None
    return rejected_trade(
        event=event,
        target=target,
        reject_code="insufficient_future_bars",
        entry_ts=inputs.entry_ts,
        entry_price=inputs.entry_price,
        stop_price=inputs.initial_stop_price,
        target_price=inputs.target_price,
        initial_risk=inputs.initial_risk,
    )


def _simulate_target_path_loop(
    *,
    prepared: _PreparedIntraday,
    inputs: _TargetSimulationInputs,
    eval_start: int,
    eval_end: int,
    gap_fill_policy: GapFillPolicy,
    stop_move_rules: tuple[StopMoveRule, ...],
) -> _TargetLoopState:
    state = _TargetLoopState(stop_price=float(inputs.initial_stop_price))
    for offset, row_index in enumerate(range(eval_start, eval_end), start=1):
        state.holding_bars = offset
        open_price = float(prepared.open_values[row_index])
        high_price = float(prepared.high_values[row_index])
        low_price = float(prepared.low_values[row_index])
        close_price = float(prepared.close_values[row_index])
        open_exit = _open_exit_decision(
            direction=inputs.direction,
            open_price=open_price,
            stop_price=state.stop_price,
            target_price=inputs.target_price,
            gap_fill_policy=gap_fill_policy,
        )
        if open_exit is not None:
            reason, price, gap_fill_applied = open_exit
            _set_loop_exit(
                state=state,
                bar_ts=timestamp_from_ns(int(prepared.timestamp_ns[row_index])),
                reason=reason,
                price=price,
                prices=(open_price, price),
                inputs=inputs,
                gap_fill_applied=gap_fill_applied,
            )
            break

        intrabar_exit = _intrabar_exit_decision(
            direction=inputs.direction,
            low_price=low_price,
            high_price=high_price,
            stop_price=state.stop_price,
            target_price=inputs.target_price,
        )
        if intrabar_exit is not None:
            reason, price = intrabar_exit
            _set_loop_exit(
                state=state,
                bar_ts=timestamp_from_ns(int(prepared.timestamp_ns[row_index])),
                reason=reason,
                price=price,
                prices=(open_price, price),
                inputs=inputs,
                gap_fill_applied=False,
            )
            break

        state.mae_r, state.mfe_r = update_excursions(
            prices=(open_price, high_price, low_price, close_price),
            direction=inputs.direction,
            entry_price=inputs.entry_price,
            initial_risk=inputs.initial_risk,
            current_mae_r=state.mae_r,
            current_mfe_r=state.mfe_r,
        )
        if stop_move_rules and row_index + 1 < eval_end and state.stop_move_index < len(stop_move_rules):
            state.stop_price, state.stop_move_index = apply_stop_move_rules(
                stop_move_rules=stop_move_rules,
                stop_move_index=state.stop_move_index,
                direction=inputs.direction,
                entry_price=inputs.entry_price,
                initial_risk=inputs.initial_risk,
                close_price=close_price,
                stop_price=state.stop_price,
            )
    return state


def _open_exit_decision(
    *,
    direction: str,
    open_price: float,
    stop_price: float,
    target_price: float,
    gap_fill_policy: GapFillPolicy,
) -> tuple[TradeExitReason, float, bool] | None:
    stop_at_open = open_hits_stop(direction=direction, open_price=open_price, stop_price=stop_price)
    target_at_open = open_hits_target(direction=direction, open_price=open_price, target_price=target_price)
    if not (stop_at_open or target_at_open):
        return None
    if stop_at_open:
        gap_fill_applied = gap_fill_policy == "fill_at_open" and abs(open_price - stop_price) > _EPSILON
        return "stop_hit", open_price, gap_fill_applied
    gap_fill_applied = gap_fill_policy == "fill_at_open" and abs(open_price - target_price) > _EPSILON
    return "target_hit", open_price, gap_fill_applied


def _intrabar_exit_decision(
    *,
    direction: str,
    low_price: float,
    high_price: float,
    stop_price: float,
    target_price: float,
) -> tuple[TradeExitReason, float] | None:
    stop_touched = intrabar_hits_stop(
        direction=direction,
        low_price=low_price,
        high_price=high_price,
        stop_price=stop_price,
    )
    target_touched = intrabar_hits_target(
        direction=direction,
        low_price=low_price,
        high_price=high_price,
        target_price=target_price,
    )
    if stop_touched and target_touched:
        return "stop_hit", stop_price
    if stop_touched:
        return "stop_hit", stop_price
    if target_touched:
        return "target_hit", target_price
    return None


def _set_loop_exit(
    *,
    state: _TargetLoopState,
    bar_ts: pd.Timestamp,
    reason: TradeExitReason,
    price: float,
    prices: Sequence[float],
    inputs: _TargetSimulationInputs,
    gap_fill_applied: bool,
) -> None:
    state.exit_reason = reason
    state.exit_price = price
    state.exit_ts = bar_ts
    state.gap_fill_applied = gap_fill_applied
    state.mae_r, state.mfe_r = update_excursions(
        prices=prices,
        direction=inputs.direction,
        entry_price=inputs.entry_price,
        initial_risk=inputs.initial_risk,
        current_mae_r=state.mae_r,
        current_mfe_r=state.mfe_r,
    )


__all__ = ["simulate_one_target"]
