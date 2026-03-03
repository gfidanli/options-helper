from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math
from typing import TYPE_CHECKING, Sequence

import numpy as np
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
    price_to_r,
    rejected_trade,
    target_price,
    timestamp_from_ns,
    update_excursions,
)
from options_helper.schemas.strategy_modeling_contracts import (
    StrategySignalEvent,
    StrategyTradeStopUpdate,
    StrategyTradeSimulation,
    TradeExitReason,
)
from options_helper.schemas.strategy_modeling_policy import (
    GapFillPolicy,
    MaxHoldUnit,
    StopMoveRule,
    StopTrailRule,
)

if TYPE_CHECKING:
    from options_helper.analysis.strategy_simulator import (
        StrategyRTarget,
        _EntryDecision,
        _PreparedIntraday,
    )


_EPSILON = 1e-12


@dataclass(frozen=True)
class _DailyStopTrailIndicators:
    session_day_ordinals: np.ndarray
    ema_by_span: dict[int, np.ndarray]
    atr14_values: np.ndarray


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
    max_close_r: float | None = None
    last_session_day_ordinal: int | None = None
    gap_fill_applied: bool = False
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: TradeExitReason | None = None
    holding_bars: int = 0
    stop_updates: list[StrategyTradeStopUpdate] = field(default_factory=list)


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
    stop_trail_rules: Sequence[StopTrailRule] = (),
    daily_ohlc: pd.DataFrame | None = None,
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
        stop_trail_rules=stop_trail_rules,
        daily_ohlc=daily_ohlc,
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
    stop_trail_rules: Sequence[StopTrailRule],
    daily_ohlc: pd.DataFrame | None,
) -> _TargetLoopState:
    state = _TargetLoopState(stop_price=float(inputs.initial_stop_price))
    trail_indicators = _build_daily_stop_trail_indicators(
        daily_ohlc=daily_ohlc,
        stop_trail_rules=stop_trail_rules,
    )
    for offset, row_index in enumerate(range(eval_start, eval_end), start=1):
        state.holding_bars = offset
        bar_ts = timestamp_from_ns(int(prepared.timestamp_ns[row_index]))
        session_day_ordinal = _session_day_ordinal(
            prepared.session_days[row_index],
            fallback_ts=bar_ts,
        )
        if (
            trail_indicators is not None
            and state.last_session_day_ordinal is not None
            and session_day_ordinal != state.last_session_day_ordinal
        ):
            _apply_stop_trail_for_new_session(
                state=state,
                inputs=inputs,
                bar_ts=bar_ts,
                session_day_ordinal=session_day_ordinal,
                trail_indicators=trail_indicators,
                stop_trail_rules=stop_trail_rules,
            )
        state.last_session_day_ordinal = session_day_ordinal

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
                bar_ts=bar_ts,
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
                bar_ts=bar_ts,
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
        close_r = price_to_r(
            direction=inputs.direction,
            entry_price=inputs.entry_price,
            initial_risk=inputs.initial_risk,
            price=close_price,
        )
        if state.max_close_r is None or close_r > state.max_close_r:
            state.max_close_r = close_r
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


def _build_daily_stop_trail_indicators(
    *,
    daily_ohlc: pd.DataFrame | None,
    stop_trail_rules: Sequence[StopTrailRule],
) -> _DailyStopTrailIndicators | None:
    if not stop_trail_rules:
        return None
    spans = sorted({int(rule.ema_span) for rule in stop_trail_rules})
    if daily_ohlc is None or daily_ohlc.empty:
        return _DailyStopTrailIndicators(
            session_day_ordinals=np.array([], dtype="int64"),
            ema_by_span={span: np.array([], dtype="float64") for span in spans},
            atr14_values=np.array([], dtype="float64"),
        )

    normalized_daily = _normalize_daily_ohlc(daily_ohlc)
    if normalized_daily.empty:
        return _DailyStopTrailIndicators(
            session_day_ordinals=np.array([], dtype="int64"),
            ema_by_span={span: np.array([], dtype="float64") for span in spans},
            atr14_values=np.array([], dtype="float64"),
        )

    close_values = normalized_daily["close"]
    ema_by_span: dict[int, np.ndarray] = {}
    for span in spans:
        ema_by_span[span] = (
            close_values.ewm(span=span, adjust=False, min_periods=span).mean().to_numpy(dtype="float64")
        )
    atr14_values = _atr14_series(
        high=normalized_daily["high"],
        low=normalized_daily["low"],
        close=close_values,
    ).to_numpy(dtype="float64")
    return _DailyStopTrailIndicators(
        session_day_ordinals=normalized_daily["session_day_ordinal"].to_numpy(dtype="int64"),
        ema_by_span=ema_by_span,
        atr14_values=atr14_values,
    )


def _normalize_daily_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["session_day_ordinal", "high", "low", "close"])

    out = frame.copy()
    date_series: pd.Series
    if "session_date" in out.columns:
        date_series = pd.to_datetime(out["session_date"], errors="coerce").dt.date
    elif "date" in out.columns:
        date_series = pd.to_datetime(out["date"], errors="coerce").dt.date
    elif "timestamp" in out.columns:
        date_series = pd.to_datetime(out["timestamp"], errors="coerce", utc=True).dt.date
    elif "ts" in out.columns:
        date_series = pd.to_datetime(out["ts"], errors="coerce", utc=True).dt.date
    else:
        date_series = pd.to_datetime(pd.Index(out.index), errors="coerce").to_series(index=out.index).dt.date

    normalized = pd.DataFrame(index=out.index)
    normalized["session_day_ordinal"] = date_series.map(
        lambda value: value.toordinal() if isinstance(value, date) else np.nan
    )
    for column in ("high", "low", "close"):
        normalized[column] = pd.to_numeric(out.get(column), errors="coerce")
    normalized = normalized.dropna(subset=["session_day_ordinal", "high", "low", "close"])
    if normalized.empty:
        return pd.DataFrame(columns=["session_day_ordinal", "high", "low", "close"])
    normalized = normalized.sort_values(by="session_day_ordinal", kind="stable")
    normalized["session_day_ordinal"] = normalized["session_day_ordinal"].astype("int64")
    # Keep the last candle for duplicate session dates to match prior-session semantics.
    return normalized.groupby("session_day_ordinal", sort=True, as_index=False).last()


def _atr14_series(*, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window=14, min_periods=14).mean()


def _apply_stop_trail_for_new_session(
    *,
    state: _TargetLoopState,
    inputs: _TargetSimulationInputs,
    bar_ts: pd.Timestamp,
    session_day_ordinal: int,
    trail_indicators: _DailyStopTrailIndicators,
    stop_trail_rules: Sequence[StopTrailRule],
) -> None:
    active_rule = _active_stop_trail_rule(
        stop_trail_rules=stop_trail_rules,
        max_close_r=state.max_close_r,
    )
    if active_rule is None:
        return

    trail_candidate = _stop_trail_candidate_for_session(
        session_day_ordinal=session_day_ordinal,
        direction=inputs.direction,
        rule=active_rule,
        trail_indicators=trail_indicators,
    )
    stage = _stop_trail_stage(active_rule)
    if trail_candidate is None:
        state.stop_updates.append(
            StrategyTradeStopUpdate(
                ts=bar_ts.to_pydatetime(),
                stop_price=float(state.stop_price),
                reason="stop_trail_missing_prior_session_indicator",
                stage=stage,
            )
        )
        return

    should_tighten = (
        trail_candidate > state.stop_price + _EPSILON
        if inputs.direction == "long"
        else trail_candidate < state.stop_price - _EPSILON
    )
    if not should_tighten:
        return

    state.stop_price = float(trail_candidate)
    state.stop_updates.append(
        StrategyTradeStopUpdate(
            ts=bar_ts.to_pydatetime(),
            stop_price=float(state.stop_price),
            reason="stop_trail_tightened",
            stage=stage,
        )
    )


def _active_stop_trail_rule(
    *,
    stop_trail_rules: Sequence[StopTrailRule],
    max_close_r: float | None,
) -> StopTrailRule | None:
    if max_close_r is None:
        return None
    active_rule: StopTrailRule | None = None
    for rule in stop_trail_rules:
        if max_close_r + _EPSILON < float(rule.start_r):
            break
        active_rule = rule
    return active_rule


def _stop_trail_candidate_for_session(
    *,
    session_day_ordinal: int,
    direction: str,
    rule: StopTrailRule,
    trail_indicators: _DailyStopTrailIndicators,
) -> float | None:
    prior_index = int(np.searchsorted(trail_indicators.session_day_ordinals, session_day_ordinal, side="left")) - 1
    if prior_index < 0:
        return None

    ema_values = trail_indicators.ema_by_span.get(int(rule.ema_span))
    if ema_values is None or prior_index >= int(ema_values.size):
        return None
    ema_value = float(ema_values[prior_index])
    if not math.isfinite(ema_value):
        return None

    atr_buffer = 0.0
    if rule.buffer_atr_multiple is not None:
        atr14_value = float(trail_indicators.atr14_values[prior_index])
        if not math.isfinite(atr14_value):
            return None
        atr_buffer = atr14_value * float(rule.buffer_atr_multiple)

    candidate = ema_value - atr_buffer if direction == "long" else ema_value + atr_buffer
    if not math.isfinite(candidate):
        return None
    return float(candidate)


def _session_day_ordinal(value: object, *, fallback_ts: pd.Timestamp) -> int:
    if value is None or pd.isna(value):
        return fallback_ts.date().toordinal()
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return fallback_ts.date().toordinal()
        return value.date().toordinal()
    if isinstance(value, datetime):
        return value.date().toordinal()
    if isinstance(value, date):
        return value.toordinal()
    try:
        parsed = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return fallback_ts.date().toordinal()
    if pd.isna(parsed):
        return fallback_ts.date().toordinal()
    return parsed.date().toordinal()


def _stop_trail_stage(rule: StopTrailRule) -> str:
    start_r = format(float(rule.start_r), "g")
    return f"start_{start_r}R_ema{int(rule.ema_span)}"


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
