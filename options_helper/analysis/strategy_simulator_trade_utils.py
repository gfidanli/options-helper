from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_helper.schemas.strategy_modeling_contracts import (
    StrategySignalEvent,
    StrategyTradeSimulation,
    TradeExitReason,
    TradeRejectCode,
)
from options_helper.schemas.strategy_modeling_policy import MaxHoldUnit, StopMoveRule


_EPSILON = 1e-12
_MARKET_TZ = ZoneInfo("America/New_York")


class _PreparedIntradayLike(Protocol):
    row_count: int
    timestamp_ns: np.ndarray
    session_days: np.ndarray
    close_values: np.ndarray


class _TargetLike(Protocol):
    label: str


class _TargetInputsLike(Protocol):
    direction: str
    initial_stop_price: float
    entry_ts: pd.Timestamp
    entry_price: float
    initial_risk: float
    target_price: float


class _TargetStateLike(Protocol):
    stop_price: float
    mae_r: float
    mfe_r: float
    gap_fill_applied: bool
    exit_ts: pd.Timestamp | None
    exit_price: float | None
    exit_reason: TradeExitReason | None
    holding_bars: int


def normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


def timestamp_from_ns(value: int) -> pd.Timestamp:
    return pd.Timestamp(value, unit="ns", tz="UTC")


def timestamp_to_datetime(value: pd.Timestamp) -> datetime:
    return value.to_pydatetime()


def target_price(*, direction: str, entry_price: float, risk: float, target_r: float) -> float:
    if direction == "long":
        return entry_price + (risk * target_r)
    return entry_price - (risk * target_r)


def open_hits_stop(*, direction: str, open_price: float, stop_price: float) -> bool:
    if direction == "long":
        return open_price <= stop_price + _EPSILON
    return open_price >= stop_price - _EPSILON


def open_hits_target(*, direction: str, open_price: float, target_price: float) -> bool:
    if direction == "long":
        return open_price >= target_price - _EPSILON
    return open_price <= target_price + _EPSILON


def intrabar_hits_stop(*, direction: str, low_price: float, high_price: float, stop_price: float) -> bool:
    if direction == "long":
        return low_price <= stop_price + _EPSILON
    return high_price >= stop_price - _EPSILON


def intrabar_hits_target(*, direction: str, low_price: float, high_price: float, target_price: float) -> bool:
    if direction == "long":
        return high_price >= target_price - _EPSILON
    return low_price <= target_price + _EPSILON


def update_excursions(
    *,
    prices: Sequence[float],
    direction: str,
    entry_price: float,
    initial_risk: float,
    current_mae_r: float,
    current_mfe_r: float,
) -> tuple[float, float]:
    mae_r = current_mae_r
    mfe_r = current_mfe_r
    for price in prices:
        r_value = price_to_r(
            direction=direction,
            entry_price=entry_price,
            initial_risk=initial_risk,
            price=price,
        )
        mae_r = min(mae_r, r_value)
        mfe_r = max(mfe_r, r_value)
    return mae_r, mfe_r


def price_to_r(*, direction: str, entry_price: float, initial_risk: float, price: float) -> float:
    if direction == "long":
        return (price - entry_price) / initial_risk
    return (entry_price - price) / initial_risk


def rejected_trade(
    *,
    event: StrategySignalEvent,
    target: _TargetLike,
    reject_code: TradeRejectCode,
    entry_ts: pd.Timestamp | None = None,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    initial_risk: float | None = None,
    mae_r: float | None = None,
    mfe_r: float | None = None,
) -> StrategyTradeSimulation:
    safe_entry_price = 0.0 if entry_price is None else float(entry_price)
    safe_initial_risk = 0.0 if initial_risk is None else float(initial_risk)

    return StrategyTradeSimulation(
        trade_id=_trade_id(event=event, target=target),
        event_id=event.event_id,
        strategy=event.strategy,
        symbol=normalize_symbol(event.symbol),
        direction=event.direction,
        signal_ts=event.signal_ts,
        signal_confirmed_ts=event.signal_confirmed_ts,
        entry_ts=event.entry_ts if entry_ts is None else timestamp_to_datetime(entry_ts),
        entry_price_source=event.entry_price_source,
        entry_price=safe_entry_price,
        stop_price=stop_price,
        stop_price_final=stop_price,
        target_price=target_price,
        exit_ts=None,
        exit_price=None,
        status="rejected",
        exit_reason=None,
        reject_code=reject_code,
        initial_risk=safe_initial_risk,
        realized_r=None,
        mae_r=mae_r,
        mfe_r=mfe_r,
        holding_bars=0,
        gap_fill_applied=False,
    )


def finalize_target_exit_or_reject(
    *,
    event: StrategySignalEvent,
    target: _TargetLike,
    prepared: _PreparedIntradayLike,
    inputs: _TargetInputsLike,
    state: _TargetStateLike,
    eval_end: int,
    eval_len: int,
    max_hold_bars: int | None,
    has_full_hold_coverage: bool,
) -> StrategyTradeSimulation | None:
    if state.exit_reason is not None and state.exit_price is not None and state.exit_ts is not None:
        return None
    if max_hold_bars is not None and not has_full_hold_coverage:
        return rejected_trade(
            event=event,
            target=target,
            reject_code="insufficient_future_bars",
            entry_ts=inputs.entry_ts,
            entry_price=inputs.entry_price,
            stop_price=inputs.initial_stop_price,
            target_price=inputs.target_price,
            initial_risk=inputs.initial_risk,
            mae_r=state.mae_r,
            mfe_r=state.mfe_r,
        )
    final_row_index = int(eval_end - 1)
    state.exit_reason = "time_stop"
    state.exit_ts = timestamp_from_ns(int(prepared.timestamp_ns[final_row_index]))
    state.exit_price = float(prepared.close_values[final_row_index])
    state.holding_bars = eval_len
    return None


def build_closed_trade(
    *,
    event: StrategySignalEvent,
    target: _TargetLike,
    inputs: _TargetInputsLike,
    state: _TargetStateLike,
) -> StrategyTradeSimulation:
    assert state.exit_price is not None
    assert state.exit_reason is not None
    assert state.exit_ts is not None
    realized_r = price_to_r(
        direction=inputs.direction,
        entry_price=inputs.entry_price,
        initial_risk=inputs.initial_risk,
        price=state.exit_price,
    )
    return StrategyTradeSimulation(
        trade_id=_trade_id(event=event, target=target),
        event_id=event.event_id,
        strategy=event.strategy,
        symbol=normalize_symbol(event.symbol),  # type: ignore[arg-type]
        direction=inputs.direction,  # type: ignore[arg-type]
        signal_ts=event.signal_ts,
        signal_confirmed_ts=event.signal_confirmed_ts,
        entry_ts=timestamp_to_datetime(inputs.entry_ts),
        entry_price_source=event.entry_price_source,
        entry_price=inputs.entry_price,
        stop_price=float(inputs.initial_stop_price),
        stop_price_final=float(state.stop_price),
        target_price=inputs.target_price,
        exit_ts=timestamp_to_datetime(state.exit_ts),
        exit_price=state.exit_price,
        status="closed",
        exit_reason=state.exit_reason,
        reject_code=None,
        initial_risk=inputs.initial_risk,
        realized_r=realized_r,
        mae_r=state.mae_r,
        mfe_r=state.mfe_r,
        holding_bars=int(state.holding_bars),
        gap_fill_applied=state.gap_fill_applied,
    )


def apply_stop_move_rules(
    *,
    stop_move_rules: tuple[StopMoveRule, ...],
    stop_move_index: int,
    direction: str,
    entry_price: float,
    initial_risk: float,
    close_price: float,
    stop_price: float,
) -> tuple[float, int]:
    close_r = price_to_r(
        direction=direction,
        entry_price=entry_price,
        initial_risk=initial_risk,
        price=close_price,
    )

    updated_stop = float(stop_price)
    idx = int(stop_move_index)
    while idx < len(stop_move_rules) and close_r + _EPSILON >= float(stop_move_rules[idx].trigger_r):
        candidate_stop = _stop_price_from_r(
            direction=direction,
            entry_price=entry_price,
            initial_risk=initial_risk,
            stop_r=float(stop_move_rules[idx].stop_r),
        )
        if direction == "long":
            candidate_stop = min(candidate_stop, close_price)
            if candidate_stop > updated_stop + _EPSILON:
                updated_stop = candidate_stop
        else:
            candidate_stop = max(candidate_stop, close_price)
            if candidate_stop < updated_stop - _EPSILON:
                updated_stop = candidate_stop
        idx += 1

    return updated_stop, idx


def evaluation_end_index(
    *,
    prepared: _PreparedIntradayLike,
    entry_row_index: int,
    entry_ts: pd.Timestamp,
    max_hold_bars: int | None,
    max_hold_timeframe: tuple[MaxHoldUnit, int],
) -> tuple[int, bool]:
    if max_hold_bars is not None:
        unit, unit_size = max_hold_timeframe
        if unit == "entry":
            end = min(prepared.row_count, int(entry_row_index + max_hold_bars))
            return end, int(end - entry_row_index) >= int(max_hold_bars)

        if unit in {"min", "h"}:
            horizon_minutes = int(max_hold_bars * unit_size)
            if unit == "h":
                horizon_minutes *= 60
            cutoff = entry_ts + pd.Timedelta(minutes=horizon_minutes)
            cutoff_ns = int(cutoff.value)
            end = int(np.searchsorted(prepared.timestamp_ns, cutoff_ns, side="left"))
            end = min(prepared.row_count, end)
            has_boundary = (
                end < prepared.row_count
                or (
                    prepared.row_count > 0
                    and int(prepared.timestamp_ns[prepared.row_count - 1]) >= cutoff_ns
                )
            )
            return end, has_boundary

        required_sessions = int(max_hold_bars * unit_size)
        if unit == "w":
            required_sessions *= 5
        entry_session_day = _session_day_value(
            prepared.session_days[entry_row_index],
            fallback=entry_ts.tz_convert(_MARKET_TZ).date(),
        )
        sessions_seen = 1
        end = int(entry_row_index + 1)
        while end < prepared.row_count:
            session_day = _session_day_value(prepared.session_days[end], fallback=entry_session_day)
            if session_day != entry_session_day:
                entry_session_day = session_day
                sessions_seen += 1
                if sessions_seen > required_sessions:
                    break
            end += 1
        return end, sessions_seen >= required_sessions

    entry_session_date = prepared.session_days[entry_row_index]
    if entry_session_date is None or pd.isna(entry_session_date):
        entry_session_date = entry_ts.tz_convert(_MARKET_TZ).date()

    end = int(entry_row_index + 1)
    while end < prepared.row_count:
        session_day = prepared.session_days[end]
        if session_day != entry_session_date:
            break
        end += 1
    return end, True


def _trade_id(*, event: StrategySignalEvent, target: _TargetLike) -> str:
    return f"{event.event_id}:{target.label}"


def _stop_price_from_r(*, direction: str, entry_price: float, initial_risk: float, stop_r: float) -> float:
    if direction == "long":
        return entry_price + (initial_risk * stop_r)
    return entry_price - (initial_risk * stop_r)


def _session_day_value(value: object, *, fallback: object) -> object:
    if value is None or pd.isna(value):
        return fallback
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return fallback
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    return value


__all__ = [
    "apply_stop_move_rules",
    "build_closed_trade",
    "evaluation_end_index",
    "finalize_target_exit_or_reject",
    "intrabar_hits_stop",
    "intrabar_hits_target",
    "normalize_symbol",
    "open_hits_stop",
    "open_hits_target",
    "price_to_r",
    "rejected_trade",
    "target_price",
    "timestamp_from_ns",
    "timestamp_to_datetime",
    "update_excursions",
]
