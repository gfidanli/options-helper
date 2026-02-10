from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_helper.analysis.strategy_modeling_contracts import parse_strategy_signal_events
from options_helper.schemas.strategy_modeling_contracts import (
    StrategySignalEvent,
    StrategyTradeSimulation,
    TradeExitReason,
    TradeRejectCode,
)
from options_helper.schemas.strategy_modeling_policy import (
    GapFillPolicy,
    MaxHoldUnit,
    StopMoveRule,
    StrategyModelingPolicyConfig,
    parse_max_hold_timeframe,
)


_INTRADAY_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "session_date")
_DEFAULT_MIN_TARGET_TENTHS = 10
_DEFAULT_MAX_TARGET_TENTHS = 20
_DEFAULT_STEP_TENTHS = 1
_EPSILON = 1e-12
_MARKET_TZ = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class StrategyRTarget:
    label: str
    target_r: float
    target_tenths: int


@dataclass(frozen=True)
class _PreparedIntraday:
    frame: pd.DataFrame
    timestamp_ns: np.ndarray
    session_days: np.ndarray
    regular_open_mask: np.ndarray
    open_values: np.ndarray
    high_values: np.ndarray
    low_values: np.ndarray
    close_values: np.ndarray

    @property
    def row_count(self) -> int:
        return int(self.timestamp_ns.size)


@dataclass(frozen=True)
class _EntryDecision:
    reject_code: TradeRejectCode | None
    direction: str | None = None
    stop_price: float | None = None
    entry_row_index: int | None = None
    entry_ts: pd.Timestamp | None = None
    entry_price: float | None = None
    initial_risk: float | None = None


def build_r_target_ladder(
    *,
    min_target_tenths: int = _DEFAULT_MIN_TARGET_TENTHS,
    max_target_tenths: int = _DEFAULT_MAX_TARGET_TENTHS,
    step_tenths: int = _DEFAULT_STEP_TENTHS,
) -> tuple[StrategyRTarget, ...]:
    """Build stable R targets (for example 1.0R..2.0R) using integer tenths."""

    min_t = int(min_target_tenths)
    max_t = int(max_target_tenths)
    step_t = int(step_tenths)

    if step_t <= 0:
        raise ValueError("step_tenths must be > 0")
    if min_t <= 0:
        raise ValueError("min_target_tenths must be > 0")
    if max_t < min_t:
        raise ValueError("max_target_tenths must be >= min_target_tenths")

    out: list[StrategyRTarget] = []
    for tenths in range(min_t, max_t + 1, step_t):
        label = f"{tenths // 10}.{tenths % 10}R"
        out.append(StrategyRTarget(label=label, target_r=tenths / 10.0, target_tenths=tenths))
    return tuple(out)


def simulate_strategy_trade_paths(
    events: Iterable[Mapping[str, Any] | StrategySignalEvent],
    intraday_bars_by_symbol: Mapping[str, pd.DataFrame],
    *,
    policy: StrategyModelingPolicyConfig | None = None,
    max_hold_bars: int | None = None,
    max_hold_timeframe: str | None = None,
    target_ladder: Sequence[StrategyRTarget] | None = None,
) -> list[StrategyTradeSimulation]:
    """Simulate one path per (event, target) with deterministic reject/exit semantics."""

    cfg = policy or StrategyModelingPolicyConfig()
    max_hold: int | None = max_hold_bars if max_hold_bars is not None else cfg.max_hold_bars
    if max_hold is not None:
        max_hold = int(max_hold)
        if max_hold < 1:
            raise ValueError("max_hold_bars must be >= 1")
    hold_timeframe = max_hold_timeframe if max_hold_timeframe is not None else cfg.max_hold_timeframe
    parsed_hold_timeframe = parse_max_hold_timeframe(hold_timeframe)
    if cfg.gap_fill_policy != "fill_at_open":
        raise ValueError(f"Unsupported gap_fill_policy: {cfg.gap_fill_policy}")

    ladder = tuple(target_ladder) if target_ladder is not None else build_r_target_ladder()
    if not ladder:
        raise ValueError("target_ladder cannot be empty")

    normalized_intraday = _normalize_intraday_by_symbol(intraday_bars_by_symbol)
    prepared_intraday = _prepare_intraday_by_symbol(normalized_intraday)
    parsed_events = parse_strategy_signal_events(events)
    sorted_events = sorted(parsed_events, key=_event_sort_key)

    entry_decisions: dict[str, _EntryDecision] = {}
    trades: list[StrategyTradeSimulation] = []
    for event in sorted_events:
        symbol = _normalize_symbol(event.symbol)
        prepared = prepared_intraday.get(symbol)
        decision = entry_decisions.get(event.event_id)
        if decision is None:
            decision = _resolve_entry_decision(event=event, prepared=prepared)
            entry_decisions[event.event_id] = decision
        for target in ladder:
            trades.append(
                _simulate_one_target(
                    event=event,
                    prepared=prepared,
                    entry_decision=decision,
                    target=target,
                    max_hold_bars=max_hold,
                    max_hold_timeframe=parsed_hold_timeframe,
                    gap_fill_policy=cfg.gap_fill_policy,
                    stop_move_rules=cfg.stop_move_rules,
                )
            )
    return trades


def _simulate_one_target(
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
        return _rejected_trade(
            event=event,
            target=target,
            reject_code=entry_decision.reject_code,
            entry_ts=entry_decision.entry_ts,
            entry_price=entry_decision.entry_price,
            stop_price=entry_decision.stop_price,
            initial_risk=entry_decision.initial_risk,
        )

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
        return _rejected_trade(
            event=event,
            target=target,
            reject_code="invalid_signal",
        )

    target_price = _target_price(
        direction=direction,
        entry_price=entry_price,
        risk=initial_risk,
        target_r=target.target_r,
    )

    eval_start = int(entry_row_index)
    eval_end, has_full_hold_coverage = _evaluation_end_index(
        prepared=prepared,
        entry_row_index=entry_row_index,
        entry_ts=entry_ts,
        max_hold_bars=max_hold_bars,
        max_hold_timeframe=max_hold_timeframe,
    )
    eval_len = int(eval_end - eval_start)
    if eval_len <= 0:
        return _rejected_trade(
            event=event,
            target=target,
            reject_code="insufficient_future_bars",
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_price=initial_stop_price,
            target_price=target_price,
            initial_risk=initial_risk,
        )

    stop_price = float(initial_stop_price)
    stop_move_index = 0

    mae_r = 0.0
    mfe_r = 0.0
    gap_fill_applied = False
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: TradeExitReason | None = None
    holding_bars = 0

    for offset, row_index in enumerate(range(eval_start, eval_end), start=1):
        holding_bars = offset
        open_price = float(prepared.open_values[row_index])
        high_price = float(prepared.high_values[row_index])
        low_price = float(prepared.low_values[row_index])
        close_price = float(prepared.close_values[row_index])

        stop_at_open = _open_hits_stop(direction=direction, open_price=open_price, stop_price=stop_price)
        target_at_open = _open_hits_target(direction=direction, open_price=open_price, target_price=target_price)
        if stop_at_open or target_at_open:
            bar_ts = _timestamp_from_ns(int(prepared.timestamp_ns[row_index]))
            if stop_at_open:
                exit_reason = "stop_hit"
                exit_price = open_price
                gap_fill_applied = gap_fill_policy == "fill_at_open" and abs(open_price - stop_price) > _EPSILON
            else:
                exit_reason = "target_hit"
                exit_price = open_price
                gap_fill_applied = gap_fill_policy == "fill_at_open" and abs(open_price - target_price) > _EPSILON
            exit_ts = bar_ts
            mae_r, mfe_r = _update_excursions(
                prices=(open_price, exit_price),
                direction=direction,
                entry_price=entry_price,
                initial_risk=initial_risk,
                current_mae_r=mae_r,
                current_mfe_r=mfe_r,
            )
            break

        stop_touched = _intrabar_hits_stop(direction=direction, low_price=low_price, high_price=high_price, stop_price=stop_price)
        target_touched = _intrabar_hits_target(
            direction=direction,
            low_price=low_price,
            high_price=high_price,
            target_price=target_price,
        )
        if stop_touched and target_touched:
            # Conservative tie-break: treat stop as hit first.
            bar_ts = _timestamp_from_ns(int(prepared.timestamp_ns[row_index]))
            exit_reason = "stop_hit"
            exit_price = stop_price
            exit_ts = bar_ts
            mae_r, mfe_r = _update_excursions(
                prices=(open_price, stop_price),
                direction=direction,
                entry_price=entry_price,
                initial_risk=initial_risk,
                current_mae_r=mae_r,
                current_mfe_r=mfe_r,
            )
            break
        if stop_touched:
            bar_ts = _timestamp_from_ns(int(prepared.timestamp_ns[row_index]))
            exit_reason = "stop_hit"
            exit_price = stop_price
            exit_ts = bar_ts
            mae_r, mfe_r = _update_excursions(
                prices=(open_price, stop_price),
                direction=direction,
                entry_price=entry_price,
                initial_risk=initial_risk,
                current_mae_r=mae_r,
                current_mfe_r=mfe_r,
            )
            break
        if target_touched:
            bar_ts = _timestamp_from_ns(int(prepared.timestamp_ns[row_index]))
            exit_reason = "target_hit"
            exit_price = target_price
            exit_ts = bar_ts
            mae_r, mfe_r = _update_excursions(
                prices=(open_price, target_price),
                direction=direction,
                entry_price=entry_price,
                initial_risk=initial_risk,
                current_mae_r=mae_r,
                current_mfe_r=mfe_r,
            )
            break

        mae_r, mfe_r = _update_excursions(
            prices=(open_price, high_price, low_price, close_price),
            direction=direction,
            entry_price=entry_price,
            initial_risk=initial_risk,
            current_mae_r=mae_r,
            current_mfe_r=mfe_r,
        )

        if stop_move_rules and row_index + 1 < eval_end and stop_move_index < len(stop_move_rules):
            stop_price, stop_move_index = _apply_stop_move_rules(
                stop_move_rules=stop_move_rules,
                stop_move_index=stop_move_index,
                direction=direction,
                entry_price=entry_price,
                initial_risk=initial_risk,
                close_price=close_price,
                stop_price=stop_price,
            )

    if exit_reason is None or exit_price is None or exit_ts is None:
        if max_hold_bars is not None and not has_full_hold_coverage:
            return _rejected_trade(
                event=event,
                target=target,
                reject_code="insufficient_future_bars",
                entry_ts=entry_ts,
                entry_price=entry_price,
                stop_price=initial_stop_price,
                target_price=target_price,
                initial_risk=initial_risk,
                mae_r=mae_r,
                mfe_r=mfe_r,
            )

        final_row_index = int(eval_end - 1)
        final_ts = _timestamp_from_ns(int(prepared.timestamp_ns[final_row_index]))
        final_close = float(prepared.close_values[final_row_index])
        exit_reason = "time_stop"
        exit_ts = final_ts
        exit_price = final_close
        holding_bars = eval_len

    realized_r = _price_to_r(
        direction=direction,
        entry_price=entry_price,
        initial_risk=initial_risk,
        price=exit_price,
    )

    return StrategyTradeSimulation(
        trade_id=_trade_id(event=event, target=target),
        event_id=event.event_id,
        strategy=event.strategy,
        symbol=_normalize_symbol(event.symbol),
        direction=direction,  # type: ignore[arg-type]
        signal_ts=event.signal_ts,
        signal_confirmed_ts=event.signal_confirmed_ts,
        entry_ts=_timestamp_to_datetime(entry_ts),
        entry_price_source=event.entry_price_source,
        entry_price=entry_price,
        stop_price=float(initial_stop_price),
        stop_price_final=float(stop_price),
        target_price=target_price,
        exit_ts=_timestamp_to_datetime(exit_ts),
        exit_price=exit_price,
        status="closed",
        exit_reason=exit_reason,
        reject_code=None,
        initial_risk=initial_risk,
        realized_r=realized_r,
        mae_r=mae_r,
        mfe_r=mfe_r,
        holding_bars=int(holding_bars),
        gap_fill_applied=gap_fill_applied,
    )


def _normalize_intraday_by_symbol(
    intraday_bars_by_symbol: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for raw_symbol in sorted(intraday_bars_by_symbol):
        symbol = _normalize_symbol(raw_symbol)
        out[symbol] = _normalize_intraday_frame(intraday_bars_by_symbol[raw_symbol])
    return out


def _prepare_intraday_by_symbol(
    intraday_by_symbol: Mapping[str, pd.DataFrame],
) -> dict[str, _PreparedIntraday]:
    out: dict[str, _PreparedIntraday] = {}
    for symbol in sorted(intraday_by_symbol):
        out[symbol] = _prepare_intraday_frame(intraday_by_symbol[symbol])
    return out


def _prepare_intraday_frame(frame: pd.DataFrame) -> _PreparedIntraday:
    if frame.empty:
        empty_float = np.array([], dtype="float64")
        empty_int = np.array([], dtype="int64")
        empty_obj = np.array([], dtype="object")
        empty_bool = np.array([], dtype="bool")
        return _PreparedIntraday(
            frame=frame,
            timestamp_ns=empty_int,
            session_days=empty_obj,
            regular_open_mask=empty_bool,
            open_values=empty_float,
            high_values=empty_float,
            low_values=empty_float,
            close_values=empty_float,
        )

    ts = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    timestamp_ns = ts.astype("int64").to_numpy(copy=False)

    session_series = pd.Series(frame["session_date"], index=frame.index)
    missing_mask = pd.isna(session_series)
    if bool(missing_mask.any()):
        session_series = session_series.where(~missing_mask, ts.dt.date)
    session_days = session_series.to_numpy(copy=False)

    market_ts = ts.dt.tz_convert(_MARKET_TZ)
    regular_open_mask = (
        market_ts.dt.hour.eq(9).to_numpy(copy=False)
        & market_ts.dt.minute.eq(30).to_numpy(copy=False)
    )

    return _PreparedIntraday(
        frame=frame,
        timestamp_ns=timestamp_ns,
        session_days=session_days,
        regular_open_mask=np.asarray(regular_open_mask, dtype="bool"),
        open_values=frame["open"].to_numpy(dtype="float64", copy=False),
        high_values=frame["high"].to_numpy(dtype="float64", copy=False),
        low_values=frame["low"].to_numpy(dtype="float64", copy=False),
        close_values=frame["close"].to_numpy(dtype="float64", copy=False),
    )


def _normalize_intraday_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out = frame.copy()
    ts_column = "timestamp" if "timestamp" in out.columns else "ts" if "ts" in out.columns else None
    if ts_column is None:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out["timestamp"] = pd.to_datetime(out[ts_column], errors="coerce", utc=True)
    if "session_date" in out.columns:
        out["session_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date
    else:
        out["session_date"] = out["timestamp"].dt.date
    for column in ("open", "high", "low", "close"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = pd.Series([None] * len(out), index=out.index)
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out = out.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return out.reindex(columns=list(_INTRADAY_COLUMNS))


def _event_sort_key(event: StrategySignalEvent) -> tuple[int, int, int, str, str]:
    return (
        _timestamp_sort_key(event.signal_confirmed_ts),
        _timestamp_sort_key(event.signal_ts),
        _timestamp_sort_key(event.entry_ts),
        _normalize_symbol(event.symbol),
        event.event_id,
    )


def _timestamp_sort_key(value: object) -> int:
    ts = _to_utc_timestamp(value)
    if ts is None:
        return -1
    return int(ts.value)


def _trade_id(*, event: StrategySignalEvent, target: StrategyRTarget) -> str:
    return f"{event.event_id}:{target.label}"


def _target_price(*, direction: str, entry_price: float, risk: float, target_r: float) -> float:
    if direction == "long":
        return entry_price + (risk * target_r)
    return entry_price - (risk * target_r)


def _open_hits_stop(*, direction: str, open_price: float, stop_price: float) -> bool:
    if direction == "long":
        return open_price <= stop_price + _EPSILON
    return open_price >= stop_price - _EPSILON


def _open_hits_target(*, direction: str, open_price: float, target_price: float) -> bool:
    if direction == "long":
        return open_price >= target_price - _EPSILON
    return open_price <= target_price + _EPSILON


def _intrabar_hits_stop(*, direction: str, low_price: float, high_price: float, stop_price: float) -> bool:
    if direction == "long":
        return low_price <= stop_price + _EPSILON
    return high_price >= stop_price - _EPSILON


def _intrabar_hits_target(*, direction: str, low_price: float, high_price: float, target_price: float) -> bool:
    if direction == "long":
        return high_price >= target_price - _EPSILON
    return low_price <= target_price + _EPSILON


def _update_excursions(
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
        r_value = _price_to_r(
            direction=direction,
            entry_price=entry_price,
            initial_risk=initial_risk,
            price=price,
        )
        mae_r = min(mae_r, r_value)
        mfe_r = max(mfe_r, r_value)
    return mae_r, mfe_r


def _price_to_r(*, direction: str, entry_price: float, initial_risk: float, price: float) -> float:
    if direction == "long":
        return (price - entry_price) / initial_risk
    return (entry_price - price) / initial_risk


def _rejected_trade(
    *,
    event: StrategySignalEvent,
    target: StrategyRTarget,
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
        symbol=_normalize_symbol(event.symbol),
        direction=event.direction,
        signal_ts=event.signal_ts,
        signal_confirmed_ts=event.signal_confirmed_ts,
        entry_ts=event.entry_ts if entry_ts is None else _timestamp_to_datetime(entry_ts),
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


def _stop_price_from_r(*, direction: str, entry_price: float, initial_risk: float, stop_r: float) -> float:
    if direction == "long":
        return entry_price + (initial_risk * stop_r)
    return entry_price - (initial_risk * stop_r)


def _apply_stop_move_rules(
    *,
    stop_move_rules: tuple[StopMoveRule, ...],
    stop_move_index: int,
    direction: str,
    entry_price: float,
    initial_risk: float,
    close_price: float,
    stop_price: float,
) -> tuple[float, int]:
    close_r = _price_to_r(
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


def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:  # noqa: BLE001
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_entry_decision(
    *,
    event: StrategySignalEvent,
    prepared: _PreparedIntraday | None,
) -> _EntryDecision:
    direction = str(event.direction).strip().lower()
    if direction not in {"long", "short"}:
        return _EntryDecision(reject_code="invalid_signal")

    stop_price = _finite_float(event.stop_price)
    signal_confirmed_ts = _to_utc_timestamp(event.signal_confirmed_ts)
    entry_anchor_ts = _to_utc_timestamp(event.entry_ts)
    if stop_price is None or signal_confirmed_ts is None or entry_anchor_ts is None:
        return _EntryDecision(reject_code="invalid_signal")
    if entry_anchor_ts <= signal_confirmed_ts:
        return _EntryDecision(reject_code="invalid_signal")

    if prepared is None or prepared.row_count == 0:
        return _EntryDecision(reject_code="missing_intraday_coverage")

    entry_cutoff = max(entry_anchor_ts, signal_confirmed_ts + pd.Timedelta(microseconds=1))
    entry_row_index = _select_entry_row_index(prepared, entry_cutoff=entry_cutoff)
    if entry_row_index is None:
        return _EntryDecision(reject_code="missing_entry_bar")

    entry_ts = _timestamp_from_ns(int(prepared.timestamp_ns[entry_row_index]))
    entry_price = float(prepared.open_values[entry_row_index])

    signal_low = _finite_float(event.signal_low)
    signal_high = _finite_float(event.signal_high)
    if direction == "long" and signal_low is not None and entry_price < (signal_low - _EPSILON):
        return _EntryDecision(
            reject_code="entry_open_outside_signal_range",
            stop_price=stop_price,
            entry_row_index=entry_row_index,
            entry_ts=entry_ts,
            entry_price=entry_price,
        )
    if direction == "short" and signal_high is not None and entry_price > (signal_high + _EPSILON):
        return _EntryDecision(
            reject_code="entry_open_outside_signal_range",
            stop_price=stop_price,
            entry_row_index=entry_row_index,
            entry_ts=entry_ts,
            entry_price=entry_price,
        )

    if direction == "long":
        initial_risk = entry_price - stop_price
    else:
        initial_risk = stop_price - entry_price
    if initial_risk <= 0.0:
        return _EntryDecision(
            reject_code="non_positive_risk",
            stop_price=stop_price,
            entry_row_index=entry_row_index,
            entry_ts=entry_ts,
            entry_price=entry_price,
            initial_risk=initial_risk,
        )

    return _EntryDecision(
        reject_code=None,
        direction=direction,
        stop_price=stop_price,
        entry_row_index=entry_row_index,
        entry_ts=entry_ts,
        entry_price=entry_price,
        initial_risk=initial_risk,
    )


def _select_entry_row_index(
    prepared: _PreparedIntraday,
    *,
    entry_cutoff: pd.Timestamp,
) -> int | None:
    """Pick next regular open after cutoff; fallback to first candidate at/after cutoff."""
    row_count = prepared.row_count
    if row_count <= 0:
        return None

    start = int(np.searchsorted(prepared.timestamp_ns, int(entry_cutoff.value), side="left"))
    if start >= row_count:
        return None

    anchor_day = entry_cutoff.date()
    eligible_start = start
    while eligible_start < row_count:
        session_day = prepared.session_days[eligible_start]
        if session_day is not None and not pd.isna(session_day) and session_day >= anchor_day:
            break
        eligible_start += 1
    if eligible_start >= row_count:
        eligible_start = start

    open_offsets = np.flatnonzero(prepared.regular_open_mask[eligible_start:])
    if open_offsets.size > 0:
        return int(eligible_start + int(open_offsets[0]))
    return int(eligible_start)


def _evaluation_end_index(
    *,
    prepared: _PreparedIntraday,
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


def _timestamp_from_ns(value: int) -> pd.Timestamp:
    return pd.Timestamp(value, unit="ns", tz="UTC")


def _timestamp_to_datetime(value: pd.Timestamp) -> datetime:
    return value.to_pydatetime()


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not pd.notna(number) or not math.isfinite(number):
        return None
    return number


def _normalize_symbol(value: object) -> str:
    return str(value or "").strip().upper()


__all__ = [
    "StrategyRTarget",
    "build_r_target_ladder",
    "simulate_strategy_trade_paths",
]
