from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from options_helper.analysis.strategy_modeling_contracts import parse_strategy_signal_events
from options_helper.schemas.strategy_modeling_contracts import (
    StrategySignalEvent,
    StrategyTradeSimulation,
    TradeExitReason,
    TradeRejectCode,
)
from options_helper.schemas.strategy_modeling_policy import GapFillPolicy, StrategyModelingPolicyConfig


_INTRADAY_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close")
_DEFAULT_MIN_TARGET_TENTHS = 10
_DEFAULT_MAX_TARGET_TENTHS = 20
_DEFAULT_STEP_TENTHS = 1
_EPSILON = 1e-12


@dataclass(frozen=True)
class StrategyRTarget:
    label: str
    target_r: float
    target_tenths: int


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
    target_ladder: Sequence[StrategyRTarget] | None = None,
) -> list[StrategyTradeSimulation]:
    """Simulate one path per (event, target) with deterministic reject/exit semantics."""

    cfg = policy or StrategyModelingPolicyConfig()
    max_hold = int(max_hold_bars if max_hold_bars is not None else cfg.max_hold_bars)
    if max_hold < 1:
        raise ValueError("max_hold_bars must be >= 1")
    if cfg.gap_fill_policy != "fill_at_open":
        raise ValueError(f"Unsupported gap_fill_policy: {cfg.gap_fill_policy}")

    ladder = tuple(target_ladder) if target_ladder is not None else build_r_target_ladder()
    if not ladder:
        raise ValueError("target_ladder cannot be empty")

    normalized_intraday = _normalize_intraday_by_symbol(intraday_bars_by_symbol)
    parsed_events = parse_strategy_signal_events(events)
    sorted_events = sorted(parsed_events, key=_event_sort_key)

    trades: list[StrategyTradeSimulation] = []
    for event in sorted_events:
        symbol = _normalize_symbol(event.symbol)
        symbol_bars = normalized_intraday.get(symbol)
        for target in ladder:
            trades.append(
                _simulate_one_target(
                    event=event,
                    bars=symbol_bars,
                    target=target,
                    max_hold_bars=max_hold,
                    gap_fill_policy=cfg.gap_fill_policy,
                )
            )
    return trades


def _simulate_one_target(
    *,
    event: StrategySignalEvent,
    bars: pd.DataFrame | None,
    target: StrategyRTarget,
    max_hold_bars: int,
    gap_fill_policy: GapFillPolicy,
) -> StrategyTradeSimulation:
    direction = str(event.direction).strip().lower()
    if direction not in {"long", "short"}:
        return _rejected_trade(event=event, target=target, reject_code="invalid_signal")

    stop_price = _finite_float(event.stop_price)
    signal_confirmed_ts = _to_utc_timestamp(event.signal_confirmed_ts)
    entry_anchor_ts = _to_utc_timestamp(event.entry_ts)
    if stop_price is None or signal_confirmed_ts is None or entry_anchor_ts is None:
        return _rejected_trade(event=event, target=target, reject_code="invalid_signal")
    if entry_anchor_ts <= signal_confirmed_ts:
        return _rejected_trade(event=event, target=target, reject_code="invalid_signal")

    if bars is None or bars.empty:
        return _rejected_trade(event=event, target=target, reject_code="missing_intraday_coverage")

    entry_cutoff = max(entry_anchor_ts, signal_confirmed_ts + pd.Timedelta(microseconds=1))
    entry_candidates = bars.loc[bars["timestamp"] >= entry_cutoff]
    if entry_candidates.empty:
        return _rejected_trade(event=event, target=target, reject_code="missing_entry_bar")

    entry_row = entry_candidates.iloc[0]
    entry_row_index = int(entry_row.name)
    entry_ts = _to_utc_timestamp(entry_row["timestamp"])
    entry_price = _finite_float(entry_row["open"])
    if entry_ts is None or entry_price is None:
        return _rejected_trade(event=event, target=target, reject_code="missing_entry_bar")

    if direction == "long":
        initial_risk = entry_price - stop_price
    else:
        initial_risk = stop_price - entry_price
    if initial_risk <= 0.0:
        return _rejected_trade(
            event=event,
            target=target,
            reject_code="non_positive_risk",
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_price=stop_price,
            initial_risk=initial_risk,
        )

    target_price = _target_price(direction=direction, entry_price=entry_price, risk=initial_risk, target_r=target.target_r)

    eval_bars = bars.iloc[entry_row_index : entry_row_index + max_hold_bars]
    if eval_bars.empty:
        return _rejected_trade(
            event=event,
            target=target,
            reject_code="insufficient_future_bars",
            entry_ts=entry_ts,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            initial_risk=initial_risk,
        )

    mae_r = 0.0
    mfe_r = 0.0
    gap_fill_applied = False
    exit_ts: pd.Timestamp | None = None
    exit_price: float | None = None
    exit_reason: TradeExitReason | None = None
    holding_bars = 0

    for holding_bars, (_, row) in enumerate(eval_bars.iterrows(), start=1):
        bar_ts = _to_utc_timestamp(row["timestamp"])
        open_price = _finite_float(row["open"])
        high_price = _finite_float(row["high"])
        low_price = _finite_float(row["low"])
        close_price = _finite_float(row["close"])
        if (
            bar_ts is None
            or open_price is None
            or high_price is None
            or low_price is None
            or close_price is None
        ):
            continue

        stop_at_open = _open_hits_stop(direction=direction, open_price=open_price, stop_price=stop_price)
        target_at_open = _open_hits_target(direction=direction, open_price=open_price, target_price=target_price)
        if stop_at_open or target_at_open:
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

    if exit_reason is None or exit_price is None or exit_ts is None:
        if len(eval_bars) < max_hold_bars:
            return _rejected_trade(
                event=event,
                target=target,
                reject_code="insufficient_future_bars",
                entry_ts=entry_ts,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                initial_risk=initial_risk,
                mae_r=mae_r,
                mfe_r=mfe_r,
            )

        final_bar = eval_bars.iloc[-1]
        final_ts = _to_utc_timestamp(final_bar["timestamp"])
        final_close = _finite_float(final_bar["close"])
        if final_ts is None or final_close is None:
            return _rejected_trade(
                event=event,
                target=target,
                reject_code="insufficient_future_bars",
                entry_ts=entry_ts,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                initial_risk=initial_risk,
                mae_r=mae_r,
                mfe_r=mfe_r,
            )
        exit_reason = "time_stop"
        exit_ts = final_ts
        exit_price = final_close
        holding_bars = max_hold_bars

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
        stop_price=stop_price,
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


def _normalize_intraday_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out = frame.copy()
    ts_column = "timestamp" if "timestamp" in out.columns else "ts" if "ts" in out.columns else None
    if ts_column is None:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out["timestamp"] = pd.to_datetime(out[ts_column], errors="coerce", utc=True)
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
