from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from options_helper.analysis.orb import normalize_intraday_ohlc_frame
from options_helper.analysis.strategy_features import StrategyFeatureConfig, compute_ema_slope
from options_helper.analysis.strategy_modeling_contracts import parse_strategy_signal_events
from options_helper.analysis.strategy_modeling_io_adapter import normalize_symbol
from options_helper.schemas.strategy_modeling_contracts import StrategySignalEvent
from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig

FILTER_REJECT_REASONS: tuple[str, ...] = (
    "shorts_disabled",
    "missing_daily_context",
    "rsi_not_extreme",
    "ema9_regime_mismatch",
    "volatility_regime_disallowed",
    "orb_opening_range_missing",
    "orb_breakout_missing",
    "atr_floor_failed",
)

_MARKET_TZ = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_CONFIRMATION_FLOOR = pd.Timedelta(microseconds=1)


@dataclass(frozen=True)
class _PreparedIntraday:
    frame: pd.DataFrame
    timestamp_ns: np.ndarray
    session_days: np.ndarray
    open_values: np.ndarray


@dataclass(frozen=True)
class _DailyContextIndex:
    ordered_dates: tuple[date, ...]
    rows_by_date: dict[date, dict[str, Any]]


@dataclass(frozen=True)
class _OrbBreakoutDecision:
    signal_confirmed_ts: pd.Timestamp
    entry_ts: pd.Timestamp
    stop_price: float


@dataclass(frozen=True)
class _OrbSessionDecision:
    has_opening_range: bool
    opening_range_high: float | None = None
    opening_range_low: float | None = None
    long_breakout: _OrbBreakoutDecision | None = None
    short_breakout: _OrbBreakoutDecision | None = None

    def breakout_for_direction(self, direction: str) -> _OrbBreakoutDecision | None:
        if direction == "long":
            return self.long_breakout
        if direction == "short":
            return self.short_breakout
        return None


def apply_entry_filters(
    events: Iterable[Mapping[str, Any] | StrategySignalEvent],
    *,
    filter_config: StrategyEntryFilterConfig,
    feature_config: StrategyFeatureConfig,
    daily_features_by_symbol: Mapping[str, pd.DataFrame],
    daily_ohlc_by_symbol: Mapping[str, pd.DataFrame],
    intraday_bars_by_symbol: Mapping[str, pd.DataFrame] | None = None,
) -> tuple[tuple[StrategySignalEvent, ...], dict[str, Any], dict[str, Any]]:
    """Apply deterministic entry filters and return kept events + audit payloads."""

    parsed_events = parse_strategy_signal_events(events)
    sorted_events = tuple(sorted(parsed_events, key=_event_sort_key))
    reject_counts = {reason: 0 for reason in FILTER_REJECT_REASONS}

    daily_index_by_symbol = _build_daily_context_index_by_symbol(
        daily_features_by_symbol=daily_features_by_symbol,
        daily_ohlc_by_symbol=daily_ohlc_by_symbol,
        ema9_slope_lookback_bars=int(filter_config.ema9_slope_lookback_bars),
    )
    prepared_intraday_by_symbol = _prepare_intraday_by_symbol(intraday_bars_by_symbol or {})
    orb_cache_by_symbol: dict[str, dict[date, _OrbSessionDecision]] = {}
    if bool(filter_config.enable_orb_confirmation):
        orb_cache_by_symbol = _build_orb_session_cache(
            prepared_intraday_by_symbol=prepared_intraday_by_symbol,
            range_minutes=int(filter_config.orb_range_minutes),
            cutoff_et=str(filter_config.orb_confirmation_cutoff_et),
        )

    kept_events: list[StrategySignalEvent] = []
    for event in sorted_events:
        reject_code: str | None = None
        candidate = event
        direction = str(event.direction).strip().lower()
        symbol = normalize_symbol(event.symbol)

        if direction == "short" and not bool(filter_config.allow_shorts):
            reject_code = "shorts_disabled"

        daily_context: dict[str, Any] | None = None
        if reject_code is None and _requires_daily_context(filter_config):
            daily_context = _resolve_daily_context_row(
                event=candidate,
                daily_index_by_symbol=daily_index_by_symbol,
            )
            if not _has_required_daily_context(daily_context, filter_config=filter_config):
                reject_code = "missing_daily_context"

        if reject_code is None and bool(filter_config.enable_rsi_extremes):
            assert daily_context is not None
            if not _passes_rsi_extreme(
                daily_context=daily_context,
                direction=direction,
                oversold=float(feature_config.rsi_oversold),
                overbought=float(feature_config.rsi_overbought),
            ):
                reject_code = "rsi_not_extreme"

        if reject_code is None and bool(filter_config.enable_ema9_regime):
            assert daily_context is not None
            if not _passes_ema9_regime(daily_context=daily_context, direction=direction):
                reject_code = "ema9_regime_mismatch"

        if reject_code is None and bool(filter_config.enable_volatility_regime):
            assert daily_context is not None
            if not _passes_volatility_regime(
                daily_context=daily_context,
                allowed_regimes=filter_config.allowed_volatility_regimes,
            ):
                reject_code = "volatility_regime_disallowed"

        if (
            reject_code is None
            and bool(filter_config.enable_orb_confirmation)
            and str(candidate.strategy).strip().lower() in {"sfp", "msb"}
        ):
            session_day = _timestamp_to_date(candidate.entry_ts)
            session_cache = orb_cache_by_symbol.get(symbol, {})
            session_decision = session_cache.get(session_day) if session_day is not None else None
            if session_decision is None or not session_decision.has_opening_range:
                reject_code = "orb_opening_range_missing"
            else:
                breakout = session_decision.breakout_for_direction(direction)
                if breakout is None:
                    reject_code = "orb_breakout_missing"
                else:
                    updated_event = _apply_orb_confirmation_update(
                        event=candidate,
                        breakout=breakout,
                        stop_policy=str(filter_config.orb_stop_policy),
                    )
                    if updated_event is None:
                        reject_code = "orb_breakout_missing"
                    else:
                        candidate = updated_event

        if reject_code is None and bool(filter_config.enable_atr_stop_floor):
            assert daily_context is not None
            if not _passes_atr_floor(
                event=candidate,
                daily_context=daily_context,
                atr_floor_multiple=float(filter_config.atr_stop_floor_multiple),
                prepared_intraday=prepared_intraday_by_symbol.get(symbol),
            ):
                reject_code = "atr_floor_failed"

        if reject_code is not None:
            reject_counts[reject_code] += 1
            continue
        kept_events.append(candidate)

    kept_events_sorted = tuple(sorted(kept_events, key=_event_sort_key))
    filter_summary = _build_filter_summary(
        base_event_count=len(sorted_events),
        kept_event_count=len(kept_events_sorted),
        reject_counts=reject_counts,
    )
    filter_metadata = _build_filter_metadata(
        filter_config=filter_config,
        orb_cache_by_symbol=orb_cache_by_symbol,
    )
    return kept_events_sorted, filter_summary, filter_metadata


def _build_filter_summary(
    *,
    base_event_count: int,
    kept_event_count: int,
    reject_counts: Mapping[str, int],
) -> dict[str, Any]:
    ordered_counts = {reason: int(reject_counts.get(reason, 0)) for reason in FILTER_REJECT_REASONS}
    rejected = max(0, int(base_event_count) - int(kept_event_count))
    return {
        "base_event_count": int(base_event_count),
        "kept_event_count": int(kept_event_count),
        "rejected_event_count": rejected,
        "reject_counts": ordered_counts,
    }


def _build_filter_metadata(
    *,
    filter_config: StrategyEntryFilterConfig,
    orb_cache_by_symbol: Mapping[str, Mapping[date, _OrbSessionDecision]],
) -> dict[str, Any]:
    hour, minute = _parse_cutoff_parts(str(filter_config.orb_confirmation_cutoff_et))
    active_filters: list[str] = []
    if not bool(filter_config.allow_shorts):
        active_filters.append("allow_shorts")
    if bool(filter_config.enable_rsi_extremes):
        active_filters.append("rsi_extremes")
    if bool(filter_config.enable_ema9_regime):
        active_filters.append("ema9_regime")
    if bool(filter_config.enable_volatility_regime):
        active_filters.append("volatility_regime")
    if bool(filter_config.enable_orb_confirmation):
        active_filters.append("orb_confirmation")
    if bool(filter_config.enable_atr_stop_floor):
        active_filters.append("atr_stop_floor")

    metadata = filter_config.model_dump(mode="json")
    metadata.update(
        {
            "active_filters": active_filters,
            "reject_reason_order": list(FILTER_REJECT_REASONS),
            "parsed_orb_range_minutes": int(filter_config.orb_range_minutes),
            "parsed_orb_confirmation_cutoff_et": {
                "hour": int(hour),
                "minute": int(minute),
                "minutes_since_midnight": int(hour * 60 + minute),
            },
            "allowed_volatility_regimes": list(filter_config.allowed_volatility_regimes),
            "orb_cache_symbol_count": int(len(orb_cache_by_symbol)),
            "orb_cache_session_count": int(sum(len(value) for value in orb_cache_by_symbol.values())),
        }
    )
    return metadata


def _requires_daily_context(filter_config: StrategyEntryFilterConfig) -> bool:
    return any(
        (
            bool(filter_config.enable_rsi_extremes),
            bool(filter_config.enable_ema9_regime),
            bool(filter_config.enable_volatility_regime),
            bool(filter_config.enable_atr_stop_floor),
        )
    )


def _build_daily_context_index_by_symbol(
    *,
    daily_features_by_symbol: Mapping[str, pd.DataFrame],
    daily_ohlc_by_symbol: Mapping[str, pd.DataFrame],
    ema9_slope_lookback_bars: int,
) -> dict[str, _DailyContextIndex]:
    feature_map = _normalize_frame_map(daily_features_by_symbol)
    ohlc_map = _normalize_frame_map(daily_ohlc_by_symbol)
    symbols = sorted(set(feature_map) | set(ohlc_map))

    out: dict[str, _DailyContextIndex] = {}
    for symbol in symbols:
        feature_frame = feature_map.get(symbol, pd.DataFrame())
        ohlc_frame = ohlc_map.get(symbol, pd.DataFrame())
        combined = _build_daily_context_frame(
            feature_frame=feature_frame,
            ohlc_frame=ohlc_frame,
            ema9_slope_lookback_bars=ema9_slope_lookback_bars,
        )

        rows_by_date: dict[date, dict[str, Any]] = {}
        for ts, row in combined.iterrows():
            row_day = _timestamp_to_date(ts)
            if row_day is None:
                continue
            rows_by_date[row_day] = {column: row.get(column) for column in combined.columns}

        out[symbol] = _DailyContextIndex(
            ordered_dates=tuple(sorted(rows_by_date)),
            rows_by_date=rows_by_date,
        )
    return out


def _build_daily_context_frame(
    *,
    feature_frame: pd.DataFrame,
    ohlc_frame: pd.DataFrame,
    ema9_slope_lookback_bars: int,
) -> pd.DataFrame:
    feature_norm = feature_frame.copy() if feature_frame is not None else pd.DataFrame()
    ohlc_norm = ohlc_frame.copy() if ohlc_frame is not None else pd.DataFrame()
    if not feature_norm.empty:
        feature_norm.index = _normalize_datetime_index(feature_norm.index)
    if not ohlc_norm.empty:
        ohlc_norm.index = _normalize_datetime_index(ohlc_norm.index)

    idx = pd.DatetimeIndex([])
    if not ohlc_norm.empty:
        idx = pd.DatetimeIndex(ohlc_norm.index)
    if not feature_norm.empty:
        feature_idx = pd.DatetimeIndex(feature_norm.index)
        idx = feature_idx if idx.empty else idx.union(feature_idx)
    idx = idx.dropna().sort_values()
    if idx.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=idx)
    if "Close" in ohlc_norm.columns:
        out["close"] = pd.to_numeric(ohlc_norm["Close"], errors="coerce").reindex(idx)

    feature_columns = ("rsi", "atr", "ema9", "ema9_slope", "volatility_regime", "realized_vol_regime")
    for column in feature_columns:
        if column in feature_norm.columns:
            out[column] = feature_norm[column].reindex(idx)

    if "ema9" in out.columns:
        ema9_series = pd.to_numeric(out["ema9"], errors="coerce").astype("float64")
        out["ema9_slope_filter"] = compute_ema_slope(
            ema9_series,
            lookback_bars=int(ema9_slope_lookback_bars),
        )
    elif "ema9_slope" in out.columns:
        out["ema9_slope_filter"] = pd.to_numeric(out["ema9_slope"], errors="coerce")

    return out


def _resolve_daily_context_row(
    *,
    event: StrategySignalEvent,
    daily_index_by_symbol: Mapping[str, _DailyContextIndex],
) -> dict[str, Any] | None:
    symbol = normalize_symbol(event.symbol)
    index = daily_index_by_symbol.get(symbol)
    if index is None or not index.ordered_dates:
        return None

    signal_day = _timestamp_to_date(event.signal_ts)
    if signal_day is None:
        return None

    strategy = str(event.strategy).strip().lower()
    if strategy == "orb":
        anchor_day = _previous_available_day(index.ordered_dates, signal_day)
    else:
        anchor_day = signal_day
    if anchor_day is None:
        return None
    return index.rows_by_date.get(anchor_day)


def _has_required_daily_context(
    daily_context: Mapping[str, Any] | None,
    *,
    filter_config: StrategyEntryFilterConfig,
) -> bool:
    if daily_context is None:
        return False

    if bool(filter_config.enable_rsi_extremes) and _finite_float(daily_context.get("rsi")) is None:
        return False

    if bool(filter_config.enable_ema9_regime):
        if _finite_float(daily_context.get("close")) is None:
            return False
        if _finite_float(daily_context.get("ema9")) is None:
            return False
        if _finite_float(daily_context.get("ema9_slope_filter")) is None:
            return False

    if bool(filter_config.enable_volatility_regime):
        regime = _resolve_volatility_regime(daily_context)
        if regime is None:
            return False

    if bool(filter_config.enable_atr_stop_floor):
        atr_value = _finite_float(daily_context.get("atr"))
        if atr_value is None or atr_value <= 0.0:
            return False

    return True


def _passes_rsi_extreme(
    *,
    daily_context: Mapping[str, Any],
    direction: str,
    oversold: float,
    overbought: float,
) -> bool:
    rsi_value = _finite_float(daily_context.get("rsi"))
    if rsi_value is None:
        return False
    if direction == "long":
        return rsi_value <= float(oversold)
    if direction == "short":
        return rsi_value >= float(overbought)
    return False


def _passes_ema9_regime(
    *,
    daily_context: Mapping[str, Any],
    direction: str,
) -> bool:
    close_value = _finite_float(daily_context.get("close"))
    ema9_value = _finite_float(daily_context.get("ema9"))
    slope_value = _finite_float(daily_context.get("ema9_slope_filter"))
    if close_value is None or ema9_value is None or slope_value is None:
        return False

    if direction == "long":
        return close_value >= ema9_value and slope_value >= 0.0
    if direction == "short":
        return close_value <= ema9_value and slope_value <= 0.0
    return False


def _passes_volatility_regime(
    *,
    daily_context: Mapping[str, Any],
    allowed_regimes: tuple[str, ...],
) -> bool:
    regime = _resolve_volatility_regime(daily_context)
    if regime is None:
        return False
    allowed = {str(value).strip().lower() for value in allowed_regimes}
    return regime in allowed


def _passes_atr_floor(
    *,
    event: StrategySignalEvent,
    daily_context: Mapping[str, Any],
    atr_floor_multiple: float,
    prepared_intraday: _PreparedIntraday | None,
) -> bool:
    atr_value = _finite_float(daily_context.get("atr"))
    stop_price = _finite_float(event.stop_price)
    if atr_value is None or atr_value <= 0.0 or stop_price is None:
        return False

    entry_price = _estimate_entry_price(event=event, prepared_intraday=prepared_intraday)
    if entry_price is None:
        return False

    initial_risk = abs(float(entry_price) - float(stop_price))
    if initial_risk <= 0.0 or not np.isfinite(initial_risk):
        return False
    return (initial_risk / atr_value) >= float(atr_floor_multiple)


def _estimate_entry_price(
    *,
    event: StrategySignalEvent,
    prepared_intraday: _PreparedIntraday | None,
) -> float | None:
    if prepared_intraday is not None:
        session_day = _timestamp_to_date(event.entry_ts)
        entry_ts = _to_utc_timestamp(event.entry_ts)
        if session_day is not None and entry_ts is not None:
            session_mask = prepared_intraday.session_days == session_day
            if bool(np.any(session_mask)):
                session_positions = np.flatnonzero(session_mask)
                later_positions = session_positions[
                    prepared_intraday.timestamp_ns[session_positions] >= int(entry_ts.value)
                ]
                if later_positions.size:
                    return _finite_float(prepared_intraday.open_values[int(later_positions[0])])
                return _finite_float(prepared_intraday.open_values[int(session_positions[0])])

    signal_close = _finite_float(event.signal_close)
    if signal_close is not None:
        return signal_close
    return _finite_float(event.signal_open)


def _prepare_intraday_by_symbol(
    bars_by_symbol: Mapping[str, pd.DataFrame],
) -> dict[str, _PreparedIntraday]:
    normalized_map = _normalize_frame_map(bars_by_symbol)
    out: dict[str, _PreparedIntraday] = {}
    for symbol in sorted(normalized_map):
        frame = normalized_map[symbol]
        if frame is None or frame.empty:
            continue
        try:
            normalized = normalize_intraday_ohlc_frame(frame)
        except Exception:  # noqa: BLE001
            continue
        if normalized.empty:
            continue

        ts_series = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
        open_series = pd.to_numeric(normalized["open"], errors="coerce")
        valid_mask = ts_series.notna() & open_series.notna()
        if not bool(valid_mask.any()):
            continue
        trimmed = normalized.loc[valid_mask].copy()
        ts_series = pd.to_datetime(trimmed["timestamp"], errors="coerce", utc=True)
        open_series = pd.to_numeric(trimmed["open"], errors="coerce")

        out[symbol] = _PreparedIntraday(
            frame=trimmed.reset_index(drop=True),
            timestamp_ns=ts_series.astype("int64").to_numpy(dtype="int64"),
            session_days=trimmed["session_date"].to_numpy(dtype=object),
            open_values=open_series.to_numpy(dtype="float64"),
        )
    return out


def _build_orb_session_cache(
    *,
    prepared_intraday_by_symbol: Mapping[str, _PreparedIntraday],
    range_minutes: int,
    cutoff_et: str,
) -> dict[str, dict[date, _OrbSessionDecision]]:
    out: dict[str, dict[date, _OrbSessionDecision]] = {}
    cutoff_time = _parse_cutoff_time(cutoff_et)
    range_delta = pd.Timedelta(minutes=int(range_minutes))

    for symbol in sorted(prepared_intraday_by_symbol):
        prepared = prepared_intraday_by_symbol[symbol]
        frame = prepared.frame
        if frame.empty:
            continue

        bar_duration = _infer_bar_duration(frame)
        confirmation_offset = max(bar_duration - pd.Timedelta(microseconds=1), _CONFIRMATION_FLOOR)

        session_map: dict[date, _OrbSessionDecision] = {}
        grouped = frame.groupby("session_date", sort=True)
        for session_day, session in grouped:
            session_open_market = _session_timestamp(session_day, _REGULAR_OPEN)
            range_end_market = session_open_market + range_delta
            cutoff_market = _session_timestamp(session_day, cutoff_time)
            cutoff_utc = cutoff_market.tz_convert("UTC")

            session_rows = session.sort_values("timestamp", kind="stable").reset_index(drop=True)
            opening = session_rows.loc[session_rows["timestamp_market"] < range_end_market]
            if opening.empty:
                session_map[session_day] = _OrbSessionDecision(has_opening_range=False)
                continue

            opening_range_high = float(opening["high"].max())
            opening_range_low = float(opening["low"].min())
            long_breakout: _OrbBreakoutDecision | None = None
            short_breakout: _OrbBreakoutDecision | None = None

            for row_position in range(len(session_rows)):
                row_market_ts = pd.Timestamp(session_rows.at[row_position, "timestamp_market"])
                if row_market_ts < range_end_market:
                    continue

                signal_ts = pd.Timestamp(session_rows.at[row_position, "timestamp"])
                signal_confirmed_ts = signal_ts + confirmation_offset
                if signal_confirmed_ts > cutoff_utc:
                    break
                if row_position + 1 >= len(session_rows):
                    continue

                entry_ts = pd.Timestamp(session_rows.at[row_position + 1, "timestamp"])
                if entry_ts <= signal_confirmed_ts:
                    continue

                close_value = float(session_rows.at[row_position, "close"])
                if close_value > opening_range_high and long_breakout is None:
                    long_breakout = _OrbBreakoutDecision(
                        signal_confirmed_ts=signal_confirmed_ts,
                        entry_ts=entry_ts,
                        stop_price=float(opening_range_low),
                    )
                elif close_value < opening_range_low and short_breakout is None:
                    short_breakout = _OrbBreakoutDecision(
                        signal_confirmed_ts=signal_confirmed_ts,
                        entry_ts=entry_ts,
                        stop_price=float(opening_range_high),
                    )

                if long_breakout is not None and short_breakout is not None:
                    break

            session_map[session_day] = _OrbSessionDecision(
                has_opening_range=True,
                opening_range_high=opening_range_high,
                opening_range_low=opening_range_low,
                long_breakout=long_breakout,
                short_breakout=short_breakout,
            )

        out[symbol] = session_map
    return out


def _apply_orb_confirmation_update(
    *,
    event: StrategySignalEvent,
    breakout: _OrbBreakoutDecision,
    stop_policy: str,
) -> StrategySignalEvent | None:
    direction = str(event.direction).strip().lower()
    updates: dict[str, Any] = {
        "signal_confirmed_ts": breakout.signal_confirmed_ts.to_pydatetime(),
        "entry_ts": breakout.entry_ts.to_pydatetime(),
    }

    policy = str(stop_policy).strip().lower()
    if policy == "orb_range":
        updates["stop_price"] = float(breakout.stop_price)
    elif policy == "tighten":
        base_stop = _finite_float(event.stop_price)
        orb_stop = float(breakout.stop_price)
        if base_stop is None:
            updates["stop_price"] = orb_stop
        elif direction == "short":
            updates["stop_price"] = min(base_stop, orb_stop)
        else:
            updates["stop_price"] = max(base_stop, orb_stop)

    notes = list(event.notes)
    notes.append("orb_confirmation_applied=1")
    notes.append(f"orb_stop_policy={policy}")
    updates["notes"] = notes

    updated = event.model_copy(update=updates)
    confirmed_ts = _to_utc_timestamp(updated.signal_confirmed_ts)
    entry_ts = _to_utc_timestamp(updated.entry_ts)
    if confirmed_ts is None or entry_ts is None:
        return None
    if entry_ts <= confirmed_ts:
        return None
    return updated


def _event_sort_key(event: StrategySignalEvent) -> tuple[Any, ...]:
    return (
        event.signal_confirmed_ts,
        event.signal_ts,
        event.entry_ts,
        event.symbol,
        event.event_id,
    )


def _normalize_frame_map(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for raw_symbol in sorted(frames):
        symbol = normalize_symbol(raw_symbol)
        if not symbol or symbol in out:
            continue
        frame = frames[raw_symbol]
        if frame is None:
            continue
        out[symbol] = frame
    return out


def _normalize_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))


def _resolve_volatility_regime(daily_context: Mapping[str, Any]) -> str | None:
    primary = _string_or_none(daily_context.get("volatility_regime"))
    if primary is not None:
        return primary.lower()
    fallback = _string_or_none(daily_context.get("realized_vol_regime"))
    if fallback is None:
        return None
    return fallback.lower()


def _previous_available_day(days: tuple[date, ...], anchor_day: date) -> date | None:
    index = bisect_left(days, anchor_day) - 1
    if index < 0:
        return None
    return days[index]


def _timestamp_to_date(value: object) -> date | None:
    ts = _to_utc_timestamp(value)
    if ts is None:
        return None
    return ts.date()


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


def _session_timestamp(session_day: object, local_time: time) -> pd.Timestamp:
    resolved_day = pd.Timestamp(session_day).date()
    return pd.Timestamp.combine(resolved_day, local_time).tz_localize(_MARKET_TZ)


def _parse_cutoff_time(value: str) -> time:
    hour, minute = _parse_cutoff_parts(value)
    return time(hour=hour, minute=minute)


def _parse_cutoff_parts(value: str) -> tuple[int, int]:
    text = str(value).strip()
    parsed = datetime.strptime(text, "%H:%M")
    return int(parsed.hour), int(parsed.minute)


def _infer_bar_duration(frame: pd.DataFrame) -> pd.Timedelta:
    if frame.empty:
        return pd.Timedelta(minutes=1)

    deltas: list[int] = []
    grouped = frame.groupby("session_date", sort=False)
    for _, session in grouped:
        ts_ns = pd.to_datetime(session["timestamp"], errors="coerce", utc=True).astype("int64").to_numpy()
        if ts_ns.size < 2:
            continue
        diff = np.diff(ts_ns)
        positive = diff[diff > 0]
        if positive.size:
            deltas.extend(int(value) for value in positive)
    if not deltas:
        return pd.Timedelta(minutes=1)
    return pd.Timedelta(int(np.min(np.asarray(deltas, dtype="int64"))), unit="ns")


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(number):
        return None
    return number


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null", "nan"}:
        return None
    return text


__all__ = [
    "FILTER_REJECT_REASONS",
    "apply_entry_filters",
]
