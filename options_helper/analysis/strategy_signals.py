from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from options_helper.analysis.msb import compute_msb_signals, extract_msb_signal_candidates
from options_helper.analysis.orb import compute_orb_signals, extract_orb_signal_candidates
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_signal_candidates
from options_helper.schemas.strategy_modeling_contracts import EntryPriceSource, StrategySignalEvent


StrategySignalAdapter = Callable[..., list[StrategySignalEvent]]

_DEFAULT_ENTRY_PRICE_SOURCE: EntryPriceSource = "first_tradable_bar_open_after_signal_confirmed_ts"
_SIGNAL_ADAPTER_REGISTRY: dict[str, StrategySignalAdapter] = {}


def _strategy_key(strategy: str) -> str:
    key = str(strategy).strip().lower()
    if not key:
        raise ValueError("strategy must be non-empty")
    return key


def _normalize_timeframe_label(timeframe: str | None) -> str:
    if timeframe is None:
        return "1d"
    label = str(timeframe).strip()
    if not label:
        return "1d"
    if label.lower() in {"native", "raw", "none"}:
        return "1d"
    return label


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return pd.Timestamp(value).to_pydatetime()


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(number):
        return None
    return number


def register_strategy_signal_adapter(
    strategy: str,
    adapter: StrategySignalAdapter,
    *,
    replace: bool = False,
) -> None:
    key = _strategy_key(strategy)
    if key in _SIGNAL_ADAPTER_REGISTRY and not replace:
        raise ValueError(f"strategy adapter already registered: {strategy}")
    _SIGNAL_ADAPTER_REGISTRY[key] = adapter


def get_strategy_signal_adapter(strategy: str) -> StrategySignalAdapter:
    key = _strategy_key(strategy)
    adapter = _SIGNAL_ADAPTER_REGISTRY.get(key)
    if adapter is None:
        registered = ", ".join(sorted(_SIGNAL_ADAPTER_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown strategy signal adapter: {strategy}. Registered: {registered}")
    return adapter


def list_registered_strategy_signal_adapters() -> tuple[str, ...]:
    return tuple(sorted(_SIGNAL_ADAPTER_REGISTRY))


def build_strategy_signal_events(
    strategy: str,
    ohlc: pd.DataFrame,
    **kwargs: Any,
) -> list[StrategySignalEvent]:
    adapter = get_strategy_signal_adapter(strategy)
    return adapter(ohlc, **kwargs)


def normalize_sfp_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1d",
    swing_right_bars: int = 1,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    if signals is None or signals.empty:
        return []

    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    required_lag = int(swing_right_bars)
    if required_lag < 1:
        raise ValueError("swing_right_bars must be >= 1")

    index_values = list(signals.index)
    timeframe_label = _normalize_timeframe_label(timeframe)
    events: list[StrategySignalEvent] = []

    for row in extract_sfp_signal_candidates(signals):
        row_position = int(row["row_position"])
        if row_position + 1 >= len(index_values):
            # The last bar has no first tradable bar after confirmation in this frame.
            continue

        bars_since_swing = int(row["bars_since_swing"])
        if bars_since_swing < required_lag:
            continue

        raw_direction = str(row["direction"])
        if raw_direction == "bullish":
            direction = "long"
        elif raw_direction == "bearish":
            direction = "short"
        else:
            continue

        signal_ts = _to_datetime(index_values[row_position])
        signal_confirmed_ts = signal_ts
        entry_ts = _to_datetime(index_values[row_position + 1])
        if entry_ts <= signal_confirmed_ts:
            continue

        signal_open = _float_or_none(row.get("candle_open"))
        signal_high = _float_or_none(row.get("candle_high"))
        signal_low = _float_or_none(row.get("candle_low"))
        signal_close = _float_or_none(row.get("candle_close"))
        stop_price = signal_low if direction == "long" else signal_high

        notes = [
            f"swept_swing_timestamp={row['swept_swing_timestamp']}",
            f"sweep_level={row['sweep_level']}",
            f"bars_since_swing={bars_since_swing}",
            f"swing_right_bars={required_lag}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]

        event_id = (
            f"sfp:{normalized_symbol}:{timeframe_label}:"
            f"{pd.Timestamp(signal_ts).isoformat()}:{direction}"
        )
        events.append(
            StrategySignalEvent(
                event_id=event_id,
                strategy="sfp",
                symbol=normalized_symbol,
                timeframe=timeframe_label,
                direction=direction,
                signal_ts=signal_ts,
                signal_confirmed_ts=signal_confirmed_ts,
                entry_ts=entry_ts,
                entry_price_source=entry_price_source,
                signal_open=signal_open,
                signal_high=signal_high,
                signal_low=signal_low,
                signal_close=signal_close,
                stop_price=stop_price,
                notes=notes,
            )
        )

    return events


def normalize_msb_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1d",
    swing_right_bars: int = 1,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    if signals is None or signals.empty:
        return []

    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    required_lag = int(swing_right_bars)
    if required_lag < 1:
        raise ValueError("swing_right_bars must be >= 1")

    index_values = list(signals.index)
    timeframe_label = _normalize_timeframe_label(timeframe)
    events: list[StrategySignalEvent] = []

    for row in extract_msb_signal_candidates(signals):
        row_position = int(row["row_position"])
        if row_position + 1 >= len(index_values):
            # The last bar has no first tradable bar after confirmation in this frame.
            continue

        bars_since_swing = int(row["bars_since_swing"])
        if bars_since_swing < required_lag:
            continue

        raw_direction = str(row["direction"])
        if raw_direction == "bullish":
            direction = "long"
        elif raw_direction == "bearish":
            direction = "short"
        else:
            continue

        signal_ts = _to_datetime(index_values[row_position])
        signal_confirmed_ts = signal_ts
        entry_ts = _to_datetime(index_values[row_position + 1])
        if entry_ts <= signal_confirmed_ts:
            continue

        signal_open = _float_or_none(row.get("candle_open"))
        signal_high = _float_or_none(row.get("candle_high"))
        signal_low = _float_or_none(row.get("candle_low"))
        signal_close = _float_or_none(row.get("candle_close"))
        stop_price = signal_low if direction == "long" else signal_high

        notes = [
            f"broken_swing_timestamp={row['broken_swing_timestamp']}",
            f"break_level={row['break_level']}",
            f"bars_since_swing={bars_since_swing}",
            f"swing_right_bars={required_lag}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]

        event_id = (
            f"msb:{normalized_symbol}:{timeframe_label}:"
            f"{pd.Timestamp(signal_ts).isoformat()}:{direction}"
        )
        events.append(
            StrategySignalEvent(
                event_id=event_id,
                strategy="msb",
                symbol=normalized_symbol,
                timeframe=timeframe_label,
                direction=direction,
                signal_ts=signal_ts,
                signal_confirmed_ts=signal_confirmed_ts,
                entry_ts=entry_ts,
                entry_price_source=entry_price_source,
                signal_open=signal_open,
                signal_high=signal_high,
                signal_low=signal_low,
                signal_close=signal_close,
                stop_price=stop_price,
                notes=notes,
            )
        )

    return events


def normalize_orb_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1m",
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    if signals is None or signals.empty:
        return []

    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    timeframe_label = _normalize_timeframe_label(timeframe or "1m")
    events: list[StrategySignalEvent] = []

    for row in extract_orb_signal_candidates(signals):
        direction = str(row.get("direction") or "").strip().lower()
        if direction not in {"long", "short"}:
            continue

        signal_ts = _to_datetime(row["signal_ts"])
        signal_confirmed_ts = _to_datetime(row["signal_confirmed_ts"])
        entry_ts = _to_datetime(row["entry_ts"])
        if entry_ts <= signal_confirmed_ts:
            continue

        signal_open = _float_or_none(row.get("candle_open"))
        signal_high = _float_or_none(row.get("candle_high"))
        signal_low = _float_or_none(row.get("candle_low"))
        signal_close = _float_or_none(row.get("candle_close"))
        stop_price = _float_or_none(row.get("stop_price"))

        range_end_ts = row.get("range_end_ts")
        cutoff_ts = row.get("cutoff_ts")
        notes = [
            f"session_date={row.get('session_date')}",
            f"opening_range_high={row.get('opening_range_high')}",
            f"opening_range_low={row.get('opening_range_low')}",
            f"range_end_ts={pd.Timestamp(range_end_ts).isoformat()}" if range_end_ts is not None else "range_end_ts=",
            f"cutoff_ts={pd.Timestamp(cutoff_ts).isoformat()}" if cutoff_ts is not None else "cutoff_ts=",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]

        event_id = (
            f"orb:{normalized_symbol}:{timeframe_label}:"
            f"{pd.Timestamp(signal_ts).isoformat()}:{direction}"
        )
        events.append(
            StrategySignalEvent(
                event_id=event_id,
                strategy="orb",
                symbol=normalized_symbol,
                timeframe=timeframe_label,
                direction=direction,  # type: ignore[arg-type]
                signal_ts=signal_ts,
                signal_confirmed_ts=signal_confirmed_ts,
                entry_ts=entry_ts,
                entry_price_source=entry_price_source,
                signal_open=signal_open,
                signal_high=signal_high,
                signal_low=signal_low,
                signal_close=signal_close,
                stop_price=stop_price,
                notes=notes,
            )
        )

    return events


def adapt_sfp_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
    swing_left_bars: int = 1,
    swing_right_bars: int = 1,
    min_swing_distance_bars: int = 1,
    ignore_swept_swings: bool = False,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    signals = compute_sfp_signals(
        ohlc,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        ignore_swept_swings=ignore_swept_swings,
        timeframe=timeframe,
    )
    return normalize_sfp_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
        swing_right_bars=swing_right_bars,
        entry_price_source=entry_price_source,
    )


def adapt_msb_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
    swing_left_bars: int = 1,
    swing_right_bars: int = 1,
    min_swing_distance_bars: int = 1,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    signals = compute_msb_signals(
        ohlc,
        swing_left_bars=swing_left_bars,
        swing_right_bars=swing_right_bars,
        min_swing_distance_bars=min_swing_distance_bars,
        timeframe=timeframe,
    )
    return normalize_msb_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
        swing_right_bars=swing_right_bars,
        entry_price_source=entry_price_source,
    )


def _resolve_orb_intraday_bars(
    *,
    symbol: str,
    intraday_bars: pd.DataFrame | None,
    intraday_bars_by_symbol: Mapping[str, pd.DataFrame] | None,
) -> pd.DataFrame | None:
    if intraday_bars is not None:
        return intraday_bars
    if not intraday_bars_by_symbol:
        return None

    normalized_symbol = str(symbol).strip().upper()
    direct = intraday_bars_by_symbol.get(symbol)
    if direct is not None:
        return direct

    direct = intraday_bars_by_symbol.get(normalized_symbol)
    if direct is not None:
        return direct

    for raw_symbol, frame in intraday_bars_by_symbol.items():
        if str(raw_symbol).strip().upper() == normalized_symbol:
            return frame
    return None


def adapt_orb_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1m",
    intraday_bars: pd.DataFrame | None = None,
    intraday_bars_by_symbol: Mapping[str, pd.DataFrame] | None = None,
    orb_range_minutes: int = 15,
    orb_confirmation_cutoff_et: str = "10:30",
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
    **_: Any,
) -> list[StrategySignalEvent]:
    _ = ohlc
    intraday = _resolve_orb_intraday_bars(
        symbol=symbol,
        intraday_bars=intraday_bars,
        intraday_bars_by_symbol=intraday_bars_by_symbol,
    )
    if intraday is None or intraday.empty:
        return []

    signals = compute_orb_signals(
        intraday,
        range_minutes=orb_range_minutes,
        cutoff_et=orb_confirmation_cutoff_et,
    )
    return normalize_orb_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
        entry_price_source=entry_price_source,
    )


register_strategy_signal_adapter("sfp", adapt_sfp_signal_events, replace=True)
register_strategy_signal_adapter("msb", adapt_msb_signal_events, replace=True)
register_strategy_signal_adapter("orb", adapt_orb_signal_events, replace=True)


__all__ = [
    "StrategySignalAdapter",
    "adapt_orb_signal_events",
    "adapt_msb_signal_events",
    "adapt_sfp_signal_events",
    "build_strategy_signal_events",
    "get_strategy_signal_adapter",
    "list_registered_strategy_signal_adapters",
    "normalize_orb_signal_events",
    "normalize_msb_signal_events",
    "normalize_sfp_signal_events",
    "register_strategy_signal_adapter",
]
