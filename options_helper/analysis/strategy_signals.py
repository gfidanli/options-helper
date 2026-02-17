from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from options_helper.analysis.fib_retracement import compute_fib_retracement_signals
from options_helper.analysis.ma_crossover import (
    compute_ma_crossover_signals,
    extract_ma_crossover_signal_candidates,
)
from options_helper.analysis.msb import compute_msb_signals, extract_msb_signal_candidates
from options_helper.analysis.orb import compute_orb_signals, extract_orb_signal_candidates
from options_helper.analysis.sfp import compute_sfp_signals, extract_sfp_signal_candidates
from options_helper.analysis.trend_following import (
    compute_trend_following_signals,
    extract_trend_following_signal_candidates,
)
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


def _normalize_signal_context(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None,
) -> tuple[str, list[object], str] | None:
    if signals is None or signals.empty:
        return None
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")
    index_values = list(signals.index)
    timeframe_label = _normalize_timeframe_label(timeframe)
    return normalized_symbol, index_values, timeframe_label


def _next_bar_signal_timing(
    index_values: list[object],
    *,
    row_position: int,
) -> tuple[datetime, datetime, datetime] | None:
    if row_position + 1 >= len(index_values):
        # The last bar has no first tradable bar after confirmation in this frame.
        return None
    signal_ts = _to_datetime(index_values[row_position])
    signal_confirmed_ts = signal_ts
    entry_ts = _to_datetime(index_values[row_position + 1])
    if entry_ts <= signal_confirmed_ts:
        return None
    return signal_ts, signal_confirmed_ts, entry_ts


def _direction_from_bullish_bearish(raw_direction: object) -> str | None:
    direction_key = str(raw_direction)
    if direction_key == "bullish":
        return "long"
    if direction_key == "bearish":
        return "short"
    return None


def _direction_from_long_short(raw_direction: object) -> str | None:
    direction_key = str(raw_direction or "").strip().lower()
    if direction_key not in {"long", "short"}:
        return None
    return direction_key


def _extract_signal_ohlc(
    row: object,
    *,
    open_key: str = "candle_open",
    high_key: str = "candle_high",
    low_key: str = "candle_low",
    close_key: str = "candle_close",
) -> tuple[float | None, float | None, float | None, float | None]:
    row_getter = getattr(row, "get")
    return (
        _float_or_none(row_getter(open_key)),
        _float_or_none(row_getter(high_key)),
        _float_or_none(row_getter(low_key)),
        _float_or_none(row_getter(close_key)),
    )


def _build_strategy_signal_event(
    *,
    strategy: str,
    normalized_symbol: str,
    timeframe_label: str,
    direction: str,
    signal_ts: datetime,
    signal_confirmed_ts: datetime,
    entry_ts: datetime,
    entry_price_source: EntryPriceSource,
    signal_open: float | None,
    signal_high: float | None,
    signal_low: float | None,
    signal_close: float | None,
    stop_price: float | None,
    notes: list[str],
) -> StrategySignalEvent:
    event_id = (
        f"{strategy}:{normalized_symbol}:{timeframe_label}:"
        f"{pd.Timestamp(signal_ts).isoformat()}:{direction}"
    )
    return StrategySignalEvent(
        event_id=event_id,
        strategy=strategy,
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
    context = _normalize_signal_context(signals, symbol=symbol, timeframe=timeframe)
    if context is None:
        return []
    normalized_symbol, index_values, timeframe_label = context

    required_lag = int(swing_right_bars)
    if required_lag < 1:
        raise ValueError("swing_right_bars must be >= 1")

    events: list[StrategySignalEvent] = []
    for row in extract_sfp_signal_candidates(signals):
        row_position = int(row["row_position"])
        timing = _next_bar_signal_timing(index_values, row_position=row_position)
        if timing is None:
            continue
        bars_since_swing = int(row["bars_since_swing"])
        if bars_since_swing < required_lag:
            continue
        direction = _direction_from_bullish_bearish(row["direction"])
        if direction is None:
            continue
        signal_ts, signal_confirmed_ts, entry_ts = timing
        signal_open, signal_high, signal_low, signal_close = _extract_signal_ohlc(row)
        stop_price = signal_low if direction == "long" else signal_high
        notes = [
            f"swept_swing_timestamp={row['swept_swing_timestamp']}",
            f"sweep_level={row['sweep_level']}",
            f"bars_since_swing={bars_since_swing}",
            f"swing_right_bars={required_lag}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]
        events.append(
            _build_strategy_signal_event(
                strategy="sfp",
                normalized_symbol=normalized_symbol,
                timeframe_label=timeframe_label,
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
    context = _normalize_signal_context(signals, symbol=symbol, timeframe=timeframe)
    if context is None:
        return []
    normalized_symbol, index_values, timeframe_label = context

    required_lag = int(swing_right_bars)
    if required_lag < 1:
        raise ValueError("swing_right_bars must be >= 1")

    events: list[StrategySignalEvent] = []
    for row in extract_msb_signal_candidates(signals):
        row_position = int(row["row_position"])
        timing = _next_bar_signal_timing(index_values, row_position=row_position)
        if timing is None:
            continue
        bars_since_swing = int(row["bars_since_swing"])
        if bars_since_swing < required_lag:
            continue
        direction = _direction_from_bullish_bearish(row["direction"])
        if direction is None:
            continue
        signal_ts, signal_confirmed_ts, entry_ts = timing
        signal_open, signal_high, signal_low, signal_close = _extract_signal_ohlc(row)
        stop_price = signal_low if direction == "long" else signal_high
        notes = [
            f"broken_swing_timestamp={row['broken_swing_timestamp']}",
            f"break_level={row['break_level']}",
            f"bars_since_swing={bars_since_swing}",
            f"swing_right_bars={required_lag}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]
        events.append(
            _build_strategy_signal_event(
                strategy="msb",
                normalized_symbol=normalized_symbol,
                timeframe_label=timeframe_label,
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


def normalize_ma_crossover_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1d",
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    if signals is None or signals.empty:
        return []

    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("symbol must be non-empty")

    index_values = list(signals.index)
    timeframe_label = _normalize_timeframe_label(timeframe)
    events: list[StrategySignalEvent] = []

    for row in extract_ma_crossover_signal_candidates(signals):
        row_position = int(row["row_position"])
        if row_position + 1 >= len(index_values):
            continue

        direction = str(row.get("direction") or "").strip().lower()
        if direction not in {"long", "short"}:
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
        stop_price = _float_or_none(row.get("stop_price"))
        if stop_price is None:
            continue

        notes = [
            f"fast_window={row.get('fast_window')}",
            f"slow_window={row.get('slow_window')}",
            f"fast_type={row.get('fast_type')}",
            f"slow_type={row.get('slow_type')}",
            f"fast_ma={row.get('fast_ma')}",
            f"slow_ma={row.get('slow_ma')}",
            f"atr_window={row.get('atr_window')}",
            f"atr={row.get('atr')}",
            f"atr_stop_multiple={row.get('atr_stop_multiple')}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]

        event_id = (
            f"ma_crossover:{normalized_symbol}:{timeframe_label}:"
            f"{pd.Timestamp(signal_ts).isoformat()}:{direction}"
        )
        events.append(
            StrategySignalEvent(
                event_id=event_id,
                strategy="ma_crossover",
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


def normalize_trend_following_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1d",
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    context = _normalize_signal_context(signals, symbol=symbol, timeframe=timeframe)
    if context is None:
        return []
    normalized_symbol, index_values, timeframe_label = context
    events: list[StrategySignalEvent] = []
    for row in extract_trend_following_signal_candidates(signals):
        row_position = int(row["row_position"])
        timing = _next_bar_signal_timing(index_values, row_position=row_position)
        if timing is None:
            continue
        direction = _direction_from_long_short(row.get("direction"))
        if direction is None:
            continue
        signal_ts, signal_confirmed_ts, entry_ts = timing
        signal_open, signal_high, signal_low, signal_close = _extract_signal_ohlc(row)
        stop_price = _float_or_none(row.get("stop_price"))
        if stop_price is None:
            continue
        notes = [
            f"trend_window={row.get('trend_window')}",
            f"trend_type={row.get('trend_type')}",
            f"trend_ma={row.get('trend_ma')}",
            f"trend_ma_slope={row.get('trend_ma_slope')}",
            f"fast_window={row.get('fast_window')}",
            f"fast_type={row.get('fast_type')}",
            f"fast_ma={row.get('fast_ma')}",
            f"slope_lookback_bars={row.get('slope_lookback_bars')}",
            f"atr_window={row.get('atr_window')}",
            f"atr={row.get('atr')}",
            f"atr_stop_multiple={row.get('atr_stop_multiple')}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]
        events.append(
            _build_strategy_signal_event(
                strategy="trend_following",
                normalized_symbol=normalized_symbol,
                timeframe_label=timeframe_label,
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


def normalize_fib_retracement_signal_events(
    signals: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = "1d",
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
) -> list[StrategySignalEvent]:
    context = _normalize_signal_context(signals, symbol=symbol, timeframe=timeframe)
    if context is None:
        return []
    normalized_symbol, index_values, timeframe_label = context
    events: list[StrategySignalEvent] = []
    long_flags = signals.get("fib_retracement_long")
    short_flags = signals.get("fib_retracement_short")
    if long_flags is None or short_flags is None:
        return []

    long_mask = long_flags.fillna(False).to_numpy(dtype=bool)
    short_mask = short_flags.fillna(False).to_numpy(dtype=bool)
    candidate_positions = np.flatnonzero(long_mask | short_mask)

    for row_position in candidate_positions:
        timing = _next_bar_signal_timing(index_values, row_position=int(row_position))
        if timing is None:
            continue
        signal_ts, signal_confirmed_ts, entry_ts = timing
        is_long = bool(long_mask[row_position])
        is_short = bool(short_mask[row_position])
        if is_long == is_short:
            continue
        direction = "long" if is_long else "short"
        row = signals.iloc[int(row_position)]
        signal_open, signal_high, signal_low, signal_close = _extract_signal_ohlc(
            row,
            open_key="Open",
            high_key="High",
            low_key="Low",
            close_key="Close",
        )
        stop_price = _float_or_none(
            row.get("fib_range_low_level") if direction == "long" else row.get("fib_range_high_level")
        )
        if stop_price is None:
            continue
        notes = [
            f"fib_retracement_pct={row.get('fib_retracement_pct')}",
            f"fib_entry_level={row.get('fib_entry_level')}",
            f"fib_range_high_level={row.get('fib_range_high_level')}",
            f"fib_range_low_level={row.get('fib_range_low_level')}",
            f"fib_range_high_timestamp={row.get('fib_range_high_timestamp')}",
            f"fib_range_low_timestamp={row.get('fib_range_low_timestamp')}",
            f"fib_msb_timestamp={row.get('fib_msb_timestamp')}",
            f"fib_broken_swing_level={row.get('fib_broken_swing_level')}",
            f"fib_broken_swing_timestamp={row.get('fib_broken_swing_timestamp')}",
            "entry_ts_policy=next_bar_open_after_signal_confirmed_ts",
        ]
        events.append(
            _build_strategy_signal_event(
                strategy="fib_retracement",
                normalized_symbol=normalized_symbol,
                timeframe_label=timeframe_label,
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


def adapt_ma_crossover_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
    fast_window: int = 20,
    slow_window: int = 50,
    fast_type: str = "sma",
    slow_type: str = "sma",
    atr_window: int = 14,
    atr_stop_multiple: float = 2.0,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
    **_: Any,
) -> list[StrategySignalEvent]:
    signals = compute_ma_crossover_signals(
        ohlc,
        fast_window=fast_window,
        slow_window=slow_window,
        fast_type=fast_type,
        slow_type=slow_type,
        atr_window=atr_window,
        atr_stop_multiple=atr_stop_multiple,
        timeframe=timeframe,
    )
    return normalize_ma_crossover_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
        entry_price_source=entry_price_source,
    )


def adapt_trend_following_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
    trend_window: int = 200,
    trend_type: str = "sma",
    fast_window: int = 20,
    fast_type: str = "sma",
    slope_lookback_bars: int = 3,
    atr_window: int = 14,
    atr_stop_multiple: float = 2.0,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
    **_: Any,
) -> list[StrategySignalEvent]:
    signals = compute_trend_following_signals(
        ohlc,
        trend_window=trend_window,
        trend_type=trend_type,
        fast_window=fast_window,
        fast_type=fast_type,
        slope_lookback_bars=slope_lookback_bars,
        atr_window=atr_window,
        atr_stop_multiple=atr_stop_multiple,
        timeframe=timeframe,
    )
    return normalize_trend_following_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
        entry_price_source=entry_price_source,
    )


def adapt_fib_retracement_signal_events(
    ohlc: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | None = None,
    fib_retracement_pct: float = 61.8,
    entry_price_source: EntryPriceSource = _DEFAULT_ENTRY_PRICE_SOURCE,
    **_: Any,
) -> list[StrategySignalEvent]:
    signals = compute_fib_retracement_signals(
        ohlc,
        fib_retracement_pct=fib_retracement_pct,
        timeframe=timeframe,
    )
    return normalize_fib_retracement_signal_events(
        signals,
        symbol=symbol,
        timeframe=timeframe,
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
register_strategy_signal_adapter("ma_crossover", adapt_ma_crossover_signal_events, replace=True)
register_strategy_signal_adapter("trend_following", adapt_trend_following_signal_events, replace=True)
register_strategy_signal_adapter("fib_retracement", adapt_fib_retracement_signal_events, replace=True)


__all__ = [
    "StrategySignalAdapter",
    "adapt_fib_retracement_signal_events",
    "adapt_ma_crossover_signal_events",
    "adapt_trend_following_signal_events",
    "adapt_orb_signal_events",
    "adapt_msb_signal_events",
    "adapt_sfp_signal_events",
    "build_strategy_signal_events",
    "get_strategy_signal_adapter",
    "list_registered_strategy_signal_adapters",
    "normalize_fib_retracement_signal_events",
    "normalize_ma_crossover_signal_events",
    "normalize_orb_signal_events",
    "normalize_msb_signal_events",
    "normalize_sfp_signal_events",
    "normalize_trend_following_signal_events",
    "register_strategy_signal_adapter",
]
