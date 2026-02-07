from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame, resample_ohlc_frame


@dataclass(frozen=True)
class MsbEvent:
    timestamp: str
    direction: str  # "bullish" | "bearish"
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    break_level: float
    broken_swing_timestamp: str
    bars_since_swing: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_msb_signals(
    ohlc: pd.DataFrame,
    *,
    swing_left_bars: int = 1,
    swing_right_bars: int = 1,
    min_swing_distance_bars: int = 1,
    timeframe: str | None = None,
) -> pd.DataFrame:
    left = int(swing_left_bars)
    right = int(swing_right_bars)
    min_distance = int(min_swing_distance_bars)
    if left < 1:
        raise ValueError("swing_left_bars must be >= 1")
    if right < 1:
        raise ValueError("swing_right_bars must be >= 1")
    if min_distance < 1:
        raise ValueError("min_swing_distance_bars must be >= 1")

    frame = normalize_ohlc_frame(ohlc)
    if timeframe and timeframe.lower() not in {"native", "raw", "none"}:
        frame = resample_ohlc_frame(frame, timeframe=timeframe)

    n = len(frame)
    out = frame.copy()
    out["swing_high"] = False
    out["swing_low"] = False
    out["swing_high_level"] = np.nan
    out["swing_low_level"] = np.nan
    out["last_swing_high_level"] = np.nan
    out["last_swing_low_level"] = np.nan
    out["bars_since_swing_high"] = np.nan
    out["bars_since_swing_low"] = np.nan
    out["broken_swing_high_level"] = np.nan
    out["broken_swing_low_level"] = np.nan
    out["broken_swing_high_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["broken_swing_low_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["bullish_msb"] = False
    out["bearish_msb"] = False
    out["msb"] = pd.Series([None] * n, index=out.index, dtype="object")

    if n == 0:
        return out

    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)

    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        h = high[i]
        low_val = low[i]
        if not np.isfinite(h) or not np.isfinite(low_val):
            continue
        left_high = high[i - left : i]
        right_high = high[i + 1 : i + 1 + right]
        left_low = low[i - left : i]
        right_low = low[i + 1 : i + 1 + right]
        if (
            np.isfinite(left_high).all()
            and np.isfinite(right_high).all()
            and h >= float(np.max(left_high))
            and h >= float(np.max(right_high))
        ):
            swing_high[i] = True
        if (
            np.isfinite(left_low).all()
            and np.isfinite(right_low).all()
            and low_val <= float(np.min(left_low))
            and low_val <= float(np.min(right_low))
        ):
            swing_low[i] = True

    bullish_msb = np.zeros(n, dtype=bool)
    bearish_msb = np.zeros(n, dtype=bool)
    bars_since_high = np.full(n, np.nan, dtype=float)
    bars_since_low = np.full(n, np.nan, dtype=float)
    last_high_level = np.full(n, np.nan, dtype=float)
    last_low_level = np.full(n, np.nan, dtype=float)
    broken_high_level = np.full(n, np.nan, dtype=float)
    broken_low_level = np.full(n, np.nan, dtype=float)
    broken_high_ts: list[str | None] = [None] * n
    broken_low_ts: list[str | None] = [None] * n

    last_swing_high_idx: int | None = None
    last_swing_low_idx: int | None = None

    def _label(ts: object) -> str:
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        return str(ts)

    for i in range(n):
        prev_close = close[i - 1] if i > 0 else np.nan

        # A swing at index k is confirmed only after `right` future bars.
        # Expose it to break logic starting at bar i where k = i - right.
        confirm_idx = i - right
        if confirm_idx >= 0:
            if swing_high[confirm_idx]:
                last_swing_high_idx = confirm_idx
            if swing_low[confirm_idx]:
                last_swing_low_idx = confirm_idx

        if last_swing_high_idx is not None:
            level = float(high[last_swing_high_idx])
            age = i - last_swing_high_idx
            last_high_level[i] = level
            bars_since_high[i] = float(age)
            crossed_above = np.isfinite(close[i]) and close[i] > level
            prev_below_or_equal = i == 0 or (np.isfinite(prev_close) and prev_close <= level)
            if age >= min_distance and crossed_above and prev_below_or_equal:
                bullish_msb[i] = True
                broken_high_level[i] = level
                broken_high_ts[i] = _label(frame.index[last_swing_high_idx])

        if last_swing_low_idx is not None:
            level = float(low[last_swing_low_idx])
            age = i - last_swing_low_idx
            last_low_level[i] = level
            bars_since_low[i] = float(age)
            crossed_below = np.isfinite(close[i]) and close[i] < level
            prev_above_or_equal = i == 0 or (np.isfinite(prev_close) and prev_close >= level)
            if age >= min_distance and crossed_below and prev_above_or_equal:
                bearish_msb[i] = True
                broken_low_level[i] = level
                broken_low_ts[i] = _label(frame.index[last_swing_low_idx])

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["swing_high_level"] = np.where(swing_high, high, np.nan)
    out["swing_low_level"] = np.where(swing_low, low, np.nan)
    out["last_swing_high_level"] = last_high_level
    out["last_swing_low_level"] = last_low_level
    out["bars_since_swing_high"] = bars_since_high
    out["bars_since_swing_low"] = bars_since_low
    out["broken_swing_high_level"] = broken_high_level
    out["broken_swing_low_level"] = broken_low_level
    out["broken_swing_high_timestamp"] = pd.Series(broken_high_ts, index=out.index, dtype="object")
    out["broken_swing_low_timestamp"] = pd.Series(broken_low_ts, index=out.index, dtype="object")
    out["bullish_msb"] = bullish_msb
    out["bearish_msb"] = bearish_msb

    direction = np.full(n, None, dtype=object)
    direction[(bullish_msb) & (~bearish_msb)] = "bullish"
    direction[(bearish_msb) & (~bullish_msb)] = "bearish"
    direction[(bullish_msb) & (bearish_msb)] = "both"
    out["msb"] = pd.Series(direction, index=out.index, dtype="object")

    return out


def extract_msb_events(signals: pd.DataFrame) -> list[MsbEvent]:
    if signals is None or signals.empty:
        return []

    events: list[MsbEvent] = []

    def _value_or_default(value: object, default: float = float("nan")) -> float:
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return default

    def _bars(value: object) -> int:
        try:
            return int(float(value))
        except Exception:  # noqa: BLE001
            return 0

    def _timestamp(value: object) -> str:
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return str(value)

    for idx, row in signals.iterrows():
        ts = _timestamp(idx)
        o = _value_or_default(row.get("Open"))
        h = _value_or_default(row.get("High"))
        low_val = _value_or_default(row.get("Low"))
        c = _value_or_default(row.get("Close"))

        if bool(row.get("bullish_msb", False)):
            events.append(
                MsbEvent(
                    timestamp=ts,
                    direction="bullish",
                    candle_open=o,
                    candle_high=h,
                    candle_low=low_val,
                    candle_close=c,
                    break_level=_value_or_default(row.get("broken_swing_high_level")),
                    broken_swing_timestamp=str(row.get("broken_swing_high_timestamp")),
                    bars_since_swing=_bars(row.get("bars_since_swing_high")),
                )
            )
        if bool(row.get("bearish_msb", False)):
            events.append(
                MsbEvent(
                    timestamp=ts,
                    direction="bearish",
                    candle_open=o,
                    candle_high=h,
                    candle_low=low_val,
                    candle_close=c,
                    break_level=_value_or_default(row.get("broken_swing_low_level")),
                    broken_swing_timestamp=str(row.get("broken_swing_low_timestamp")),
                    bars_since_swing=_bars(row.get("bars_since_swing_low")),
                )
            )

    return events
