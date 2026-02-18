from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame, resample_ohlc_frame


_DIRECTIONAL_SIGNAL_SPECS = (
    (
        "bullish_msb",
        "bullish",
        "broken_swing_high_level",
        "broken_swing_high_timestamp",
        "bars_since_swing_high",
    ),
    (
        "bearish_msb",
        "bearish",
        "broken_swing_low_level",
        "broken_swing_low_timestamp",
        "bars_since_swing_low",
    ),
)


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


@dataclass(frozen=True)
class _MsbBreakArrays:
    bullish_msb: np.ndarray
    bearish_msb: np.ndarray
    bars_since_high: np.ndarray
    bars_since_low: np.ndarray
    last_high_level: np.ndarray
    last_low_level: np.ndarray
    broken_high_level: np.ndarray
    broken_low_level: np.ndarray
    broken_high_ts: list[str | None]
    broken_low_ts: list[str | None]


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

    out = _initialize_msb_output(frame)
    n = len(out)
    if n == 0:
        return out

    high = frame["High"].to_numpy(dtype=float)
    low = frame["Low"].to_numpy(dtype=float)
    close = frame["Close"].to_numpy(dtype=float)
    swing_high, swing_low = _compute_swings(
        high=high,
        low=low,
        left=left,
        right=right,
    )
    breaks = _compute_msb_break_arrays(
        index_values=frame.index.tolist(),
        high=high,
        low=low,
        close=close,
        swing_high=swing_high,
        swing_low=swing_low,
        right=right,
        min_distance=min_distance,
    )
    _apply_msb_arrays(
        out=out,
        high=high,
        low=low,
        swing_high=swing_high,
        swing_low=swing_low,
        breaks=breaks,
    )
    return out


def _initialize_msb_output(frame: pd.DataFrame) -> pd.DataFrame:
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
    return out


def _compute_swings(
    *,
    high: np.ndarray,
    low: np.ndarray,
    left: int,
    right: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(high)
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
    return swing_high, swing_low


def _compute_msb_break_arrays(
    *,
    index_values: list[object],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    right: int,
    min_distance: int,
) -> _MsbBreakArrays:
    n = len(high)
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
        _mark_msb_break(
            idx=i,
            close=close[i],
            prev_close=prev_close,
            swing_idx=last_swing_high_idx,
            swing_levels=high,
            min_distance=min_distance,
            bars_since=bars_since_high,
            last_level=last_high_level,
            break_flag=bullish_msb,
            broken_level=broken_high_level,
            broken_ts=broken_high_ts,
            direction="bullish",
            index_values=index_values,
        )
        _mark_msb_break(
            idx=i,
            close=close[i],
            prev_close=prev_close,
            swing_idx=last_swing_low_idx,
            swing_levels=low,
            min_distance=min_distance,
            bars_since=bars_since_low,
            last_level=last_low_level,
            break_flag=bearish_msb,
            broken_level=broken_low_level,
            broken_ts=broken_low_ts,
            direction="bearish",
            index_values=index_values,
        )
    return _MsbBreakArrays(
        bullish_msb=bullish_msb,
        bearish_msb=bearish_msb,
        bars_since_high=bars_since_high,
        bars_since_low=bars_since_low,
        last_high_level=last_high_level,
        last_low_level=last_low_level,
        broken_high_level=broken_high_level,
        broken_low_level=broken_low_level,
        broken_high_ts=broken_high_ts,
        broken_low_ts=broken_low_ts,
    )


def _mark_msb_break(
    *,
    idx: int,
    close: float,
    prev_close: float,
    swing_idx: int | None,
    swing_levels: np.ndarray,
    min_distance: int,
    bars_since: np.ndarray,
    last_level: np.ndarray,
    break_flag: np.ndarray,
    broken_level: np.ndarray,
    broken_ts: list[str | None],
    direction: str,
    index_values: list[object],
) -> None:
    if swing_idx is None:
        return
    level = float(swing_levels[swing_idx])
    age = idx - swing_idx
    last_level[idx] = level
    bars_since[idx] = float(age)
    crossed = np.isfinite(close) and (close > level if direction == "bullish" else close < level)
    if direction == "bullish":
        prev_guard = idx == 0 or (np.isfinite(prev_close) and prev_close <= level)
    else:
        prev_guard = idx == 0 or (np.isfinite(prev_close) and prev_close >= level)
    if age >= min_distance and crossed and prev_guard:
        break_flag[idx] = True
        broken_level[idx] = level
        broken_ts[idx] = _timestamp_label(index_values[swing_idx])


def _apply_msb_arrays(
    *,
    out: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    breaks: _MsbBreakArrays,
) -> None:
    n = len(out)
    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["swing_high_level"] = np.where(swing_high, high, np.nan)
    out["swing_low_level"] = np.where(swing_low, low, np.nan)
    out["last_swing_high_level"] = breaks.last_high_level
    out["last_swing_low_level"] = breaks.last_low_level
    out["bars_since_swing_high"] = breaks.bars_since_high
    out["bars_since_swing_low"] = breaks.bars_since_low
    out["broken_swing_high_level"] = breaks.broken_high_level
    out["broken_swing_low_level"] = breaks.broken_low_level
    out["broken_swing_high_timestamp"] = pd.Series(breaks.broken_high_ts, index=out.index, dtype="object")
    out["broken_swing_low_timestamp"] = pd.Series(breaks.broken_low_ts, index=out.index, dtype="object")
    out["bullish_msb"] = breaks.bullish_msb
    out["bearish_msb"] = breaks.bearish_msb
    direction = np.full(n, None, dtype=object)
    direction[(breaks.bullish_msb) & (~breaks.bearish_msb)] = "bullish"
    direction[(breaks.bearish_msb) & (~breaks.bullish_msb)] = "bearish"
    direction[(breaks.bullish_msb) & (breaks.bearish_msb)] = "both"
    out["msb"] = pd.Series(direction, index=out.index, dtype="object")


def _timestamp_label(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def extract_msb_events(signals: pd.DataFrame) -> list[MsbEvent]:
    rows = extract_msb_signal_candidates(signals)
    return [
        MsbEvent(
            timestamp=str(row["timestamp"]),
            direction=str(row["direction"]),
            candle_open=float(row["candle_open"]),
            candle_high=float(row["candle_high"]),
            candle_low=float(row["candle_low"]),
            candle_close=float(row["candle_close"]),
            break_level=float(row["break_level"]),
            broken_swing_timestamp=str(row["broken_swing_timestamp"]),
            bars_since_swing=int(row["bars_since_swing"]),
        )
        for row in rows
    ]


def extract_msb_signal_candidates(signals: pd.DataFrame) -> list[dict[str, Any]]:
    if signals is None or signals.empty:
        return []

    candidates: list[dict[str, Any]] = []

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

    for row_position, (idx, row) in enumerate(signals.iterrows()):
        ts = _timestamp(idx)
        o = _value_or_default(row.get("Open"))
        h = _value_or_default(row.get("High"))
        low_val = _value_or_default(row.get("Low"))
        c = _value_or_default(row.get("Close"))

        for (
            flag_column,
            direction,
            break_level_column,
            broken_timestamp_column,
            bars_column,
        ) in _DIRECTIONAL_SIGNAL_SPECS:
            if not bool(row.get(flag_column, False)):
                continue
            candidates.append(
                {
                    "row_position": row_position,
                    "index_value": idx,
                    "timestamp": ts,
                    "direction": direction,
                    "candle_open": o,
                    "candle_high": h,
                    "candle_low": low_val,
                    "candle_close": c,
                    "break_level": _value_or_default(row.get(break_level_column)),
                    "broken_swing_timestamp": str(row.get(broken_timestamp_column)),
                    "bars_since_swing": _bars(row.get(bars_column)),
                }
            )
    return candidates
