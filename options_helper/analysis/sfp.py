from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import re

_REQUIRED_OHLC_COLUMNS = ("Open", "High", "Low", "Close")
_WEEKLY_RULE_RE = re.compile(r"^\d*W(?:-[A-Z]{3})?$")
_DIRECTIONAL_SIGNAL_SPECS = (
    ("bearish_sfp", "bearish", "swept_swing_high_level", "swept_swing_high_timestamp", "bars_since_swing_high"),
    ("bullish_sfp", "bullish", "swept_swing_low_level", "swept_swing_low_timestamp", "bars_since_swing_low"),
)


def _is_weekly_rule(timeframe: str) -> bool:
    return bool(_WEEKLY_RULE_RE.match(str(timeframe).strip().upper()))


def _week_start_monday(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=int(ts.weekday()))).normalize()


@dataclass(frozen=True)
class SfpEvent:
    timestamp: str
    direction: str  # "bearish" | "bullish"
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    sweep_level: float
    swept_swing_timestamp: str
    bars_since_swing: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _SfpBreakArrays:
    bearish_sfp: np.ndarray
    bullish_sfp: np.ndarray
    bars_since_high: np.ndarray
    bars_since_low: np.ndarray
    last_high_level: np.ndarray
    last_low_level: np.ndarray
    swept_high_level: np.ndarray
    swept_low_level: np.ndarray
    swept_high_ts: list[str | None]
    swept_low_ts: list[str | None]


def normalize_ohlc_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    if ohlc is None or ohlc.empty:
        return pd.DataFrame(columns=list(_REQUIRED_OHLC_COLUMNS))

    alias_map = {str(col).lower(): col for col in ohlc.columns}
    selected: dict[str, str] = {}
    for name in _REQUIRED_OHLC_COLUMNS:
        exact = name if name in ohlc.columns else None
        alias = alias_map.get(name.lower())
        source = exact or alias
        if source is None:
            raise ValueError(f"Missing OHLC column: {name}")
        selected[name] = source

    frame = ohlc[[selected["Open"], selected["High"], selected["Low"], selected["Close"]]].copy()
    frame.columns = list(_REQUIRED_OHLC_COLUMNS)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.sort_index()
    return frame


def resample_ohlc_frame(ohlc: pd.DataFrame, *, timeframe: str) -> pd.DataFrame:
    frame = normalize_ohlc_frame(ohlc)
    if frame.empty:
        return frame
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("OHLC index must be a DatetimeIndex to resample by timeframe.")
    if not timeframe:
        raise ValueError("timeframe must be non-empty when resampling OHLC data.")

    resampled = (
        frame.resample(timeframe)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna(subset=["Open", "High", "Low", "Close"])
    )
    # Weekly bars are normalized to Monday labels so artifacts align with charting tools
    # that label each weekly candle by the first day of its covered week.
    if _is_weekly_rule(timeframe) and not resampled.empty:
        resampled.index = pd.DatetimeIndex([_week_start_monday(ts) for ts in resampled.index])
        resampled = resampled.sort_index()
    return resampled


def compute_sfp_signals(
    ohlc: pd.DataFrame,
    *,
    swing_left_bars: int = 1,   # 1 = nearest candle, 2 = next 2 candles
    swing_right_bars: int = 1,  # 1 = nearest candle, 2 = next 2 candles
    min_swing_distance_bars: int = 1,
    ignore_swept_swings: bool = False,
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

    out = _initialize_sfp_output(frame)
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
    breaks = _compute_sfp_break_arrays(
        index_values=frame.index.tolist(),
        high=high,
        low=low,
        close=close,
        swing_high=swing_high,
        swing_low=swing_low,
        right=right,
        min_distance=min_distance,
        ignore_swept_swings=ignore_swept_swings,
    )
    _apply_sfp_arrays(
        out=out,
        high=high,
        low=low,
        swing_high=swing_high,
        swing_low=swing_low,
        breaks=breaks,
    )
    return out


def _initialize_sfp_output(frame: pd.DataFrame) -> pd.DataFrame:
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
    out["swept_swing_high_level"] = np.nan
    out["swept_swing_low_level"] = np.nan
    out["swept_swing_high_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["swept_swing_low_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["bearish_sfp"] = False
    out["bullish_sfp"] = False
    out["sfp"] = pd.Series([None] * n, index=out.index, dtype="object")
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


def _compute_sfp_break_arrays(
    *,
    index_values: list[object],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    right: int,
    min_distance: int,
    ignore_swept_swings: bool,
) -> _SfpBreakArrays:
    n = len(high)
    bearish_sfp = np.zeros(n, dtype=bool)
    bullish_sfp = np.zeros(n, dtype=bool)
    bars_since_high = np.full(n, np.nan, dtype=float)
    bars_since_low = np.full(n, np.nan, dtype=float)
    last_high_level = np.full(n, np.nan, dtype=float)
    last_low_level = np.full(n, np.nan, dtype=float)
    swept_high_level = np.full(n, np.nan, dtype=float)
    swept_low_level = np.full(n, np.nan, dtype=float)
    swept_high_ts: list[str | None] = [None] * n
    swept_low_ts: list[str | None] = [None] * n
    last_swing_high_idx: int | None = None
    last_swing_low_idx: int | None = None

    for i in range(n):
        # A swing at index k needs `right` future bars to be confirmed.
        # Make it available only when processing bar i where k = i - right.
        confirm_idx = i - right
        if confirm_idx >= 0:
            if swing_high[confirm_idx]:
                last_swing_high_idx = confirm_idx
            if swing_low[confirm_idx]:
                last_swing_low_idx = confirm_idx
        last_swing_high_idx = _mark_sfp_break(
            idx=i,
            extreme_price=high[i],
            close_price=close[i],
            swing_idx=last_swing_high_idx,
            swing_levels=high,
            min_distance=min_distance,
            bars_since=bars_since_high,
            last_level=last_high_level,
            signal_flag=bearish_sfp,
            swept_level=swept_high_level,
            swept_ts=swept_high_ts,
            direction="bearish",
            index_values=index_values,
            ignore_swept_swings=ignore_swept_swings,
        )
        last_swing_low_idx = _mark_sfp_break(
            idx=i,
            extreme_price=low[i],
            close_price=close[i],
            swing_idx=last_swing_low_idx,
            swing_levels=low,
            min_distance=min_distance,
            bars_since=bars_since_low,
            last_level=last_low_level,
            signal_flag=bullish_sfp,
            swept_level=swept_low_level,
            swept_ts=swept_low_ts,
            direction="bullish",
            index_values=index_values,
            ignore_swept_swings=ignore_swept_swings,
        )
    return _SfpBreakArrays(
        bearish_sfp=bearish_sfp,
        bullish_sfp=bullish_sfp,
        bars_since_high=bars_since_high,
        bars_since_low=bars_since_low,
        last_high_level=last_high_level,
        last_low_level=last_low_level,
        swept_high_level=swept_high_level,
        swept_low_level=swept_low_level,
        swept_high_ts=swept_high_ts,
        swept_low_ts=swept_low_ts,
    )


def _mark_sfp_break(
    *,
    idx: int,
    extreme_price: float,
    close_price: float,
    swing_idx: int | None,
    swing_levels: np.ndarray,
    min_distance: int,
    bars_since: np.ndarray,
    last_level: np.ndarray,
    signal_flag: np.ndarray,
    swept_level: np.ndarray,
    swept_ts: list[str | None],
    direction: str,
    index_values: list[object],
    ignore_swept_swings: bool,
) -> int | None:
    if swing_idx is None:
        return None
    level = float(swing_levels[swing_idx])
    age = idx - swing_idx
    last_level[idx] = level
    bars_since[idx] = float(age)
    if age < min_distance or not np.isfinite(extreme_price) or not np.isfinite(close_price):
        return swing_idx
    if direction == "bearish":
        triggered = extreme_price > level and close_price < level
    else:
        triggered = extreme_price < level and close_price > level
    if not triggered:
        return swing_idx
    signal_flag[idx] = True
    swept_level[idx] = level
    swept_ts[idx] = _timestamp_label(index_values[swing_idx])
    return None if ignore_swept_swings else swing_idx


def _apply_sfp_arrays(
    *,
    out: pd.DataFrame,
    high: np.ndarray,
    low: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    breaks: _SfpBreakArrays,
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
    out["swept_swing_high_level"] = breaks.swept_high_level
    out["swept_swing_low_level"] = breaks.swept_low_level
    out["swept_swing_high_timestamp"] = pd.Series(breaks.swept_high_ts, index=out.index, dtype="object")
    out["swept_swing_low_timestamp"] = pd.Series(breaks.swept_low_ts, index=out.index, dtype="object")
    out["bearish_sfp"] = breaks.bearish_sfp
    out["bullish_sfp"] = breaks.bullish_sfp
    direction = np.full(n, None, dtype=object)
    direction[(breaks.bearish_sfp) & (~breaks.bullish_sfp)] = "bearish"
    direction[(breaks.bullish_sfp) & (~breaks.bearish_sfp)] = "bullish"
    direction[(breaks.bullish_sfp) & (breaks.bearish_sfp)] = "both"
    out["sfp"] = pd.Series(direction, index=out.index, dtype="object")


def _timestamp_label(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def extract_sfp_events(signals: pd.DataFrame) -> list[SfpEvent]:
    rows = extract_sfp_signal_candidates(signals)
    return [
        SfpEvent(
            timestamp=str(row["timestamp"]),
            direction=str(row["direction"]),
            candle_open=float(row["candle_open"]),
            candle_high=float(row["candle_high"]),
            candle_low=float(row["candle_low"]),
            candle_close=float(row["candle_close"]),
            sweep_level=float(row["sweep_level"]),
            swept_swing_timestamp=str(row["swept_swing_timestamp"]),
            bars_since_swing=int(row["bars_since_swing"]),
        )
        for row in rows
    ]


def extract_sfp_signal_candidates(signals: pd.DataFrame) -> list[dict[str, Any]]:
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
            sweep_level_column,
            swept_timestamp_column,
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
                    "sweep_level": _value_or_default(row.get(sweep_level_column)),
                    "swept_swing_timestamp": str(row.get(swept_timestamp_column)),
                    "bars_since_swing": _bars(row.get(bars_column)),
                }
            )
    return candidates
