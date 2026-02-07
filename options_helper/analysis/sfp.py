from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import re

_REQUIRED_OHLC_COLUMNS = ("Open", "High", "Low", "Close")
_WEEKLY_RULE_RE = re.compile(r"^\d*W(?:-[A-Z]{3})?$")


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
    out["swept_swing_high_level"] = np.nan
    out["swept_swing_low_level"] = np.nan
    out["swept_swing_high_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["swept_swing_low_timestamp"] = pd.Series([None] * n, index=out.index, dtype="object")
    out["bearish_sfp"] = False
    out["bullish_sfp"] = False
    out["sfp"] = pd.Series([None] * n, index=out.index, dtype="object")

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

    def _label(ts: object) -> str:
        if isinstance(ts, pd.Timestamp):
            return ts.isoformat()
        return str(ts)

    for i in range(n):
        # A swing at index k needs `right` future bars to be confirmed.
        # Make it available only when processing bar i where k = i - right.
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
            if age >= min_distance and np.isfinite(high[i]) and np.isfinite(close[i]):
                if high[i] > level and close[i] < level:
                    bearish_sfp[i] = True
                    swept_high_level[i] = level
                    swept_high_ts[i] = _label(frame.index[last_swing_high_idx])

        if last_swing_low_idx is not None:
            level = float(low[last_swing_low_idx])
            age = i - last_swing_low_idx
            last_low_level[i] = level
            bars_since_low[i] = float(age)
            if age >= min_distance and np.isfinite(low[i]) and np.isfinite(close[i]):
                if low[i] < level and close[i] > level:
                    bullish_sfp[i] = True
                    swept_low_level[i] = level
                    swept_low_ts[i] = _label(frame.index[last_swing_low_idx])

    out["swing_high"] = swing_high
    out["swing_low"] = swing_low
    out["swing_high_level"] = np.where(swing_high, high, np.nan)
    out["swing_low_level"] = np.where(swing_low, low, np.nan)
    out["last_swing_high_level"] = last_high_level
    out["last_swing_low_level"] = last_low_level
    out["bars_since_swing_high"] = bars_since_high
    out["bars_since_swing_low"] = bars_since_low
    out["swept_swing_high_level"] = swept_high_level
    out["swept_swing_low_level"] = swept_low_level
    out["swept_swing_high_timestamp"] = pd.Series(swept_high_ts, index=out.index, dtype="object")
    out["swept_swing_low_timestamp"] = pd.Series(swept_low_ts, index=out.index, dtype="object")
    out["bearish_sfp"] = bearish_sfp
    out["bullish_sfp"] = bullish_sfp

    direction = np.full(n, None, dtype=object)
    direction[(bearish_sfp) & (~bullish_sfp)] = "bearish"
    direction[(bullish_sfp) & (~bearish_sfp)] = "bullish"
    direction[(bullish_sfp) & (bearish_sfp)] = "both"
    out["sfp"] = pd.Series(direction, index=out.index, dtype="object")

    return out


def extract_sfp_events(signals: pd.DataFrame) -> list[SfpEvent]:
    if signals is None or signals.empty:
        return []

    events: list[SfpEvent] = []

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

        if bool(row.get("bearish_sfp", False)):
            events.append(
                SfpEvent(
                    timestamp=ts,
                    direction="bearish",
                    candle_open=o,
                    candle_high=h,
                    candle_low=low_val,
                    candle_close=c,
                    sweep_level=_value_or_default(row.get("swept_swing_high_level")),
                    swept_swing_timestamp=str(row.get("swept_swing_high_timestamp")),
                    bars_since_swing=_bars(row.get("bars_since_swing_high")),
                )
            )
        if bool(row.get("bullish_sfp", False)):
            events.append(
                SfpEvent(
                    timestamp=ts,
                    direction="bullish",
                    candle_open=o,
                    candle_high=h,
                    candle_low=low_val,
                    candle_close=c,
                    sweep_level=_value_or_default(row.get("swept_swing_low_level")),
                    swept_swing_timestamp=str(row.get("swept_swing_low_timestamp")),
                    bars_since_swing=_bars(row.get("bars_since_swing_low")),
                )
            )
    return events
