from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame, resample_ohlc_frame


_MA_TYPES = {"sma", "ema"}


def compute_trend_following_signals(
    ohlc: pd.DataFrame,
    *,
    trend_window: int = 200,
    trend_type: str = "sma",
    fast_window: int = 20,
    fast_type: str = "sma",
    slope_lookback_bars: int = 3,
    atr_window: int = 14,
    atr_stop_multiple: float = 2.0,
    timeframe: str | None = None,
) -> pd.DataFrame:
    trend_w = int(trend_window)
    fast_w = int(fast_window)
    slope_lookback = int(slope_lookback_bars)
    atr_w = int(atr_window)
    atr_multiple = float(atr_stop_multiple)
    trend_ma_type = _normalize_ma_type(trend_type, option_name="trend_type")
    fast_ma_type = _normalize_ma_type(fast_type, option_name="fast_type")

    if trend_w < 1:
        raise ValueError("trend_window must be >= 1")
    if fast_w < 1:
        raise ValueError("fast_window must be >= 1")
    if slope_lookback < 1:
        raise ValueError("slope_lookback_bars must be >= 1")
    if atr_w < 1:
        raise ValueError("atr_window must be >= 1")
    if not np.isfinite(atr_multiple) or atr_multiple <= 0.0:
        raise ValueError("atr_stop_multiple must be > 0")

    frame = normalize_ohlc_frame(ohlc)
    if timeframe and timeframe.lower() not in {"native", "raw", "none"}:
        frame = resample_ohlc_frame(frame, timeframe=timeframe)

    out = frame.copy()
    out["trend_ma"] = np.nan
    out["trend_ma_slope"] = np.nan
    out["fast_ma"] = np.nan
    out["atr"] = np.nan
    out["trend_following_long"] = False
    out["trend_following_short"] = False
    out["trend_window"] = trend_w
    out["trend_type"] = trend_ma_type
    out["fast_window"] = fast_w
    out["fast_type"] = fast_ma_type
    out["slope_lookback_bars"] = slope_lookback
    out["atr_window"] = atr_w
    out["atr_stop_multiple"] = atr_multiple

    if out.empty:
        return out

    close = pd.to_numeric(out["Close"], errors="coerce").astype("float64")
    high = pd.to_numeric(out["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(out["Low"], errors="coerce").astype("float64")

    trend_ma = _moving_average(close, window=trend_w, ma_type=trend_ma_type)
    fast_ma = _moving_average(close, window=fast_w, ma_type=fast_ma_type)
    trend_slope = ((trend_ma - trend_ma.shift(slope_lookback)) / float(slope_lookback)).astype("float64")
    atr = _atr_series(high=high, low=low, close=close, window=atr_w)

    prev_close = close.shift(1)
    prev_trend = trend_ma.shift(1)
    cross_up = prev_close.le(prev_trend) & close.gt(trend_ma)
    cross_down = prev_close.ge(prev_trend) & close.lt(trend_ma)
    slope_up = trend_slope.ge(0.0)
    slope_down = trend_slope.le(0.0)
    fast_above_trend = fast_ma.ge(trend_ma)
    fast_below_trend = fast_ma.le(trend_ma)
    valid = close.notna() & trend_ma.notna() & fast_ma.notna() & trend_slope.notna() & atr.notna()

    out["trend_ma"] = trend_ma
    out["trend_ma_slope"] = trend_slope
    out["fast_ma"] = fast_ma
    out["atr"] = atr
    out["trend_following_long"] = cross_up & slope_up & fast_above_trend & valid
    out["trend_following_short"] = cross_down & slope_down & fast_below_trend & valid
    return out


def extract_trend_following_signal_candidates(signals: pd.DataFrame | None) -> list[dict[str, Any]]:
    if signals is None or signals.empty:
        return []
    if "trend_following_long" not in signals.columns or "trend_following_short" not in signals.columns:
        return []

    candidates: list[dict[str, Any]] = []
    for row_position, (idx, row) in enumerate(signals.iterrows()):
        long_flag = bool(row.get("trend_following_long", False))
        short_flag = bool(row.get("trend_following_short", False))
        if long_flag == short_flag:
            continue

        direction = "long" if long_flag else "short"
        close = _to_float(row.get("Close"))
        atr_value = _to_float(row.get("atr"))
        atr_multiple = _to_float(row.get("atr_stop_multiple"))
        if close is None or atr_value is None or atr_multiple is None or atr_multiple <= 0.0:
            continue

        stop_price = (
            close - (atr_multiple * atr_value)
            if direction == "long"
            else close + (atr_multiple * atr_value)
        )
        candidates.append(
            {
                "row_position": row_position,
                "index_value": idx,
                "timestamp": _timestamp_label(idx),
                "direction": direction,
                "candle_open": _to_float(row.get("Open")),
                "candle_high": _to_float(row.get("High")),
                "candle_low": _to_float(row.get("Low")),
                "candle_close": close,
                "trend_ma": _to_float(row.get("trend_ma")),
                "trend_ma_slope": _to_float(row.get("trend_ma_slope")),
                "fast_ma": _to_float(row.get("fast_ma")),
                "atr": atr_value,
                "stop_price": stop_price,
                "trend_window": int(row.get("trend_window", 0) or 0),
                "trend_type": str(row.get("trend_type") or "").strip().lower(),
                "fast_window": int(row.get("fast_window", 0) or 0),
                "fast_type": str(row.get("fast_type") or "").strip().lower(),
                "slope_lookback_bars": int(row.get("slope_lookback_bars", 0) or 0),
                "atr_window": int(row.get("atr_window", 0) or 0),
                "atr_stop_multiple": atr_multiple,
            }
        )

    return candidates


def _moving_average(series: pd.Series, *, window: int, ma_type: str) -> pd.Series:
    if ma_type == "sma":
        return series.rolling(window=window, min_periods=window).mean().astype("float64")
    if ma_type == "ema":
        return series.ewm(span=window, adjust=False, min_periods=window).mean().astype("float64")
    raise ValueError(f"Unsupported ma_type: {ma_type}")


def _atr_series(*, high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean().astype("float64")


def _normalize_ma_type(value: str, *, option_name: str) -> str:
    token = str(value or "").strip().lower()
    if token not in _MA_TYPES:
        allowed = ", ".join(sorted(_MA_TYPES))
        raise ValueError(f"{option_name} must be one of: {allowed}")
    return token


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(number):
        return None
    return number


def _timestamp_label(value: object) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


__all__ = [
    "compute_trend_following_signals",
    "extract_trend_following_signal_candidates",
]
