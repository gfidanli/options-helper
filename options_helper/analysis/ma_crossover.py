from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from options_helper.analysis.sfp import normalize_ohlc_frame, resample_ohlc_frame


_MA_TYPES = {"sma", "ema"}


def compute_ma_crossover_signals(
    ohlc: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    fast_type: str = "sma",
    slow_type: str = "sma",
    atr_window: int = 14,
    atr_stop_multiple: float = 2.0,
    timeframe: str | None = None,
) -> pd.DataFrame:
    fast = int(fast_window)
    slow = int(slow_window)
    atr_w = int(atr_window)
    atr_multiple = float(atr_stop_multiple)
    fast_ma_type = _normalize_ma_type(fast_type, option_name="fast_type")
    slow_ma_type = _normalize_ma_type(slow_type, option_name="slow_type")

    if fast < 1:
        raise ValueError("fast_window must be >= 1")
    if slow < 1:
        raise ValueError("slow_window must be >= 1")
    if fast >= slow:
        raise ValueError("fast_window must be < slow_window")
    if atr_w < 1:
        raise ValueError("atr_window must be >= 1")
    if not np.isfinite(atr_multiple) or atr_multiple <= 0.0:
        raise ValueError("atr_stop_multiple must be > 0")

    frame = normalize_ohlc_frame(ohlc)
    if timeframe and timeframe.lower() not in {"native", "raw", "none"}:
        frame = resample_ohlc_frame(frame, timeframe=timeframe)

    out = frame.copy()
    out["ma_fast"] = np.nan
    out["ma_slow"] = np.nan
    out["atr"] = np.nan
    out["ma_crossover_long"] = False
    out["ma_crossover_short"] = False
    out["ma_fast_window"] = fast
    out["ma_slow_window"] = slow
    out["ma_fast_type"] = fast_ma_type
    out["ma_slow_type"] = slow_ma_type
    out["atr_window"] = atr_w
    out["atr_stop_multiple"] = atr_multiple

    if out.empty:
        return out

    close = pd.to_numeric(out["Close"], errors="coerce").astype("float64")
    high = pd.to_numeric(out["High"], errors="coerce").astype("float64")
    low = pd.to_numeric(out["Low"], errors="coerce").astype("float64")

    fast_ma = _moving_average(close, window=fast, ma_type=fast_ma_type)
    slow_ma = _moving_average(close, window=slow, ma_type=slow_ma_type)
    atr = _atr_series(high=high, low=low, close=close, window=atr_w)

    prev_fast = fast_ma.shift(1)
    prev_slow = slow_ma.shift(1)
    long_cross = prev_fast.le(prev_slow) & fast_ma.gt(slow_ma)
    short_cross = prev_fast.ge(prev_slow) & fast_ma.lt(slow_ma)
    valid = fast_ma.notna() & slow_ma.notna() & atr.notna() & close.notna()

    out["ma_fast"] = fast_ma
    out["ma_slow"] = slow_ma
    out["atr"] = atr
    out["ma_crossover_long"] = long_cross & valid
    out["ma_crossover_short"] = short_cross & valid
    return out


def extract_ma_crossover_signal_candidates(signals: pd.DataFrame | None) -> list[dict[str, Any]]:
    if signals is None or signals.empty:
        return []
    if "ma_crossover_long" not in signals.columns or "ma_crossover_short" not in signals.columns:
        return []

    candidates: list[dict[str, Any]] = []
    for row_position, (idx, row) in enumerate(signals.iterrows()):
        long_flag = bool(row.get("ma_crossover_long", False))
        short_flag = bool(row.get("ma_crossover_short", False))
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
                "fast_ma": _to_float(row.get("ma_fast")),
                "slow_ma": _to_float(row.get("ma_slow")),
                "atr": atr_value,
                "stop_price": stop_price,
                "fast_window": int(row.get("ma_fast_window", 0) or 0),
                "slow_window": int(row.get("ma_slow_window", 0) or 0),
                "fast_type": str(row.get("ma_fast_type") or "").strip().lower(),
                "slow_type": str(row.get("ma_slow_type") or "").strip().lower(),
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
    "compute_ma_crossover_signals",
    "extract_ma_crossover_signal_candidates",
]
