from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> float | None:
    if window <= 0 or close.empty or len(close) < window:
        return None
    val = close.rolling(window=window).mean().iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def rsi(close: pd.Series, window: int = 14) -> float | None:
    if window <= 0 or close.empty or len(close) < window + 1:
        return None

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    val = rsi_series.iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def ema(close: pd.Series, span: int) -> float | None:
    if span <= 0 or close.empty or len(close) < span:
        return None
    val = close.ewm(span=span, adjust=False, min_periods=span).mean().iloc[-1]
    if pd.isna(val):
        return None
    return float(val)


def breakout_up(close: pd.Series, lookback: int, *, buffer_pct: float = 0.0) -> bool | None:
    if lookback <= 1 or close.empty or len(close) < lookback + 1:
        return None
    last = float(close.iloc[-1])
    prev_max = float(close.iloc[-(lookback + 1) : -1].max())
    return last > prev_max * (1.0 + buffer_pct)


def breakout_down(close: pd.Series, lookback: int, *, buffer_pct: float = 0.0) -> bool | None:
    if lookback <= 1 or close.empty or len(close) < lookback + 1:
        return None
    last = float(close.iloc[-1])
    prev_min = float(close.iloc[-(lookback + 1) : -1].min())
    return last < prev_min * (1.0 - buffer_pct)
