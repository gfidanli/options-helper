from __future__ import annotations

import math

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """
    Annualized realized volatility from daily log returns.

    Returns a Series aligned to the input index. Values are NaN until the rolling
    window is filled (min_periods=window).
    """
    if close is None:
        return pd.Series(dtype="float64")

    series = pd.to_numeric(close, errors="coerce")
    if series.empty:
        return pd.Series(dtype="float64")

    if window <= 1:
        return pd.Series([float("nan")] * len(series), index=series.index, dtype="float64")

    # Log returns require positive prices; non-positive values become NaN.
    series = series.where(series > 0)
    log_ret = np.log(series).diff()
    rv = log_ret.rolling(window=window, min_periods=window).std() * math.sqrt(252)
    return rv
