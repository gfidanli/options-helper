from __future__ import annotations

import math

import numpy as np
import pandas as pd

from options_helper.analysis.volatility import realized_vol


def test_realized_vol_matches_log_return_std() -> None:
    close = pd.Series([100.0, 110.0, 100.0, 110.0, 100.0], dtype="float64")
    window = 3

    log_ret = np.log(close).diff()
    expected = log_ret.rolling(window=window, min_periods=window).std() * math.sqrt(252)

    result = realized_vol(close, window)
    pd.testing.assert_series_equal(result, expected)


def test_realized_vol_constant_series_is_zero_after_window() -> None:
    close = pd.Series([100.0] * 30, dtype="float64")
    rv = realized_vol(close, 20)
    assert math.isnan(rv.iloc[0])
    assert rv.iloc[-1] == 0.0


def test_realized_vol_handles_non_positive_prices() -> None:
    close = pd.Series([100.0, 0.0, -1.0, 100.0], dtype="float64")
    rv = realized_vol(close, 2)
    assert rv.isna().sum() >= 1


def test_realized_vol_small_or_invalid_window() -> None:
    close = pd.Series([100.0, 101.0, 102.0], dtype="float64")
    rv = realized_vol(close, 1)
    assert rv.isna().all()
