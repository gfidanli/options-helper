from __future__ import annotations

import pandas as pd

from options_helper.technicals_backtesting.max_forward_returns import forward_max_up_return


def test_forward_max_up_return_uses_high_and_requires_full_horizon() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    close = pd.Series([100, 100, 100, 100, 100, 100], index=idx, dtype="float64")
    high = pd.Series([100, 110, 105, 120, 115, 101], index=idx, dtype="float64")

    # From day 0, next 3 bars have highs 110/105/120 => +20%.
    r3 = forward_max_up_return(close_series=close, high_series=high, start_iloc=0, horizon_bars=3)
    assert round(float(r3) * 100.0, 1) == 20.0

    # From day 4, we don't have 3 full bars ahead.
    assert forward_max_up_return(close_series=close, high_series=high, start_iloc=4, horizon_bars=3) is None


def test_forward_max_up_return_handles_bad_inputs() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    close = pd.Series([0.0, 100.0, 100.0], index=idx, dtype="float64")
    high = pd.Series([100.0, 101.0, 102.0], index=idx, dtype="float64")

    assert forward_max_up_return(close_series=close, high_series=high, start_iloc=0, horizon_bars=1) is None
    assert forward_max_up_return(close_series=close, high_series=high, start_iloc=-1, horizon_bars=1) is None
    assert forward_max_up_return(close_series=close, high_series=high, start_iloc=0, horizon_bars=0) is None

