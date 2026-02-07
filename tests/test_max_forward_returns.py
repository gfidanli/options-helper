from __future__ import annotations

import pandas as pd

from options_helper.technicals_backtesting.max_forward_returns import (
    forward_max_down_move,
    forward_max_up_move,
    forward_max_up_return,
)


def test_forward_max_up_return_uses_high_and_requires_full_horizon() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    open_ = pd.Series([100, 80, 100, 100, 100, 100], index=idx, dtype="float64")
    high = pd.Series([100, 110, 105, 120, 115, 101], index=idx, dtype="float64")

    # From day 0, entry is next open (80) and next 3 bars have highs 110/105/120 => +50%.
    r3 = forward_max_up_return(open_series=open_, high_series=high, start_iloc=0, horizon_bars=3)
    assert round(float(r3) * 100.0, 1) == 50.0

    # From day 4, we don't have 3 full bars ahead.
    assert forward_max_up_return(open_series=open_, high_series=high, start_iloc=4, horizon_bars=3) is None


def test_forward_max_up_return_handles_bad_inputs() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    open_ = pd.Series([100.0, 0.0, 100.0], index=idx, dtype="float64")
    high = pd.Series([100.0, 101.0, 102.0], index=idx, dtype="float64")

    assert forward_max_up_return(open_series=open_, high_series=high, start_iloc=0, horizon_bars=1) is None
    assert forward_max_up_return(open_series=open_, high_series=high, start_iloc=-1, horizon_bars=1) is None
    assert forward_max_up_return(open_series=open_, high_series=high, start_iloc=0, horizon_bars=0) is None


def test_forward_max_up_move_is_clamped_to_non_negative() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    open_ = pd.Series([100, 100, 100, 100, 100], index=idx, dtype="float64")
    # All highs below next-bar entry open -> raw max-up return is negative, but move should be 0.
    high = pd.Series([99, 99, 98, 97, 96], index=idx, dtype="float64")

    r = forward_max_up_move(open_series=open_, high_series=high, start_iloc=0, horizon_bars=3)
    assert r == 0.0


def test_forward_max_down_move_is_pullback_magnitude() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="B")
    open_ = pd.Series([100, 90, 100, 100, 100, 100], index=idx, dtype="float64")
    low = pd.Series([100, 99, 95, 97, 98, 101], index=idx, dtype="float64")

    # From day 0, entry is next open (90); min low over next 3 bars is 95 so pullback is 0.
    r3 = forward_max_down_move(open_series=open_, low_series=low, start_iloc=0, horizon_bars=3)
    assert r3 == 0.0

    # If lows stay above next-bar entry open, pullback magnitude should be 0.
    low2 = pd.Series([100, 101, 102, 103, 104, 105], index=idx, dtype="float64")
    r3b = forward_max_down_move(open_series=open_, low_series=low2, start_iloc=0, horizon_bars=3)
    assert r3b == 0.0
