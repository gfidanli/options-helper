from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from options_helper.technicals_backtesting.backtest.runner import run_backtest
from options_helper.technicals_backtesting.feature_selection import (
    required_feature_columns_for_strategy,
)
from options_helper.technicals_backtesting.strategies.cvd_divergence_msb import (
    CvdDivergenceMSB,
)
from options_helper.technicals_backtesting.strategies.registry import get_strategy


def _bt_cfg() -> dict:
    return {
        "cash": 100_000,
        "commission": 0.0,
        "trade_on_close": False,
        "exclusive_orders": True,
        "hedging": False,
        "margin": 1.0,
        "slippage_bps": 0.0,
    }


def _strategy_params() -> dict:
    return {
        "atr_window": 14,
        "stop_mult_atr": 1.0,
        "take_profit_mult_atr": 1.0,
        "max_holding_bars": 2,
        "use_weekly_filter": False,
        "cvd_smooth_span": 2,
        "cvd_z_window": 3,
        "pivot_left": 1,
        "pivot_right": 1,
        "divergence_window_bars": 10,
        "min_separation_bars": 2,
        "min_price_delta_pct": 1.0,
        "min_cvd_z_delta": -999.0,
        "max_setup_age_bars": 10,
        "msb_min_distance_bars": 1,
    }


def _cvd_divergence_frame(*, include_volume: bool = True, invalid_volume: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "Open": np.array([10.0, 10.2, 10.4, 10.3, 10.8, 10.5, 12.8, 10.0, 11.0, 12.2, 11.6, 11.4]),
            "High": np.array([10.4, 10.6, 11.2, 10.5, 12.0, 11.0, 13.0, 10.5, 13.0, 14.5, 11.8, 11.6]),
            "Low": np.array([9.8, 9.0, 8.0, 9.2, 10.8, 10.0, 8.2, 9.3, 10.9, 11.5, 11.2, 11.0]),
            "Close": np.array([10.2, 10.4, 10.6, 10.1, 10.5, 10.1, 12.4, 10.2, 12.6, 11.7, 11.5, 11.3]),
            "atr_14": np.full(12, 1.0, dtype=float),
            "weekly_trend_up": np.ones(12, dtype=bool),
        },
        index=idx,
    )
    if include_volume:
        volume = np.array([100, 120, 80, 300, 280, 260, 320, 200, 220, 200, 180, 160], dtype=float)
        if invalid_volume:
            volume[5] = np.nan
        frame["Volume"] = volume
    return frame


def test_cvd_divergence_msb_waits_for_confirmed_pivot_before_entry() -> None:
    frame = _cvd_divergence_frame()
    break_level = float(frame.iloc[2:6]["High"].max())
    assert frame.iloc[6]["Close"] > break_level  # pre-confirmation breakout
    assert frame.iloc[8]["Close"] > break_level  # post-confirmation breakout

    stats = run_backtest(frame, CvdDivergenceMSB, _bt_cfg(), _strategy_params(), warmup_bars=0)
    trades = stats["_trades"]
    assert len(trades) == 1
    assert int(trades.iloc[0]["EntryBar"]) == 9
    assert trades.iloc[0]["EntryTime"] == frame.index[9]


def test_cvd_divergence_msb_missing_volume_warns_and_suppresses_entries() -> None:
    frame = _cvd_divergence_frame(include_volume=False)
    with pytest.warns(UserWarning, match="missing Volume"):
        stats = run_backtest(frame, CvdDivergenceMSB, _bt_cfg(), _strategy_params(), warmup_bars=0)
    assert int(stats["# Trades"]) == 0
    assert stats["_trades"].empty


def test_cvd_divergence_msb_invalid_volume_warns_and_suppresses_entries() -> None:
    frame = _cvd_divergence_frame(invalid_volume=True)
    with pytest.warns(UserWarning, match="invalid/non-finite Volume"):
        stats = run_backtest(frame, CvdDivergenceMSB, _bt_cfg(), _strategy_params(), warmup_bars=0)
    assert int(stats["# Trades"]) == 0
    assert stats["_trades"].empty


def test_cvd_divergence_strategy_registry_and_required_columns() -> None:
    assert get_strategy("CvdDivergenceMSB") is CvdDivergenceMSB
    strat_cfg = {
        "defaults": {"atr_window": 14},
        "search_space": {"atr_window": [10, 14, 21]},
    }
    assert required_feature_columns_for_strategy("CvdDivergenceMSB", strat_cfg) == [
        "atr_10",
        "atr_14",
        "atr_21",
        "weekly_trend_up",
    ]
