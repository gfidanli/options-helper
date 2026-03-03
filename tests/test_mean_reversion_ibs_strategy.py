from __future__ import annotations

import numpy as np
import pandas as pd

from options_helper.technicals_backtesting.backtest.runner import run_backtest
from options_helper.technicals_backtesting.feature_selection import (
    required_feature_columns_for_strategy,
)
from options_helper.technicals_backtesting.strategies.mean_reversion_ibs import (
    MeanReversionIBS,
    _compute_ibs,
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
        "lookback_high": 3,
        "range_window": 3,
        "range_mult": 2.5,
        "ibs_threshold": 0.3,
        "exit_lookback": 1,
    }


def _entry_exit_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "Open": np.array([9.6, 9.6, 9.6, 9.7, 9.6, 12.7, 9.8, 9.2, 9.2, 10.2, 10.1, 10.0]),
            "High": np.array([10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 10.0, 10.0, 10.0, 10.4, 10.8, 10.4]),
            "Low": np.array([9.0, 9.0, 9.0, 9.0, 9.0, 12.5, 9.0, 9.0, 9.0, 9.8, 9.9, 9.8]),
            "Close": np.array([9.5, 9.5, 9.5, 9.6, 9.5, 12.8, 9.7, 9.1, 9.2, 10.3, 10.2, 10.1]),
        },
        index=index,
    )


def _zero_range_frame() -> pd.DataFrame:
    index = pd.date_range("2024-02-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "Open": np.array([9.6, 9.6, 9.6, 9.7, 9.6, 12.7, 9.8, 9.0, 9.9, 10.1, 10.0, 10.0]),
            "High": np.array([10.0, 10.0, 10.0, 10.0, 10.0, 13.0, 10.0, 9.0, 10.1, 10.4, 10.5, 10.3]),
            "Low": np.array([9.0, 9.0, 9.0, 9.0, 9.0, 12.5, 9.0, 9.0, 9.1, 9.7, 9.8, 9.8]),
            "Close": np.array([9.5, 9.5, 9.5, 9.6, 9.5, 12.8, 9.8, 9.0, 9.9, 10.2, 10.1, 10.0]),
        },
        index=index,
    )


def test_mean_reversion_ibs_entry_exit_contract() -> None:
    frame = _entry_exit_frame()
    stats = run_backtest(frame, MeanReversionIBS, _bt_cfg(), _strategy_params(), warmup_bars=0)
    trades = stats["_trades"]
    assert len(trades) == 1
    assert int(trades.iloc[0]["EntryBar"]) == 8
    assert trades.iloc[0]["EntryTime"] == frame.index[8]
    assert int(trades.iloc[0]["ExitBar"]) == 10
    assert trades.iloc[0]["ExitTime"] == frame.index[10]


def test_mean_reversion_ibs_zero_range_uses_neutral_fallback() -> None:
    ibs = _compute_ibs(
        close=np.array([9.5, 9.0]),
        low=np.array([9.0, 9.0]),
        high=np.array([11.0, 9.0]),
        zero_range_value=0.5,
    )
    assert ibs[0] == 0.25
    assert ibs[1] == 0.5

    stats = run_backtest(
        _zero_range_frame(),
        MeanReversionIBS,
        _bt_cfg(),
        _strategy_params(),
        warmup_bars=0,
    )
    assert int(stats["# Trades"]) == 0
    assert stats["_trades"].empty


def test_mean_reversion_ibs_registry_and_required_columns() -> None:
    assert get_strategy("MeanReversionIBS") is MeanReversionIBS
    strat_cfg = {
        "defaults": {
            "lookback_high": 10,
            "range_window": 25,
            "range_mult": 2.5,
            "ibs_threshold": 0.3,
            "exit_lookback": 1,
        },
        "search_space": {
            "lookback_high": [8, 10, 12],
            "range_window": [20, 25, 30],
            "range_mult": [2.0, 2.5, 3.0],
            "ibs_threshold": [0.2, 0.3, 0.35],
            "exit_lookback": [1],
        },
    }
    assert required_feature_columns_for_strategy("MeanReversionIBS", strat_cfg) == [
        "Close",
        "High",
        "Low",
    ]
