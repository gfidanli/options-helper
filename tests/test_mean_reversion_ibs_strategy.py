from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def _short_entry_exit_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=12, freq="D")
    return pd.DataFrame(
        {
            "Open": np.array([8.5, 8.5, 8.5, 8.5, 8.5, 6.2, 8.4, 9.9, 9.8, 8.5, 8.0, 8.1]),
            "High": np.array([9.0, 9.0, 9.0, 9.0, 9.0, 6.5, 9.0, 10.0, 10.0, 8.8, 8.4, 8.5]),
            "Low": np.array([8.0, 8.0, 8.0, 8.0, 8.0, 6.0, 8.0, 9.0, 9.0, 8.2, 7.7, 7.8]),
            "Close": np.array([8.5, 8.5, 8.5, 8.5, 8.5, 6.2, 8.4, 9.9, 9.7, 8.3, 8.0, 8.2]),
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


def _overlay_gate_frame(*, weekly_trend_at_entry: bool = True) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "Open": np.array([8.0, 8.0, 8.0, 8.0, 8.0, 12.7, 9.8, 9.2, 9.2, 10.2, 10.1, 10.0]),
            "High": np.array([9.0, 9.0, 9.0, 9.0, 9.0, 13.0, 10.0, 10.0, 10.0, 10.4, 10.8, 10.4]),
            "Low": np.array([7.0, 7.0, 7.0, 7.0, 7.0, 12.5, 9.0, 9.0, 9.0, 9.8, 9.9, 9.8]),
            "Close": np.array([8.0, 8.0, 8.0, 8.0, 8.0, 12.8, 9.7, 9.1, 9.2, 10.3, 10.2, 10.1]),
        },
        index=index,
    )
    weekly_trend = np.ones(len(frame), dtype=bool)
    weekly_trend[7] = weekly_trend_at_entry
    frame["weekly_trend_up"] = weekly_trend
    return frame


def _run(frame: pd.DataFrame, **param_overrides: object):
    trade_on_close = bool(param_overrides.pop("trade_on_close", False))
    params = _strategy_params()
    params.update(param_overrides)
    bt_cfg = _bt_cfg()
    bt_cfg["trade_on_close"] = trade_on_close
    return run_backtest(frame, MeanReversionIBS, bt_cfg, params, warmup_bars=0)


def _contract_terms(frame: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    high = frame["High"]
    low = frame["Low"]
    close = frame["Close"]
    intrabar_range = high - low
    rolling_high = high.rolling(window=params["lookback_high"], min_periods=params["lookback_high"]).max()
    rolling_low = low.rolling(window=params["lookback_high"], min_periods=params["lookback_high"]).min()
    avg_range = intrabar_range.rolling(window=params["range_window"], min_periods=params["range_window"]).mean()
    ibs = pd.Series(
        _compute_ibs(
            close=close.to_numpy(dtype=float),
            low=low.to_numpy(dtype=float),
            high=high.to_numpy(dtype=float),
            zero_range_value=0.5,
        ),
        index=frame.index,
    )
    long_threshold = rolling_high - params["range_mult"] * avg_range
    short_threshold = rolling_low + params["range_mult"] * avg_range
    long_entry_signal = (close < long_threshold) & (ibs < params["ibs_threshold"])
    short_entry_signal = (close > short_threshold) & (ibs > (1.0 - params["ibs_threshold"]))
    long_exit_signal = close > high.shift(params["exit_lookback"])
    short_exit_signal = close < low.shift(params["exit_lookback"])
    return {
        "rolling_high": rolling_high,
        "rolling_low": rolling_low,
        "avg_range": avg_range,
        "ibs": ibs,
        "long_threshold": long_threshold,
        "short_threshold": short_threshold,
        "long_entry_signal": long_entry_signal,
        "short_entry_signal": short_entry_signal,
        "long_exit_signal": long_exit_signal,
        "short_exit_signal": short_exit_signal,
    }


def test_mean_reversion_ibs_entry_exit_contract() -> None:
    frame = _entry_exit_frame()
    params = _strategy_params()
    terms = _contract_terms(frame, params)

    signal_bar = 7
    exit_signal_bar = 9
    assert terms["rolling_high"].iloc[signal_bar] == pytest.approx(13.0)
    assert terms["avg_range"].iloc[signal_bar] == pytest.approx(5.0 / 6.0)
    assert terms["long_threshold"].iloc[signal_bar] == pytest.approx(13.0 - (params["range_mult"] * (5.0 / 6.0)))
    assert terms["ibs"].iloc[signal_bar] == pytest.approx(0.1)
    assert bool(terms["long_entry_signal"].iloc[signal_bar])
    assert not bool(terms["long_entry_signal"].iloc[signal_bar - 1])
    assert bool(terms["long_exit_signal"].iloc[exit_signal_bar])
    assert not bool(terms["long_exit_signal"].iloc[exit_signal_bar - 1])

    stats = _run(frame, trade_on_close=False)
    trades = stats["_trades"]
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert int(trade["EntryBar"]) == signal_bar + 1
    assert trade["EntryTime"] == frame.index[signal_bar + 1]
    assert trade["EntryPrice"] == pytest.approx(frame["Open"].iloc[signal_bar + 1])
    assert int(trade["ExitBar"]) == exit_signal_bar + 1
    assert trade["ExitTime"] == frame.index[exit_signal_bar + 1]
    assert trade["ExitPrice"] == pytest.approx(frame["Open"].iloc[exit_signal_bar + 1])
    assert float(trade["Size"]) > 0.0


def test_mean_reversion_ibs_short_entry_exit_contract() -> None:
    frame = _short_entry_exit_frame()
    params = _strategy_params()
    terms = _contract_terms(frame, params)

    signal_bar = 7
    exit_signal_bar = 9
    assert terms["rolling_low"].iloc[signal_bar] == pytest.approx(6.0)
    assert terms["avg_range"].iloc[signal_bar] == pytest.approx((0.5 + 1.0 + 1.0) / 3.0)
    assert terms["short_threshold"].iloc[signal_bar] == pytest.approx(6.0 + (params["range_mult"] * ((0.5 + 1.0 + 1.0) / 3.0)))
    assert terms["ibs"].iloc[signal_bar] == pytest.approx(0.9)
    assert bool(terms["short_entry_signal"].iloc[signal_bar])
    assert not bool(terms["short_entry_signal"].iloc[signal_bar - 1])
    assert bool(terms["short_exit_signal"].iloc[exit_signal_bar])
    assert not bool(terms["short_exit_signal"].iloc[exit_signal_bar - 1])

    stats = _run(frame, trade_on_close=False)
    trades = stats["_trades"]
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert int(trade["EntryBar"]) == signal_bar + 1
    assert trade["EntryTime"] == frame.index[signal_bar + 1]
    assert trade["EntryPrice"] == pytest.approx(frame["Open"].iloc[signal_bar + 1])
    assert int(trade["ExitBar"]) == exit_signal_bar + 1
    assert trade["ExitTime"] == frame.index[exit_signal_bar + 1]
    assert trade["ExitPrice"] == pytest.approx(frame["Open"].iloc[exit_signal_bar + 1])
    assert float(trade["Size"]) < 0.0


def test_mean_reversion_ibs_entry_size_fraction_defaults_to_full_equity() -> None:
    frame = _entry_exit_frame()
    stats_default = _run(frame, trade_on_close=False)
    stats_half = _run(frame, trade_on_close=False, entry_size_fraction=0.5)

    trade_default = stats_default["_trades"].iloc[0]
    trade_half = stats_half["_trades"].iloc[0]
    assert abs(float(trade_default["Size"])) > abs(float(trade_half["Size"]))


def test_mean_reversion_ibs_no_next_open_no_fill_when_signal_on_last_bar() -> None:
    frame = _entry_exit_frame().iloc[:8]
    terms = _contract_terms(frame, _strategy_params())
    last_bar = len(frame) - 1
    assert bool(terms["long_entry_signal"].iloc[last_bar])

    stats = _run(frame, trade_on_close=False)
    assert int(stats["# Trades"]) == 0
    assert stats["_trades"].empty


def test_mean_reversion_ibs_zero_range_uses_neutral_fallback() -> None:
    ibs = _compute_ibs(
        close=np.array([9.5, 9.0]),
        low=np.array([9.0, 9.0]),
        high=np.array([11.0, 9.0]),
        zero_range_value=0.5,
    )
    assert ibs[0] == 0.25
    assert ibs[1] == 0.5

    frame = _zero_range_frame()
    terms = _contract_terms(frame, _strategy_params())
    zero_range_bar = 7
    intrabar_range = frame["High"] - frame["Low"]
    close_below_threshold = frame["Close"] < terms["long_threshold"]
    assert intrabar_range.iloc[zero_range_bar] == pytest.approx(0.0)
    assert terms["ibs"].iloc[zero_range_bar] == pytest.approx(0.5)
    assert bool(close_below_threshold.iloc[zero_range_bar])
    assert not bool(terms["long_entry_signal"].iloc[zero_range_bar])

    stats = _run(frame)
    assert int(stats["# Trades"]) == 0
    assert stats["_trades"].empty


def test_mean_reversion_ibs_sma_trend_gate_accepts_and_rejects() -> None:
    accepted = _run(_overlay_gate_frame(), use_sma_trend_gate=True, sma_trend_window=7)
    assert int(accepted["# Trades"]) == 1

    rejected = _run(_entry_exit_frame(), use_sma_trend_gate=True, sma_trend_window=3)
    assert int(rejected["# Trades"]) == 0


def test_mean_reversion_ibs_weekly_trend_gate_accepts_and_rejects() -> None:
    accepted = _run(_overlay_gate_frame(weekly_trend_at_entry=True), use_weekly_trend_gate=True)
    assert int(accepted["# Trades"]) == 1

    rejected = _run(_overlay_gate_frame(weekly_trend_at_entry=False), use_weekly_trend_gate=True)
    assert int(rejected["# Trades"]) == 0


def test_mean_reversion_ibs_ma_direction_gate_accepts_and_rejects() -> None:
    accepted = _run(
        _overlay_gate_frame(),
        use_ma_direction_gate=True,
        ma_direction_window=6,
        ma_direction_lookback=1,
    )
    assert int(accepted["# Trades"]) == 1

    rejected = _run(
        _entry_exit_frame(),
        use_ma_direction_gate=True,
        ma_direction_window=6,
        ma_direction_lookback=1,
    )
    assert int(rejected["# Trades"]) == 0


def test_mean_reversion_ibs_overlay_gates_use_and_logic() -> None:
    accepted = _run(
        _overlay_gate_frame(weekly_trend_at_entry=True),
        use_sma_trend_gate=True,
        sma_trend_window=7,
        use_weekly_trend_gate=True,
        use_ma_direction_gate=True,
        ma_direction_window=6,
        ma_direction_lookback=1,
    )
    assert int(accepted["# Trades"]) == 1

    rejected = _run(
        _overlay_gate_frame(weekly_trend_at_entry=False),
        use_sma_trend_gate=True,
        sma_trend_window=7,
        use_weekly_trend_gate=True,
        use_ma_direction_gate=True,
        ma_direction_window=6,
        ma_direction_lookback=1,
    )
    assert int(rejected["# Trades"]) == 0


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"use_sma_trend_gate": True, "sma_trend_window": 0}, "sma_trend_window must be an integer >= 1"),
        (
            {"use_ma_direction_gate": True, "ma_direction_lookback": 0},
            "ma_direction_lookback must be an integer >= 1",
        ),
        ({"entry_size_fraction": 1.1}, "entry_size_fraction must be > 0 and <= 1"),
        ({"use_weekly_trend_gate": "yes"}, "use_weekly_trend_gate must be a boolean"),
    ],
)
def test_mean_reversion_ibs_overlay_param_validation(overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _run(_entry_exit_frame(), **overrides)


def test_mean_reversion_ibs_weekly_gate_requires_column() -> None:
    with pytest.raises(ValueError, match="weekly trend gate requires column 'weekly_trend_up'"):
        _run(_entry_exit_frame(), use_weekly_trend_gate=True)


def test_mean_reversion_ibs_registry_and_required_columns() -> None:
    assert get_strategy("MeanReversionIBS") is MeanReversionIBS
    strat_cfg = {
        "defaults": {
            "lookback_high": 10,
            "range_window": 25,
            "range_mult": 2.5,
            "ibs_threshold": 0.3,
            "exit_lookback": 1,
            "use_sma_trend_gate": True,
            "sma_trend_window": 200,
            "use_weekly_trend_gate": True,
            "use_ma_direction_gate": True,
            "ma_direction_window": 200,
            "ma_direction_lookback": 1,
        },
        "search_space": {
            "lookback_high": [8, 10, 12],
            "range_window": [20, 25, 30],
            "range_mult": [2.0, 2.5, 3.0],
            "ibs_threshold": [0.2, 0.3, 0.35],
            "exit_lookback": [1],
            "use_sma_trend_gate": [False, True],
            "sma_trend_window": [150, 200],
            "use_weekly_trend_gate": [False, True],
            "use_ma_direction_gate": [False, True],
            "ma_direction_window": [100, 200],
            "ma_direction_lookback": [1, 2],
        },
    }
    assert required_feature_columns_for_strategy("MeanReversionIBS", strat_cfg) == [
        "Close",
        "High",
        "Low",
        "weekly_trend_up",
    ]
