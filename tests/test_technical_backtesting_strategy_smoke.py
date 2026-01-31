from __future__ import annotations

from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.technicals_backtesting.backtest.runner import run_backtest
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
from options_helper.technicals_backtesting.strategies.mean_reversion_bbands import (
    MeanReversionBollinger,
)
from options_helper.technicals_backtesting.strategies.trend_pullback_atr import TrendPullbackATR
from tests.technical_backtesting_helpers import make_synthetic_ohlc


def test_strategy_backtest_smoke() -> None:
    cfg = load_technical_backtesting_config()
    cfg["data"]["warmup_bars"] = 50
    df = make_synthetic_ohlc(rows=260, seed=9)
    features = compute_features(df, cfg)

    warmup = warmup_bars(cfg)
    trend_cfg = cfg["strategies"]["TrendPullbackATR"]
    stats_trend = run_backtest(
        features,
        TrendPullbackATR,
        cfg["backtest"],
        trend_cfg["defaults"],
        warmup_bars=warmup,
    )
    assert stats_trend is not None

    mean_cfg = cfg["strategies"]["MeanReversionBollinger"]
    stats_mean = run_backtest(
        features,
        MeanReversionBollinger,
        cfg["backtest"],
        mean_cfg["defaults"],
        warmup_bars=warmup,
    )
    assert stats_mean is not None
