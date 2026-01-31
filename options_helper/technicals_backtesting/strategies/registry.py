from __future__ import annotations

from backtesting import Strategy

from options_helper.technicals_backtesting.strategies.mean_reversion_bbands import (
    MeanReversionBollinger,
)
from options_helper.technicals_backtesting.strategies.trend_pullback_atr import TrendPullbackATR


def get_strategy(name: str) -> type[Strategy]:
    mapping = {
        "TrendPullbackATR": TrendPullbackATR,
        "MeanReversionBollinger": MeanReversionBollinger,
    }
    if name not in mapping:
        raise ValueError(f"Unknown strategy: {name}")
    return mapping[name]

