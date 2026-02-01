from __future__ import annotations

from typing import Iterable

import pandas as pd

from options_helper.technicals_backtesting.adapter import apply_yfinance_price_adjustment, standardize_ohlc
from options_helper.technicals_backtesting.indicators.registry import get_provider


def compute_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    candles_cfg = cfg["data"]["candles"]
    price_adj = candles_cfg.get("price_adjustment", {}) or {}
    df = apply_yfinance_price_adjustment(
        df,
        auto_adjust=bool(price_adj.get("auto_adjust", False)),
        back_adjust=bool(price_adj.get("back_adjust", False)),
    )
    candles = standardize_ohlc(
        df,
        required_cols=candles_cfg["required_columns"],
        optional_cols=candles_cfg.get("optional_columns", []),
        dropna_ohlc=candles_cfg["dropna_ohlc"],
    )
    provider = get_provider(cfg["indicators"]["provider"])
    return provider.compute_indicators(candles, cfg)


def warmup_bars(cfg: dict) -> int:
    return int(cfg["data"].get("warmup_bars", 0))
