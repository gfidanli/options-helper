from __future__ import annotations

import inspect
from typing import Iterable

import pandas as pd
from backtesting import Backtest, Strategy


def _bt_kwargs(bt_cfg: dict) -> dict:
    sig = inspect.signature(Backtest.__init__)
    params = {
        "cash": bt_cfg["cash"],
        "commission": bt_cfg["commission"],
        "trade_on_close": bt_cfg["trade_on_close"],
        "exclusive_orders": bt_cfg["exclusive_orders"],
        "hedging": bt_cfg["hedging"],
        "margin": bt_cfg["margin"],
    }
    if "spread" in sig.parameters:
        params["spread"] = bt_cfg.get("slippage_bps", 0.0) / 10000.0
    return params


def prepare_backtest_frame(
    df: pd.DataFrame,
    *,
    warmup_bars: int = 0,
    required_cols: Iterable[str] = ("Open", "High", "Low", "Close"),
    indicator_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    if warmup_bars > 0:
        out = out.iloc[warmup_bars:]
    out = out.dropna(subset=list(required_cols))
    if indicator_cols:
        out = out.dropna(subset=list(indicator_cols))
    return out


def run_backtest(
    df_features: pd.DataFrame,
    StrategyClass: type[Strategy],
    bt_cfg: dict,
    strat_params: dict,
    *,
    warmup_bars: int = 0,
    indicator_cols: Iterable[str] | None = None,
) -> pd.Series:
    required = ("Open", "High", "Low", "Close")
    missing = [c for c in required if c not in df_features.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns for backtest: {missing}")

    df_bt = prepare_backtest_frame(
        df_features,
        warmup_bars=warmup_bars,
        required_cols=required,
        indicator_cols=indicator_cols,
    )
    if df_bt.empty:
        raise ValueError("Backtest frame is empty after warmup/NaN filtering")

    bt = Backtest(df_bt, StrategyClass, **_bt_kwargs(bt_cfg))
    stats = bt.run(**strat_params)
    return stats

