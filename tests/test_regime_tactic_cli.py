from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def _trend_ohlc(*, periods: int, start_price: float, step: float) -> pd.DataFrame:
    idx = pd.date_range("2025-01-02", periods=periods, freq="B")
    close = start_price + np.arange(periods, dtype="float64") * step
    open_ = close - 0.25
    high = close + 0.45
    low = close - 0.55
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_regime_tactic_cli_help() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["technicals", "regime-tactic", "--help"])
    assert res.exit_code == 0, res.output
    assert "--ohlc-path" in res.output
    assert "--market-ohlc-path" in res.output
    assert "--market-symbol" in res.output
    assert "--direction" in res.output


def test_regime_tactic_cli_offline_paths(tmp_path: Path) -> None:
    symbol_path = tmp_path / "symbol.csv"
    market_path = tmp_path / "market.csv"
    _trend_ohlc(periods=90, start_price=100.0, step=0.9).to_csv(symbol_path)
    _trend_ohlc(periods=90, start_price=420.0, step=0.6).to_csv(market_path)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "regime-tactic",
            "--ohlc-path",
            str(symbol_path),
            "--market-ohlc-path",
            str(market_path),
            "--symbol",
            "AAPL",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Symbol: AAPL" in res.output
    assert "Market regime: trend_up" in res.output
    assert "Symbol regime: trend_up" in res.output
    assert "Recommendation: breakout" in res.output
    assert "Support model: ema" in res.output
