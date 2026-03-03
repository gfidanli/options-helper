from __future__ import annotations

import json
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
    assert "--out" in res.output
    assert "--write-json" in res.output


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


def test_regime_tactic_cli_writes_json_artifact(tmp_path: Path) -> None:
    symbol_df = _trend_ohlc(periods=90, start_price=100.0, step=0.9)
    market_df = _trend_ohlc(periods=90, start_price=420.0, step=0.6)
    symbol_path = tmp_path / "symbol.csv"
    market_path = tmp_path / "market.csv"
    out_dir = tmp_path / "reports"
    symbol_df.to_csv(symbol_path)
    market_df.to_csv(market_path)

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
            "--market-symbol",
            "SPY",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Wrote JSON:" in res.output

    asof = symbol_df.index.max().date().isoformat()
    artifact_path = out_dir / "AAPL" / f"{asof}.json"
    assert artifact_path.exists()

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["asof_date"] == asof
    assert payload["symbol"] == "AAPL"
    assert payload["market_symbol"] == "SPY"
    assert payload["regimes"]["symbol"]["tag"] == "trend_up"
    assert payload["regimes"]["market"]["tag"] == "trend_up"
    assert payload["regimes"]["symbol"]["diagnostics"]
    assert payload["regimes"]["market"]["diagnostics"]
    assert payload["recommendation"]["direction"] == "long"
    assert payload["recommendation"]["tactic"] == "breakout"
    assert payload["recommendation"]["support_model"] == "ema"
    assert payload["recommendation"]["rationale"]
    assert payload["disclaimer"] == "Informational output only; not financial advice."


def test_regime_tactic_cli_errors_when_cache_missing_without_paths(tmp_path: Path) -> None:
    cache_dir = tmp_path / "candles"
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "regime-tactic",
            "--symbol",
            "AAPL",
            "--market-symbol",
            "SPY",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert res.exit_code == 2
    assert "No cached symbol OHLC found for AAPL" in res.output
    assert "Invalid value for --ohlc-path" in res.output


def test_regime_tactic_cli_errors_for_missing_required_columns(tmp_path: Path) -> None:
    invalid_symbol = _trend_ohlc(periods=90, start_price=100.0, step=0.9).drop(columns=["Close"])
    market_df = _trend_ohlc(periods=90, start_price=420.0, step=0.6)
    symbol_path = tmp_path / "symbol_missing_close.csv"
    market_path = tmp_path / "market.csv"
    invalid_symbol.to_csv(symbol_path)
    market_df.to_csv(market_path)

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
    assert res.exit_code == 2
    assert "missing required OHLC columns" in res.output


def test_regime_tactic_cli_errors_for_unsorted_timestamps(tmp_path: Path) -> None:
    invalid_symbol = _trend_ohlc(periods=90, start_price=100.0, step=0.9).iloc[::-1]
    market_df = _trend_ohlc(periods=90, start_price=420.0, step=0.6)
    symbol_path = tmp_path / "symbol_unsorted.csv"
    market_path = tmp_path / "market.csv"
    invalid_symbol.to_csv(symbol_path)
    market_df.to_csv(market_path)

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
    assert res.exit_code == 2
    assert "sorted ascending" in res.output


def test_regime_tactic_cli_errors_for_duplicate_timestamps(tmp_path: Path) -> None:
    base = _trend_ohlc(periods=90, start_price=100.0, step=0.9)
    invalid_symbol = pd.concat([base.iloc[:10], base.iloc[[9]], base.iloc[10:]])
    market_df = _trend_ohlc(periods=90, start_price=420.0, step=0.6)
    symbol_path = tmp_path / "symbol_duplicates.csv"
    market_path = tmp_path / "market.csv"
    invalid_symbol.to_csv(symbol_path)
    market_df.to_csv(market_path)

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
    assert res.exit_code == 2
    assert "duplicate timestamps" in res.output
