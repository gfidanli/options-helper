from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import market_analysis as market_analysis_command
from options_helper.schemas.tail_risk import TailRiskArtifact


class _StubCandleStore:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def load(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        return self._history.copy()

    def get_daily_history(self, symbol: str, *, period: str = "max") -> pd.DataFrame:  # noqa: ARG002
        return self._history.copy()


class _StubDerivedStore:
    def __init__(self, derived: pd.DataFrame) -> None:
        self._derived = derived

    def load(self, symbol: str) -> pd.DataFrame:  # noqa: ARG002
        return self._derived.copy()


def _build_history() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=450, freq="D")
    prices = pd.Series(range(100, 550), dtype="float64").pow(1.005).values
    return pd.DataFrame({"Close": prices}, index=idx)


def _build_derived() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2026-01-31",
                "atm_iv_near": 0.22,
                "rv_20d": 0.18,
                "rv_60d": 0.20,
                "iv_rv_20d": 1.22,
                "atm_iv_near_percentile": 67.0,
                "iv_term_slope": 0.013,
            }
        ]
    )


def test_market_analysis_cli_console_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore(_build_history()),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_derived_store",
        lambda *args, **kwargs: _StubDerivedStore(_build_derived()),
    )
    monkeypatch.setattr(market_analysis_command.cli_deps, "build_provider", lambda: object())

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "tail risk as-of" in res.output
    assert "Horizon End Percentiles" in res.output
    assert "IV context:" in res.output


def test_market_analysis_cli_json_output_validates_schema(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_candle_store",
        lambda *args, **kwargs: _StubCandleStore(_build_history()),
    )
    monkeypatch.setattr(
        market_analysis_command.cli_deps,
        "build_derived_store",
        lambda *args, **kwargs: _StubDerivedStore(_build_derived()),
    )
    monkeypatch.setattr(market_analysis_command.cli_deps, "build_provider", lambda: object())

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
            "--format",
            "json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = res.output.strip()
    artifact = TailRiskArtifact.model_validate_json(payload)
    assert artifact.symbol == "SPY"
    assert artifact.config.horizon_days == 20

    out_res = runner.invoke(
        app,
        [
            "market-analysis",
            "tail-risk",
            "--symbol",
            "SPY",
            "--lookback-days",
            "252",
            "--horizon-days",
            "20",
            "--num-simulations",
            "2000",
            "--seed",
            "42",
            "--out",
            str(tmp_path),
        ],
    )
    assert out_res.exit_code == 0, out_res.output
    assert (tmp_path / "tail_risk" / "SPY").exists()
