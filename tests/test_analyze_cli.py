from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.market_types import OptionsChain


def test_analyze_help_includes_offline_flags() -> None:
    runner = CliRunner()
    res = runner.invoke(app, ["analyze", "--help"])
    assert res.exit_code == 0, res.output
    assert "--offline" in res.output
    assert "--as-of" in res.output
    assert "--offline-strict" in res.output
    assert "--snapshots-dir" in res.output
    assert "--stress-spot-pct" in res.output
    assert "--stress-vol-pp" in res.output
    assert "--stress-days" in res.output


def test_analyze_online_path_stubs_still_runs(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        """
        {
          "cash": 0,
          "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null},
          "positions": [
            {"id":"a","symbol":"AAA","option_type":"call","expiry":"2026-04-17","strike":5,"contracts":1,"cost_basis":1}
          ]
        }
        """,
        encoding="utf-8",
    )

    idx = pd.date_range(end=pd.Timestamp("2026-01-30"), periods=30, freq="B")
    history = pd.DataFrame({"Close": [10.0] * len(idx), "Volume": [1000] * len(idx)}, index=idx)

    def _stub_history(self, symbol: str, *, period: str = "2y", today=None):  # noqa: ANN001,ARG001
        return history

    monkeypatch.setattr("options_helper.cli.CandleStore.get_daily_history", _stub_history)

    class StubProvider:
        def get_options_chain(self, symbol: str, expiry: date) -> OptionsChain:  # noqa: ARG002
            calls = pd.DataFrame(
                [
                    {
                        "contractSymbol": "AAA260417C00005000",
                        "strike": 5.0,
                        "bid": 1.0,
                        "ask": 1.2,
                        "lastPrice": 1.1,
                        "impliedVolatility": 0.50,
                        "openInterest": 500,
                        "volume": 20,
                    }
                ]
            )
            puts = pd.DataFrame(
                [
                    {
                        "contractSymbol": "AAA260417P00005000",
                        "strike": 5.0,
                        "bid": 0.9,
                        "ask": 1.1,
                        "lastPrice": 1.0,
                        "impliedVolatility": 0.55,
                        "openInterest": 600,
                        "volume": 18,
                    }
                ]
            )
            return OptionsChain(symbol=symbol.upper(), expiry=expiry, calls=calls, puts=puts)

    monkeypatch.setattr("options_helper.cli.get_provider", lambda *_args, **_kwargs: StubProvider())

    runner = CliRunner()
    res = runner.invoke(app, ["analyze", str(portfolio_path), "--cache-dir", str(tmp_path / "candles")])
    assert res.exit_code == 0, res.output
