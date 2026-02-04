from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_refresh_candles_includes_positions_and_watchlists(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
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

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text('{"watchlists":{"monitor":["BBB","AAA"]}}', encoding="utf-8")

    calls: list[str] = []

    def _stub_get(self, symbol: str, *, period: str = "5y", today: date | None = None):  # noqa: ARG001
        calls.append(symbol)
        idx = pd.date_range(datetime.now() - timedelta(days=3), periods=3, freq="D")
        return pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    monkeypatch.setattr("options_helper.data.candles.CandleStore.get_daily_history", _stub_get)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "refresh-candles",
            str(portfolio_path),
            "--watchlists-path",
            str(watchlists_path),
            "--candle-cache-dir",
            str(tmp_path / "candles"),
            "--period",
            "5y",
        ],
    )
    assert res.exit_code == 0, res.output
    assert sorted(set(calls)) == ["AAA", "BBB"]
