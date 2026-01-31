from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_flow_cli_can_use_watchlists_without_positions(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        '{"cash": 0, "risk_profile": {"tolerance": "high", "max_portfolio_risk_pct": null, "max_single_position_risk_pct": null}}',
        encoding="utf-8",
    )

    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text('{"watchlists":{"monitor":["AAA"]}}', encoding="utf-8")

    cache_dir = tmp_path / "snapshots"
    # Two days of snapshots for AAA so flow can diff them.
    day1 = cache_dir / "AAA" / "2026-01-01"
    day2 = cache_dir / "AAA" / "2026-01-02"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    df_prev = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA240119C00010000",
                "optionType": "call",
                "expiry": "2024-01-19",
                "strike": 10.0,
                "lastPrice": 2.0,
                "volume": 100,
                "openInterest": 100,
            }
        ]
    )
    df_today = df_prev.copy()
    df_today.loc[0, "openInterest"] = 150

    df_prev.to_csv(day1 / "2024-01-19.csv", index=False)
    df_today.to_csv(day2 / "2024-01-19.csv", index=False)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "flow",
            str(portfolio_path),
            "--cache-dir",
            str(cache_dir),
            "--watchlists-path",
            str(watchlists_path),
            "--watchlist",
            "monitor",
            "--top",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "AAA flow 2026-01-01" in res.output

