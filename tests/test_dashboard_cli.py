from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app


def _write_briefing_json(path: Path, *, report_date: str) -> None:
    payload = {
        "schema_version": 1,
        "generated_at": "2026-02-03T00:00:00+00:00",
        "as_of": report_date,
        "disclaimer": "Not financial advice. For informational/educational use only.",
        "report_date": report_date,
        "portfolio_path": "data/portfolio.json",
        "symbols": ["AAA"],
        "top": 3,
        "technicals": {
            "source": "technicals_backtesting",
            "config_path": None,
        },
        "portfolio": {
            "exposure": None,
            "stress": [],
        },
        "sections": [
            {
                "symbol": "AAA",
                "as_of": report_date,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_dashboard_cli_renders(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True)
    _write_briefing_json(daily_dir / "2026-01-02.json", report_date="2026-01-02")

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "dashboard",
            "--date",
            "2026-01-02",
            "--reports-dir",
            str(reports_dir),
            "--scanner-run-dir",
            str(tmp_path / "scanner"),
        ],
    )
    assert res.exit_code == 0, res.output
    assert "Daily Briefing Dashboard" in res.output
