from __future__ import annotations

from pathlib import Path

from rich.console import Console

from options_helper.schemas.briefing import BriefingArtifact
from options_helper.ui.dashboard import render_dashboard


def _briefing_payload() -> dict:
    return {
        "schema_version": 1,
        "generated_at": "2026-01-02T00:00:00+00:00",
        "as_of": "2026-01-02",
        "disclaimer": "Not financial advice.",
        "report_date": "2026-01-02",
        "portfolio_path": "data/portfolio.json",
        "symbols": ["AAA", "BBB"],
        "top": 3,
        "technicals": {
            "source": "technicals_backtesting",
            "config_path": None,
        },
        "portfolio": {
            "exposure": None,
            "stress": [],
        },
        "portfolio_rows": [
            {
                "id": "pos-1",
                "symbol": "AAA",
                "option_type": "call",
                "expiry": "2026-02-20",
                "strike": 110.0,
                "contracts": 1,
                "cost_basis": 1.0,
                "mark": 1.2,
                "pnl": 20.0,
                "pnl_pct": 0.2,
                "spr_pct": 0.1,
                "as_of": "2026-01-02",
            }
        ],
        "symbol_sources": [
            {"symbol": "AAA", "sources": ["portfolio"]},
            {"symbol": "BBB", "sources": ["watchlist:monitor"]},
        ],
        "watchlists": [
            {"name": "monitor", "symbols": ["BBB"]},
        ],
        "sections": [
            {
                "symbol": "AAA",
                "as_of": "2026-01-02",
                "errors": [],
                "warnings": [],
                "derived_updated": False,
                "next_earnings_date": None,
            },
            {
                "symbol": "BBB",
                "as_of": "2026-01-02",
                "errors": [],
                "warnings": ["compare unavailable"],
                "derived_updated": False,
                "next_earnings_date": None,
            },
        ],
    }


def test_render_dashboard_outputs_sections(tmp_path: Path) -> None:
    artifact = BriefingArtifact.model_validate(_briefing_payload())
    console = Console(record=True, width=140)
    render_dashboard(
        artifact=artifact,
        console=console,
        reports_dir=tmp_path / "reports",
        scanner_run_dir=tmp_path / "scanner",
    )
    output = console.export_text()
    assert "Daily Briefing Dashboard" in output
    assert "Portfolio Positions" in output
    assert "Symbol Summary" in output
    assert "Watchlist: monitor" in output
    assert "AAA" in output
