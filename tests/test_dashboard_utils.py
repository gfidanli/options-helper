from __future__ import annotations

import json
from pathlib import Path

import pytest

from options_helper.ui.dashboard import (
    load_briefing_artifact,
    load_scanner_shortlist,
    resolve_briefing_json,
    resolve_briefing_paths,
)


def _write_briefing(path: Path, *, report_date: str) -> None:
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


def test_resolve_briefing_json_latest(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True)
    _write_briefing(daily_dir / "2026-01-01.json", report_date="2026-01-01")
    _write_briefing(daily_dir / "2026-01-02.json", report_date="2026-01-02")
    (daily_dir / "latest.json").write_text("{}", encoding="utf-8")

    latest = resolve_briefing_json(reports_dir, "latest")
    assert latest.name == "2026-01-02.json"


def test_resolve_briefing_paths_specific(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    daily_dir = reports_dir / "daily"
    daily_dir.mkdir(parents=True)
    _write_briefing(daily_dir / "2026-01-02.json", report_date="2026-01-02")

    paths = resolve_briefing_paths(reports_dir, "2026-01-02")
    assert paths.date == "2026-01-02"
    assert paths.json_path.name == "2026-01-02.json"
    assert paths.md_path.name == "2026-01-02.md"


def test_resolve_briefing_json_invalid(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    (reports_dir / "daily").mkdir(parents=True)
    with pytest.raises(ValueError):
        resolve_briefing_json(reports_dir, "not-a-date")


def test_load_briefing_artifact(tmp_path: Path) -> None:
    report_path = tmp_path / "2026-01-02.json"
    _write_briefing(report_path, report_date="2026-01-02")
    artifact = load_briefing_artifact(report_path)
    assert artifact.report_date == "2026-01-02"
    assert artifact.symbols == ["AAA"]


def test_load_scanner_shortlist(tmp_path: Path) -> None:
    run_dir = tmp_path / "scanner"
    run_id = "2026-01-02_120000"
    shortlist_path = run_dir / run_id / "shortlist.json"
    shortlist_path.parent.mkdir(parents=True)
    payload = {
        "schema_version": 1,
        "generated_at": "2026-02-03T00:00:00+00:00",
        "as_of": "2026-01-02",
        "run_id": run_id,
        "universe": "us-all",
        "tail_low_pct": 2.5,
        "tail_high_pct": 97.5,
        "all_watchlist_name": "Scanner - All",
        "shortlist_watchlist_name": "Scanner - Shortlist",
        "rows": [
            {
                "symbol": "AAA",
                "score": 12.3,
                "coverage": 0.8,
                "top_reasons": "test",
            }
        ],
    }
    shortlist_path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = load_scanner_shortlist(run_dir, report_date="2026-01-02")
    assert artifact is not None
    assert artifact.run_id == run_id
