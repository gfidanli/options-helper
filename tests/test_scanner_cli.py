from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from options_helper.analysis.confluence import ConfluenceInputs, score_confluence
from options_helper.analysis.scanner_rank import ScannerRankResult
from options_helper.cli import app
from options_helper.data.scanner import ScannerLiquidityRow, ScannerScanRow


def test_scanner_run_updates_watchlists_and_writes_run_files(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text("{}", encoding="utf-8")

    def _stub_universe(*args, **kwargs):  # noqa: ANN001, ARG001
        return ["AAA", "BBB"]

    def _stub_scan(*args, **kwargs):  # noqa: ANN001, ARG001
        rows = [
            ScannerScanRow(
                symbol="AAA",
                asof="2026-01-31",
                extension_atr=1.5,
                percentile=98.0,
                window_years=3,
                window_bars=756,
                tail=True,
                status="ok",
                error=None,
            ),
            ScannerScanRow(
                symbol="BBB",
                asof="2026-01-31",
                extension_atr=0.2,
                percentile=50.0,
                window_years=3,
                window_bars=756,
                tail=False,
                status="ok",
                error=None,
            ),
        ]
        return rows, ["AAA"]

    def _stub_liquidity(*args, **kwargs):  # noqa: ANN001, ARG001
        rows = [
            ScannerLiquidityRow(
                symbol="AAA",
                snapshot_date="2026-01-31",
                eligible_contracts=2,
                eligible_expiries="2026-04-17",
                is_liquid=True,
                status="ok",
                error=None,
            )
        ]
        return rows, ["AAA"]

    def _stub_confluence(symbols, **kwargs):  # noqa: ANN001, ARG001
        return {"AAA": score_confluence(ConfluenceInputs(weekly_trend="up"))}

    def _stub_rank(symbols, **kwargs):  # noqa: ANN001, ARG001
        return {
            "AAA": ScannerRankResult(
                score=85.0,
                coverage=0.5,
                components=[],
                warnings=[],
                top_reasons=["Extension tail low (pct=98.0)."],
            )
        }

    monkeypatch.setattr("options_helper.commands.scanner.load_universe_symbols", _stub_universe)
    monkeypatch.setattr("options_helper.commands.scanner.scan_symbols", _stub_scan)
    monkeypatch.setattr("options_helper.commands.scanner.evaluate_liquidity_for_symbols", _stub_liquidity)
    monkeypatch.setattr("options_helper.commands.scanner.score_shortlist_confluence", _stub_confluence)
    monkeypatch.setattr("options_helper.commands.scanner.rank_shortlist_candidates", _stub_rank)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "scanner",
            "run",
            "--watchlists-path",
            str(watchlists_path),
            "--run-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "test-run",
            "--scanned-path",
            str(tmp_path / "scanned_symbols.txt"),
            "--no-backfill",
            "--no-snapshot-options",
            "--no-run-reports",
        ],
    )
    assert res.exit_code == 0, res.output

    wl_text = watchlists_path.read_text(encoding="utf-8")
    assert "Scanner - All" in wl_text
    assert "Scanner - Shortlist" in wl_text
    assert "AAA" in wl_text

    run_root = tmp_path / "runs" / "test-run"
    assert (run_root / "scan.csv").exists()
    assert (run_root / "liquidity.csv").exists()
    assert (run_root / "shortlist.csv").exists()
    assert (run_root / "shortlist.json").exists()
    assert (run_root / "shortlist.md").exists()
    shortlist_txt = (run_root / "shortlist.md").read_text(encoding="utf-8")
    assert "scanner" in shortlist_txt

    payload = json.loads((run_root / "shortlist.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["as_of"] == "2026-01-31"
    assert payload["run_id"] == "test-run"
    assert payload["rows"][0]["symbol"] == "AAA"


def test_scanner_run_preserves_scanned_file_when_no_skip(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text("{}", encoding="utf-8")
    scanned_path = tmp_path / "scanned_symbols.txt"
    scanned_path.write_text("OLD\n", encoding="utf-8")

    def _stub_universe(*args, **kwargs):  # noqa: ANN001, ARG001
        return ["AAA"]

    def _stub_scan(*args, **kwargs):  # noqa: ANN001, ARG001
        rows = [
            ScannerScanRow(
                symbol="AAA",
                asof="2026-01-31",
                extension_atr=1.5,
                percentile=98.0,
                window_years=3,
                window_bars=756,
                tail=True,
                status="ok",
                error=None,
            ),
        ]
        row_callback = kwargs.get("row_callback")
        if row_callback:
            for row in rows:
                row_callback(row)
        return rows, ["AAA"]

    def _stub_liquidity(*args, **kwargs):  # noqa: ANN001, ARG001
        rows = [
            ScannerLiquidityRow(
                symbol="AAA",
                snapshot_date="2026-01-31",
                eligible_contracts=2,
                eligible_expiries="2026-04-17",
                is_liquid=True,
                status="ok",
                error=None,
            )
        ]
        return rows, ["AAA"]

    def _stub_confluence(symbols, **kwargs):  # noqa: ANN001, ARG001
        return {"AAA": score_confluence(ConfluenceInputs(weekly_trend="up"))}

    def _stub_rank(symbols, **kwargs):  # noqa: ANN001, ARG001
        return {
            "AAA": ScannerRankResult(
                score=85.0,
                coverage=0.5,
                components=[],
                warnings=[],
                top_reasons=["Extension tail low (pct=98.0)."],
            )
        }

    monkeypatch.setattr("options_helper.commands.scanner.load_universe_symbols", _stub_universe)
    monkeypatch.setattr("options_helper.commands.scanner.scan_symbols", _stub_scan)
    monkeypatch.setattr("options_helper.commands.scanner.evaluate_liquidity_for_symbols", _stub_liquidity)
    monkeypatch.setattr("options_helper.commands.scanner.score_shortlist_confluence", _stub_confluence)
    monkeypatch.setattr("options_helper.commands.scanner.rank_shortlist_candidates", _stub_rank)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "scanner",
            "run",
            "--watchlists-path",
            str(watchlists_path),
            "--run-dir",
            str(tmp_path / "runs"),
            "--run-id",
            "test-run",
            "--scanned-path",
            str(scanned_path),
            "--no-skip-scanned",
            "--no-backfill",
            "--no-snapshot-options",
            "--no-run-reports",
        ],
    )
    assert res.exit_code == 0, res.output
    scanned_text = scanned_path.read_text(encoding="utf-8")
    assert "OLD" in scanned_text
    assert "AAA" in scanned_text
