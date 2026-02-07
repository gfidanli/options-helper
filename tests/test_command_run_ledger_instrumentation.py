from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.ingestion.candles import CandleIngestResult
from options_helper.data.ingestion.options_bars import (
    BarsBackfillSummary,
    ContractDiscoveryOutput,
    PreparedContracts,
    UnderlyingDiscoverySummary,
)
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.pipelines.visibility_jobs import (
    BriefingJobResult,
    DashboardJobResult,
    DerivedUpdateJobResult,
    FlowReportJobResult,
    IngestCandlesJobResult,
    IngestOptionsBarsJobResult,
    SnapshotOptionsJobResult,
)


def _write_min_portfolio(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cash": 0,
                "risk_profile": {
                    "tolerance": "high",
                    "max_portfolio_risk_pct": None,
                    "max_single_position_risk_pct": None,
                },
                "positions": [],
            }
        ),
        encoding="utf-8",
    )


def test_producer_commands_log_runs_assets_and_watermarks_duckdb(tmp_path: Path, monkeypatch) -> None:
    duckdb_path = tmp_path / "options.duckdb"
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text('{"watchlists":{"positions":["AAA"]}}', encoding="utf-8")
    portfolio_path = tmp_path / "portfolio.json"
    _write_min_portfolio(portfolio_path)

    report_day = date(2026, 1, 2)

    def _stub_ingest_candles_job(**_kwargs):  # type: ignore[no-untyped-def]
        return IngestCandlesJobResult(
            warnings=[],
            symbols=["AAA", "BBB"],
            results=[
                CandleIngestResult(symbol="AAA", status="ok", last_date=report_day, error=None),
                CandleIngestResult(symbol="BBB", status="empty", last_date=None, error=None),
            ],
            no_symbols=False,
        )

    monkeypatch.setattr("options_helper.commands.ingest.run_ingest_candles_job", _stub_ingest_candles_job)

    contracts_df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260220C00100000",
                "underlying": "AAA",
                "expiry": "2026-02-20",
                "optionType": "call",
                "strike": 100.0,
                "multiplier": 100,
            }
        ]
    )

    def _stub_ingest_options_bars_job(**_kwargs):  # type: ignore[no-untyped-def]
        return IngestOptionsBarsJobResult(
            warnings=[],
            underlyings=["AAA"],
            limited_underlyings=False,
            discovery=ContractDiscoveryOutput(
                contracts=contracts_df,
                raw_by_symbol={},
                summaries=[
                    UnderlyingDiscoverySummary(
                        underlying="AAA",
                        contracts=1,
                        years_scanned=1,
                        empty_years=0,
                        status="ok",
                        error=None,
                    )
                ],
            ),
            prepared=PreparedContracts(
                contracts=contracts_df.assign(expiry_date=date(2026, 2, 20)),
                expiries=[date(2026, 2, 20)],
            ),
            summary=BarsBackfillSummary(
                total_contracts=1,
                total_expiries=1,
                planned_contracts=1,
                skipped_contracts=0,
                ok_contracts=1,
                error_contracts=0,
                bars_rows=5,
                requests_attempted=1,
            ),
            dry_run=False,
            contracts_only=False,
            no_symbols=False,
            no_contracts=False,
            no_eligible_contracts=False,
        )

    monkeypatch.setattr("options_helper.commands.ingest.run_ingest_options_bars_job", _stub_ingest_options_bars_job)

    def _stub_snapshot_options_job(**_kwargs):  # type: ignore[no-untyped-def]
        return SnapshotOptionsJobResult(
            messages=[
                "AAA 2026-01-02: saved 20 contracts",
                "[yellow]Warning:[/yellow] BBB: candle date - != required 2026-01-02; skipping snapshot to avoid mis-dated overwrite.",
                "Snapshot complete. Data date(s): 2026-01-02.",
            ],
            dates_used=[report_day],
            symbols=["AAA", "BBB"],
            no_symbols=False,
        )

    monkeypatch.setattr("options_helper.commands.workflows.run_snapshot_options_job", _stub_snapshot_options_job)

    def _stub_flow_report_job(**_kwargs):  # type: ignore[no-untyped-def]
        return FlowReportJobResult(
            renderables=[
                "[bold]AAA[/bold] flow 2026-01-01 → 2026-01-02 | calls ΔOI$=1 | puts ΔOI$=0",
                "[yellow]No flow data for BBB:[/yellow] need at least 2 snapshots.",
            ],
            no_symbols=False,
        )

    monkeypatch.setattr("options_helper.commands.reports.run_flow_report_job", _stub_flow_report_job)

    derived_out = tmp_path / "derived" / "AAA.csv"
    derived_out.parent.mkdir(parents=True, exist_ok=True)
    derived_out.write_text("date,spot\n2026-01-02,100\n", encoding="utf-8")

    def _stub_derived_update_job(**_kwargs):  # type: ignore[no-untyped-def]
        return DerivedUpdateJobResult(symbol="AAA", as_of_date=report_day, output_path=derived_out)

    monkeypatch.setattr("options_helper.commands.derived.run_derived_update_job", _stub_derived_update_job)

    briefing_md = tmp_path / "reports" / "daily" / "2026-01-02.md"
    briefing_md.parent.mkdir(parents=True, exist_ok=True)
    briefing_md.write_text("# brief", encoding="utf-8")
    briefing_json = briefing_md.with_suffix(".json")
    briefing_json.write_text('{"as_of":"2026-01-02"}', encoding="utf-8")

    def _stub_briefing_job(**_kwargs):  # type: ignore[no-untyped-def]
        return BriefingJobResult(
            report_date="2026-01-02",
            markdown="# brief",
            markdown_path=briefing_md,
            json_path=briefing_json,
            renderables=[],
        )

    monkeypatch.setattr("options_helper.commands.reports.run_briefing_job", _stub_briefing_job)

    dashboard_json = tmp_path / "reports" / "daily" / "2026-01-02.json"
    dashboard_json.parent.mkdir(parents=True, exist_ok=True)
    dashboard_json.write_text('{"as_of":"2026-01-02"}', encoding="utf-8")

    def _stub_dashboard_job(**_kwargs):  # type: ignore[no-untyped-def]
        return DashboardJobResult(json_path=dashboard_json, artifact={"as_of": "2026-01-02"})

    monkeypatch.setattr("options_helper.commands.reports.run_dashboard_job", _stub_dashboard_job)
    monkeypatch.setattr("options_helper.commands.reports.render_dashboard_report", lambda **_kwargs: None)

    runner = CliRunner()

    command_matrix = [
        [
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "candles",
            "--watchlists-path",
            str(watchlists_path),
        ],
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "AAA",
        ],
        [
            "--duckdb-path",
            str(duckdb_path),
            "snapshot-options",
            str(portfolio_path),
        ],
        [
            "--duckdb-path",
            str(duckdb_path),
            "flow",
            str(portfolio_path),
        ],
        [
            "--duckdb-path",
            str(duckdb_path),
            "derived",
            "update",
            "--symbol",
            "AAA",
        ],
        [
            "--duckdb-path",
            str(duckdb_path),
            "briefing",
            str(portfolio_path),
        ],
        [
            "--duckdb-path",
            str(duckdb_path),
            "dashboard",
            "--reports-dir",
            str(tmp_path / "reports"),
        ],
    ]

    for cmd in command_matrix:
        res = runner.invoke(app, cmd)
        assert res.exit_code == 0, res.output

    wh = DuckDBWarehouse(duckdb_path)
    run_rows = wh.fetch_df(
        """
        SELECT job_name, status
        FROM meta.ingestion_runs
        ORDER BY job_name
        """
    )
    assert set(run_rows["job_name"]) == {
        "ingest_candles",
        "ingest_options_bars",
        "snapshot_options",
        "compute_flow",
        "compute_derived",
        "build_briefing",
        "build_dashboard",
    }
    assert set(run_rows["status"]) == {"success"}

    asset_keys = set(
        wh.fetch_df(
            """
            SELECT DISTINCT asset_key
            FROM meta.ingestion_run_assets
            """
        )["asset_key"]
    )
    assert {
        "candles_daily",
        "option_contracts",
        "option_bars",
        "options_snapshots",
        "options_flow",
        "derived_daily",
        "briefing_markdown",
        "briefing_json",
        "dashboard_view",
    }.issubset(asset_keys)

    watermark_rows = wh.fetch_df(
        """
        SELECT asset_key, scope_key
        FROM meta.asset_watermarks
        ORDER BY asset_key, scope_key
        """
    )
    watermarks = {(row["asset_key"], row["scope_key"]) for row in watermark_rows.to_dict(orient="records")}
    assert ("candles_daily", "AAA") in watermarks
    assert ("option_bars", "ALL") in watermarks
    assert ("options_snapshots", "AAA") in watermarks
    assert ("options_flow", "AAA") in watermarks
    assert ("derived_daily", "AAA") in watermarks
    assert ("briefing_markdown", "ALL") in watermarks
    assert ("dashboard_view", "ALL") in watermarks


def test_ingest_options_bars_failures_mark_run_failed(tmp_path: Path, monkeypatch) -> None:
    duckdb_path = tmp_path / "options.duckdb"

    def _boom(**_kwargs):  # type: ignore[no-untyped-def]
        raise OptionBarsStoreError("bars store failure")

    monkeypatch.setattr("options_helper.commands.ingest.run_ingest_options_bars_job", _boom)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "--duckdb-path",
            str(duckdb_path),
            "ingest",
            "options-bars",
            "--symbol",
            "AAA",
        ],
    )
    assert res.exit_code == 1, res.output
    assert "bars store failure" in res.output

    wh = DuckDBWarehouse(duckdb_path)
    row = wh.fetch_df(
        """
        SELECT status, error_type, error_message, error_stack_hash
        FROM meta.ingestion_runs
        WHERE job_name = 'ingest_options_bars'
        ORDER BY started_at DESC
        LIMIT 1
        """
    ).iloc[0]
    assert str(row["status"]) == "failed"
    assert str(row["error_type"]) == "OptionBarsStoreError"
    assert "bars store failure" in str(row["error_message"])
    assert str(row["error_stack_hash"]).strip() != ""


def test_filesystem_noop_logger_warns_once_and_does_not_crash(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "derived" / "AAA.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("date,spot\n2026-01-02,100\n", encoding="utf-8")

    def _stub_derived_update_job(**_kwargs):  # type: ignore[no-untyped-def]
        return DerivedUpdateJobResult(symbol="AAA", as_of_date=date(2026, 1, 2), output_path=out_path)

    monkeypatch.setattr("options_helper.commands.derived.run_derived_update_job", _stub_derived_update_job)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "derived",
            "update",
            "--symbol",
            "AAA",
        ],
    )
    assert res.exit_code == 0, res.output
    assert res.output.count("Run ledger disabled for filesystem storage backend") == 1
