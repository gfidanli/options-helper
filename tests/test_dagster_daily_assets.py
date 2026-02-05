from __future__ import annotations

import importlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from options_helper.data.ingestion.candles import CandleIngestResult
from options_helper.data.ingestion.options_bars import (
    BarsBackfillSummary,
    ContractDiscoveryOutput,
    PreparedContracts,
    UnderlyingDiscoverySummary,
)
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.pipelines.visibility_jobs import (
    BriefingJobResult,
    DerivedUpdateJobResult,
    FlowReportJobResult,
    IngestCandlesJobResult,
    IngestOptionsBarsJobResult,
    SnapshotOptionsJobResult,
)


def _write_watchlists(path: Path) -> None:
    path.write_text('{"watchlists":{"positions":["AAA"]}}', encoding="utf-8")


def _write_portfolio(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cash": 0.0,
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


def test_dagster_definitions_include_daily_assets_and_checks() -> None:
    pytest.importorskip("dagster")

    defs_module = importlib.import_module("apps.dagster.defs")
    assets_module = importlib.import_module("apps.dagster.defs.assets")
    checks_module = importlib.import_module("apps.dagster.defs.checks")

    from dagster import Definitions

    assert isinstance(defs_module.defs, Definitions)
    assert [asset.key.path[-1] for asset in assets_module.ASSET_DEFINITIONS] == list(assets_module.ASSET_ORDER)
    assert len(checks_module.ASSET_CHECK_DEFINITIONS) >= 6


def test_dagster_partitioned_materialization_order_and_ledger(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("dagster")

    from dagster import materialize

    partition_day = date(2026, 1, 2)
    partition_key = partition_day.isoformat()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    duckdb_path = tmp_path / "options.duckdb"
    watchlists_path = tmp_path / "watchlists.json"
    portfolio_path = tmp_path / "portfolio.json"
    _write_watchlists(watchlists_path)
    _write_portfolio(portfolio_path)

    monkeypatch.setenv("OPTIONS_HELPER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPTIONS_HELPER_DUCKDB_PATH", str(duckdb_path))
    monkeypatch.setenv("OPTIONS_HELPER_WATCHLISTS_PATH", str(watchlists_path))
    monkeypatch.setenv("OPTIONS_HELPER_PORTFOLIO_PATH", str(portfolio_path))

    assets_module = importlib.import_module("apps.dagster.defs.assets")
    resources_module = importlib.import_module("apps.dagster.defs.resources")

    call_order: list[str] = []

    def _stub_ingest_candles_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("candles_daily")
        return IngestCandlesJobResult(
            warnings=[],
            symbols=["AAA"],
            results=[CandleIngestResult(symbol="AAA", status="ok", last_date=partition_day, error=None)],
            no_symbols=False,
        )

    def _stub_ingest_options_bars_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("options_bars")
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
            no_symbols=False,
            no_contracts=False,
            no_eligible_contracts=False,
        )

    def _stub_snapshot_options_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("options_snapshot_file")
        return SnapshotOptionsJobResult(
            messages=[f"AAA {partition_key}: saved 10 contracts"],
            dates_used=[partition_day],
            symbols=["AAA"],
            no_symbols=False,
        )

    def _stub_flow_report_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("options_flow")
        return FlowReportJobResult(
            renderables=[
                "[bold]AAA[/bold] flow 2026-01-01 -> 2026-01-02 | calls dOI$=1 | puts dOI$=0",
            ],
            no_symbols=False,
        )

    derived_path = data_dir / "derived" / "AAA.csv"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.write_text("date,spot\n2026-01-02,100.0\n", encoding="utf-8")

    def _stub_derived_update_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("derived_metrics")
        return DerivedUpdateJobResult(symbol="AAA", as_of_date=partition_day, output_path=derived_path)

    briefing_md = data_dir / "reports" / "daily" / f"{partition_key}.md"
    briefing_md.parent.mkdir(parents=True, exist_ok=True)
    briefing_md.write_text("# briefing\n", encoding="utf-8")
    briefing_json = briefing_md.with_suffix(".json")
    briefing_json.write_text('{"report_date":"2026-01-02"}', encoding="utf-8")

    def _stub_briefing_job(**_kwargs):  # type: ignore[no-untyped-def]
        call_order.append("briefing_markdown")
        return BriefingJobResult(
            report_date=partition_key,
            markdown="# briefing",
            markdown_path=briefing_md,
            json_path=briefing_json,
            renderables=[],
        )

    monkeypatch.setattr(assets_module, "run_ingest_candles_job", _stub_ingest_candles_job)
    monkeypatch.setattr(assets_module, "run_ingest_options_bars_job", _stub_ingest_options_bars_job)
    monkeypatch.setattr(assets_module, "run_snapshot_options_job", _stub_snapshot_options_job)
    monkeypatch.setattr(assets_module, "run_flow_report_job", _stub_flow_report_job)
    monkeypatch.setattr(assets_module, "run_derived_update_job", _stub_derived_update_job)
    monkeypatch.setattr(assets_module, "run_briefing_job", _stub_briefing_job)

    result = materialize(
        assets_module.ASSET_DEFINITIONS,
        resources=resources_module.build_resources(),
        partition_key=partition_key,
    )
    assert result.success
    assert call_order == [
        "candles_daily",
        "options_bars",
        "options_snapshot_file",
        "options_flow",
        "derived_metrics",
        "briefing_markdown",
    ]

    warehouse = DuckDBWarehouse(duckdb_path)
    runs = warehouse.fetch_df(
        """
        SELECT job_name, triggered_by, parent_run_id, status
        FROM meta.ingestion_runs
        ORDER BY started_at ASC
        """
    )
    assert set(runs["job_name"]) == {
        "dagster_candles_daily",
        "dagster_options_bars",
        "dagster_options_snapshot_file",
        "dagster_options_flow",
        "dagster_derived_metrics",
        "dagster_briefing_markdown",
    }
    assert set(runs["triggered_by"]) == {"dagster"}
    assert set(runs["status"]) == {"success"}
    assert all(str(value).strip() for value in runs["parent_run_id"].tolist())

    run_assets = warehouse.fetch_df(
        """
        SELECT asset_key, partition_key, status
        FROM meta.ingestion_run_assets
        ORDER BY asset_key ASC
        """
    )
    assert set(run_assets["asset_key"]) == {
        "candles_daily",
        "options_bars",
        "options_snapshot_file",
        "options_flow",
        "derived_metrics",
        "briefing_markdown",
    }
    assert set(run_assets["partition_key"]) == {partition_key}
