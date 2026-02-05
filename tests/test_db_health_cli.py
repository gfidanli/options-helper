from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import duckdb
from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.observability_meta import DuckDBRunLogger
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_health_rows(duckdb_path: Path) -> None:
    warehouse = DuckDBWarehouse(duckdb_path)
    now = datetime.now(timezone.utc)

    ingest = DuckDBRunLogger(warehouse)
    ingest.start_run(job_name="ingest_candles", started_at=now - timedelta(days=2))
    ingest.upsert_watermark(
        asset_key="candles_daily",
        scope_key="AAPL",
        watermark_ts=now - timedelta(days=9),
    )
    ingest.persist_check(
        asset_key="candles_daily",
        check_name="candles_no_negative_prices",
        severity="error",
        status="pass",
    )
    ingest.finalize_success(ended_at=now - timedelta(days=2, minutes=-1))

    failed = DuckDBRunLogger(warehouse)
    failed_run_id = failed.start_run(job_name="compute_flow", started_at=now - timedelta(hours=4))
    failed.persist_check(
        asset_key="options_flow",
        partition_key="AAPL|2026-02-05",
        check_name="flow_no_null_primary_keys",
        severity="error",
        status="fail",
        message="missing row_key",
        run_id=failed_run_id,
        checked_at=now - timedelta(hours=4),
    )
    failed.finalize_failure(RuntimeError("flow failed"), ended_at=now - timedelta(hours=4, minutes=-1))

    recovered = DuckDBRunLogger(warehouse)
    recovered.start_run(job_name="compute_flow", started_at=now - timedelta(hours=1))
    recovered.finalize_success(ended_at=now - timedelta(hours=1, minutes=-1))


def test_db_health_human_output_includes_sections(tmp_path: Path) -> None:
    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    _seed_health_rows(duckdb_path)

    res = runner.invoke(
        app,
        [
            "db",
            "health",
            "--duckdb-path",
            str(duckdb_path),
            "--days",
            "7",
            "--limit",
            "20",
            "--stale-days",
            "5",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "Latest run per job" in res.output
    assert "Recent run failures" in res.output
    assert "Watermarks and freshness" in res.output
    assert "Recent failed checks" in res.output
    assert "ingest_candles" in res.output
    assert "compute_flow" in res.output
    assert "candles_daily" in res.output
    assert "flow_no_null_primary_keys" in res.output


def test_db_health_json_output_includes_filters_and_sections(tmp_path: Path) -> None:
    runner = CliRunner()
    duckdb_path = tmp_path / "options.duckdb"
    _seed_health_rows(duckdb_path)

    res = runner.invoke(
        app,
        [
            "db",
            "health",
            "--duckdb-path",
            str(duckdb_path),
            "--days",
            "7",
            "--limit",
            "1",
            "--stale-days",
            "5",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)

    assert payload["database_exists"] is True
    assert payload["meta_tables_present"] is True
    assert payload["filters"] == {"days": 7, "limit": 1, "stale_days": 5}

    assert len(payload["latest_runs"]) == 1
    assert payload["latest_runs"][0]["job_name"] == "compute_flow"

    assert len(payload["recent_failures"]) == 1
    assert payload["recent_failures"][0]["job_name"] == "compute_flow"

    assert len(payload["watermarks"]) == 1
    assert payload["watermarks"][0]["asset_key"] == "candles_daily"

    assert len(payload["recent_failed_checks"]) == 1
    assert payload["recent_failed_checks"][0]["check_name"] == "flow_no_null_primary_keys"


def test_db_health_gracefully_handles_missing_meta_tables(tmp_path: Path) -> None:
    runner = CliRunner()
    duckdb_path = tmp_path / "no_meta.duckdb"
    duckdb.connect(str(duckdb_path)).close()

    human = runner.invoke(
        app,
        [
            "db",
            "health",
            "--duckdb-path",
            str(duckdb_path),
        ],
    )
    assert human.exit_code == 0, human.output
    assert "Observability tables (meta.*) are missing." in human.output
    assert "No run history available." in human.output

    machine = runner.invoke(
        app,
        [
            "db",
            "health",
            "--duckdb-path",
            str(duckdb_path),
            "--json",
        ],
    )
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["database_exists"] is True
    assert payload["meta_tables_present"] is False
    assert payload["latest_runs"] == []
    assert payload["recent_failures"] == []
    assert payload["watermarks"] == []
    assert payload["recent_failed_checks"] == []
