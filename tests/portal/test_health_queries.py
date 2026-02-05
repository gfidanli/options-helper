from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from apps.streamlit.components.health_page import load_health_snapshot
from options_helper.data.observability_meta import DuckDBRunLogger, hash_stack_text
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_health_rows(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    now = datetime.now(timezone.utc)

    candles_run = DuckDBRunLogger(warehouse)
    candles_run.start_run(job_name="ingest_candles", started_at=now - timedelta(days=1))
    candles_run.upsert_watermark(
        asset_key="candles_daily",
        scope_key="AAPL",
        watermark_ts=now - timedelta(days=1),
    )
    candles_run.finalize_success(ended_at=now - timedelta(days=1, minutes=-2))

    flow_fail_a = DuckDBRunLogger(warehouse)
    flow_fail_a.start_run(job_name="compute_flow", started_at=now - timedelta(hours=4))
    flow_fail_a.persist_check(
        asset_key="options_flow",
        partition_key="AAPL|2026-02-05",
        check_name="flow_no_null_primary_keys",
        severity="error",
        status="fail",
        message="row_key missing",
        checked_at=now - timedelta(hours=4),
    )
    flow_fail_a.upsert_watermark(
        asset_key="options_flow",
        scope_key="AAPL",
        watermark_ts=now - timedelta(days=9),
    )
    flow_fail_a.finalize_failure(
        RuntimeError("flow failed"),
        stack_text="stack-shared",
        ended_at=now - timedelta(hours=4, minutes=-2),
    )

    flow_fail_b = DuckDBRunLogger(warehouse)
    flow_fail_b.start_run(job_name="compute_flow", started_at=now - timedelta(hours=3))
    flow_fail_b.finalize_failure(
        RuntimeError("flow failed again"),
        stack_text="stack-shared",
        ended_at=now - timedelta(hours=3, minutes=-2),
    )

    flow_success = DuckDBRunLogger(warehouse)
    flow_success.start_run(job_name="compute_flow", started_at=now - timedelta(hours=1))
    flow_success.finalize_success(ended_at=now - timedelta(hours=1, minutes=-2))


def test_health_snapshot_loads_observability_sections(tmp_path: Path) -> None:
    db_path = tmp_path / "health.duckdb"
    _seed_health_rows(db_path)

    snapshot = load_health_snapshot(
        database_path=db_path,
        days=30,
        stale_days=3,
        limit=200,
    )

    assert snapshot.database_exists is True
    assert snapshot.observability_ready is True
    assert snapshot.guidance is None

    latest_by_job = {
        str(row["job_name"]): str(row["status"])
        for row in snapshot.latest_runs.to_dict(orient="records")
    }
    assert latest_by_job["ingest_candles"] == "success"
    assert latest_by_job["compute_flow"] == "success"

    recurring = snapshot.recurring_failures.set_index("error_stack_hash")
    shared_hash = hash_stack_text("stack-shared")
    assert shared_hash is not None
    assert int(recurring.loc[shared_hash, "failure_count"]) == 2
    assert int(recurring.loc[shared_hash, "job_count"]) == 1
    assert recurring.loc[shared_hash, "jobs"] == "compute_flow"

    watermarks = snapshot.watermarks.set_index(["asset_key", "scope_key"])
    assert watermarks.loc[("options_flow", "AAPL"), "freshness"] == "stale"
    assert watermarks.loc[("candles_daily", "AAPL"), "freshness"] == "fresh"

    assert len(snapshot.failed_checks) == 1
    assert str(snapshot.failed_checks.iloc[0]["check_name"]) == "flow_no_null_primary_keys"


def test_health_snapshot_handles_missing_db(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"

    snapshot = load_health_snapshot(database_path=missing_db)

    assert snapshot.database_exists is False
    assert snapshot.observability_ready is False
    assert snapshot.guidance is not None
    assert "db init" in snapshot.guidance
    assert snapshot.latest_runs.empty
    assert snapshot.recurring_failures.empty
    assert snapshot.watermarks.empty
    assert snapshot.failed_checks.empty


def test_health_snapshot_handles_missing_meta_tables(tmp_path: Path) -> None:
    raw_db = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(raw_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
    finally:
        conn.close()

    snapshot = load_health_snapshot(database_path=raw_db)

    assert snapshot.database_exists is True
    assert snapshot.observability_ready is False
    assert snapshot.guidance is not None
    assert "Observability tables are missing" in snapshot.guidance
    assert snapshot.latest_runs.empty
    assert snapshot.recurring_failures.empty
    assert snapshot.watermarks.empty
    assert snapshot.failed_checks.empty
