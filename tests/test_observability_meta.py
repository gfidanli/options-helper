from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from options_helper import cli_deps
from options_helper.data.observability_meta import (
    DuckDBRunLogger,
    NoopRunLogger,
    hash_exception_stack,
    has_observability_tables,
    query_latest_runs,
    query_recent_check_failures,
    query_recent_failures,
    query_watermarks,
)
from options_helper.data.storage_runtime import (
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)
from options_helper.data.store_factory import close_warehouses
from options_helper.db.warehouse import DuckDBWarehouse


def test_duckdb_run_logger_lifecycle_and_writes(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    logger = DuckDBRunLogger(warehouse)

    run_id = logger.start_run(
        job_name="ingest_candles",
        triggered_by="cli",
        provider="alpaca",
        storage_backend="duckdb",
        args={"symbols": ["AAPL"], "root": tmp_path, "flags": {"refresh": True}},
        git_sha="abc123",
        app_version="0.1.0",
    )
    assert run_id

    logger.log_asset_success(
        asset_key="candles_daily",
        asset_kind="table",
        partition_key="AAPL|2026-02-05",
        rows_inserted=3,
        min_event_ts=datetime(2026, 2, 3, tzinfo=timezone.utc),
        max_event_ts=datetime(2026, 2, 5, tzinfo=timezone.utc),
        extra={"symbols": ["AAPL"]},
    )
    logger.upsert_watermark(
        asset_key="candles_daily",
        scope_key="AAPL",
        watermark_ts=datetime(2026, 2, 5, tzinfo=timezone.utc),
    )
    check_id = logger.persist_check(
        asset_key="candles_daily",
        partition_key="AAPL|2026-02-05",
        check_name="candles_unique_symbol_date",
        severity="error",
        status="pass",
        metrics={"duplicates": 0},
        message="ok",
    )
    logger.finalize_success()

    conn = warehouse.connect(read_only=True)
    try:
        run_row = conn.execute(
            """
            SELECT status, provider, storage_backend, args_json, git_sha, app_version
            FROM meta.ingestion_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        assert run_row is not None
        assert run_row[0] == "success"
        assert run_row[1] == "alpaca"
        assert run_row[2] == "duckdb"
        assert run_row[4] == "abc123"
        assert run_row[5] == "0.1.0"

        parsed_args = json.loads(run_row[3])
        assert parsed_args["symbols"] == ["AAPL"]
        assert parsed_args["root"] == str(tmp_path)
        assert parsed_args["flags"]["refresh"] is True

        asset_row = conn.execute(
            """
            SELECT status, rows_inserted, partition_key, extra_json
            FROM meta.ingestion_run_assets
            WHERE run_id = ? AND asset_key = 'candles_daily'
            """,
            [run_id],
        ).fetchone()
        assert asset_row is not None
        assert asset_row[0] == "success"
        assert asset_row[1] == 3
        assert asset_row[2] == "AAPL|2026-02-05"
        assert json.loads(asset_row[3])["symbols"] == ["AAPL"]

        watermark_row = conn.execute(
            """
            SELECT scope_key, last_run_id
            FROM meta.asset_watermarks
            WHERE asset_key = 'candles_daily'
            """
        ).fetchone()
        assert watermark_row == ("AAPL", run_id)

        check_row = conn.execute(
            """
            SELECT check_name, severity, status, run_id
            FROM meta.asset_checks
            WHERE check_id = ?
            """,
            [check_id],
        ).fetchone()
        assert check_row == ("candles_unique_symbol_date", "error", "pass", run_id)
    finally:
        conn.close()


def test_duckdb_run_logger_failure_persists_stack_hash(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    logger = DuckDBRunLogger(warehouse)
    run_id = logger.start_run(job_name="compute_flow", triggered_by="cli")

    try:
        raise ValueError("bad flow window")
    except ValueError as exc:
        expected_hash = hash_exception_stack(exc)
        logger.finalize_failure(exc)

    conn = warehouse.connect(read_only=True)
    try:
        row = conn.execute(
            """
            SELECT status, error_type, error_message, error_stack_hash
            FROM meta.ingestion_runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert row[1] == "ValueError"
        assert "bad flow window" in str(row[2])
        assert row[3] == expected_hash
    finally:
        conn.close()


def test_duckdb_run_logger_asset_upsert_and_watermark_monotonic(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    logger = DuckDBRunLogger(warehouse)
    run_id = logger.start_run(job_name="snapshot_options")

    logger.log_asset_success(
        asset_key="options_snapshot_file",
        asset_kind="file",
        partition_key="AAPL|2026-02-05",
        bytes_written=50,
    )
    logger.log_asset_failure(
        asset_key="options_snapshot_file",
        asset_kind="file",
        partition_key="AAPL|2026-02-05",
        bytes_written=120,
        extra={"reason": "partial write"},
    )

    latest_ts = datetime(2026, 2, 5, tzinfo=timezone.utc)
    older_ts = datetime(2026, 2, 1, tzinfo=timezone.utc)
    logger.upsert_watermark(
        asset_key="options_snapshot_file",
        scope_key="AAPL",
        watermark_ts=latest_ts,
    )
    logger.upsert_watermark(
        asset_key="options_snapshot_file",
        scope_key="AAPL",
        watermark_ts=older_ts,
    )
    logger.finalize_success()

    conn = warehouse.connect(read_only=True)
    try:
        asset_row = conn.execute(
            """
            SELECT status, bytes_written, extra_json
            FROM meta.ingestion_run_assets
            WHERE run_id = ?
              AND asset_key = 'options_snapshot_file'
              AND partition_key = 'AAPL|2026-02-05'
            """,
            [run_id],
        ).fetchone()
        assert asset_row is not None
        assert asset_row[0] == "failed"
        assert asset_row[1] == 120
        assert json.loads(asset_row[2])["reason"] == "partial write"

        watermark_row = conn.execute(
            """
            SELECT watermark_ts
            FROM meta.asset_watermarks
            WHERE asset_key = 'options_snapshot_file' AND scope_key = 'AAPL'
            """
        ).fetchone()
        assert watermark_row is not None
        assert watermark_row[0].date().isoformat() == latest_ts.date().isoformat()
    finally:
        conn.close()


def test_health_query_helpers_return_expected_rows(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")

    now = datetime.now(timezone.utc)

    run_a = DuckDBRunLogger(warehouse)
    run_a.start_run(job_name="ingest_candles", started_at=now - timedelta(days=1))
    run_a.upsert_watermark(
        asset_key="candles_daily",
        scope_key="AAPL",
        watermark_ts=now - timedelta(days=10),
    )
    run_a.persist_check(
        asset_key="candles_daily",
        check_name="candles_no_negative_prices",
        severity="error",
        status="pass",
    )
    run_a.finalize_success(ended_at=now - timedelta(days=1, minutes=-1))

    run_b = DuckDBRunLogger(warehouse)
    run_b_id = run_b.start_run(job_name="compute_flow", started_at=now - timedelta(hours=2))
    run_b.persist_check(
        asset_key="options_flow",
        partition_key="AAPL|2026-02-05",
        check_name="flow_no_null_primary_keys",
        severity="error",
        status="fail",
        message="missing row_key",
        run_id=run_b_id,
        checked_at=now - timedelta(hours=2),
    )
    run_b.finalize_failure(RuntimeError("flow failed"), ended_at=now - timedelta(hours=2, minutes=-1))

    run_b2 = DuckDBRunLogger(warehouse)
    run_b2.start_run(job_name="compute_flow", started_at=now - timedelta(hours=1))
    run_b2.finalize_success(ended_at=now - timedelta(hours=1, minutes=-1))

    latest_runs = query_latest_runs(warehouse, days=7, limit=20)
    assert {row["job_name"] for row in latest_runs} == {"ingest_candles", "compute_flow"}
    latest_by_job = {row["job_name"]: row for row in latest_runs}
    assert latest_by_job["compute_flow"]["status"] == "success"

    recent_failures = query_recent_failures(warehouse, days=7, limit=20)
    assert len(recent_failures) == 1
    assert recent_failures[0]["job_name"] == "compute_flow"
    assert recent_failures[0]["status"] == "failed"

    watermarks = query_watermarks(warehouse, stale_days=5, limit=20)
    assert len(watermarks) == 1
    assert watermarks[0]["asset_key"] == "candles_daily"
    assert watermarks[0]["staleness_days"] >= 5

    failed_checks = query_recent_check_failures(warehouse, days=7, limit=20)
    assert len(failed_checks) == 1
    assert failed_checks[0]["check_name"] == "flow_no_null_primary_keys"
    assert failed_checks[0]["status"] == "fail"

    assert has_observability_tables(warehouse) is True


def test_health_query_helpers_gracefully_handle_missing_meta_tables(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")

    assert has_observability_tables(warehouse) is False
    assert query_latest_runs(warehouse) == []
    assert query_recent_failures(warehouse) == []
    assert query_watermarks(warehouse) == []
    assert query_recent_check_failures(warehouse) == []


def test_noop_run_logger_does_not_write(tmp_path: Path) -> None:
    warehouse = DuckDBWarehouse(tmp_path / "options.duckdb")
    # Ensure tables exist so this test can assert no writes happened.
    DuckDBRunLogger(warehouse).start_run(job_name="bootstrap")

    conn = warehouse.connect(read_only=True)
    try:
        before = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM meta.ingestion_runs),
              (SELECT count(*) FROM meta.ingestion_run_assets),
              (SELECT count(*) FROM meta.asset_watermarks),
              (SELECT count(*) FROM meta.asset_checks)
            """
        ).fetchone()
    finally:
        conn.close()

    logger = NoopRunLogger()
    logger.start_run(job_name="compute_derived", args={"symbols": ["AAPL"]})
    logger.log_asset_success(asset_key="derived_metrics", asset_kind="table")
    logger.upsert_watermark(asset_key="derived_metrics", scope_key="AAPL", watermark_ts="2026-02-05")
    logger.persist_check(
        asset_key="derived_metrics",
        check_name="derived_no_duplicate_keys",
        severity="error",
        status="pass",
    )
    logger.finalize_success()

    conn = warehouse.connect(read_only=True)
    try:
        after = conn.execute(
            """
            SELECT
              (SELECT count(*) FROM meta.ingestion_runs),
              (SELECT count(*) FROM meta.ingestion_run_assets),
              (SELECT count(*) FROM meta.asset_watermarks),
              (SELECT count(*) FROM meta.asset_checks)
            """
        ).fetchone()
    finally:
        conn.close()

    assert before == after


def test_build_run_logger_storage_aware_behavior(tmp_path: Path) -> None:
    close_warehouses()
    duckdb_path = tmp_path / "options.duckdb"
    backend_token = set_default_storage_backend("duckdb")
    path_token = set_default_duckdb_path(duckdb_path)

    try:
        duckdb_logger = cli_deps.build_run_logger(
            job_name="build_dashboard",
            provider="alpaca",
            args={"symbols": ["AAPL", "MSFT"]},
        )
        assert isinstance(duckdb_logger, DuckDBRunLogger)
        duckdb_logger.finalize_success()

        warehouse = DuckDBWarehouse(duckdb_path)
        conn = warehouse.connect(read_only=True)
        try:
            rows = conn.execute(
                """
                SELECT count(*), max(storage_backend)
                FROM meta.ingestion_runs
                WHERE job_name = 'build_dashboard'
                """
            ).fetchone()
            assert rows == (1, "duckdb")
        finally:
            conn.close()

        close_warehouses()
        reset_default_storage_backend(backend_token)
        backend_token = set_default_storage_backend("filesystem")

        noop_logger = cli_deps.build_run_logger(job_name="build_dashboard")
        assert isinstance(noop_logger, NoopRunLogger)
        noop_logger.finalize_success()

        conn = warehouse.connect(read_only=True)
        try:
            # Filesystem mode uses NoopRunLogger and does not write to DuckDB.
            count = conn.execute(
                "SELECT count(*) FROM meta.ingestion_runs WHERE job_name = 'build_dashboard'"
            ).fetchone()
            assert count == (1,)
        finally:
            conn.close()
    finally:
        close_warehouses()
        reset_default_duckdb_path(path_token)
        reset_default_storage_backend(backend_token)
