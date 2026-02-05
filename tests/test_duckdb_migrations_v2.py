from __future__ import annotations

from pathlib import Path

from options_helper.db import migrations
from options_helper.db.migrations import current_schema_version, ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _table_columns(conn, table: str, *, schema: str = "main") -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{schema}.{table}')").fetchall()
    return {row[1] for row in rows}


def _index_names(conn, table: str, *, schema: str = "main") -> set[str]:
    rows = conn.execute(
        """
        SELECT index_name
        FROM duckdb_indexes()
        WHERE schema_name = ? AND table_name = ?;
        """,
        [schema, table],
    ).fetchall()
    return {row[0] for row in rows}


def _bootstrap_v2_database(wh: DuckDBWarehouse) -> None:
    schema_v1 = Path(migrations.__file__).with_name("schema_v1.sql").read_text(encoding="utf-8")
    schema_v2 = Path(migrations.__file__).with_name("schema_v2.sql").read_text(encoding="utf-8")

    with wh.transaction() as tx:
        tx.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
              schema_version INTEGER PRIMARY KEY,
              applied_at TIMESTAMP NOT NULL DEFAULT current_timestamp
            );
            """
        )
        tx.execute(schema_v1)
        tx.execute(schema_v2)
        tx.execute("INSERT INTO schema_migrations(schema_version) VALUES (1), (2);")


def test_duckdb_migrations_v3_contains_v2_and_new_meta_tables(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)

    info = ensure_schema(wh)
    assert info.schema_version == 3
    assert info.path == db_path
    assert current_schema_version(wh) == 3

    conn = wh.connect(read_only=True)
    try:
        tables = {
            (row[0], row[1])
            for row in conn.execute(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables;
                """
            ).fetchall()
        }
        expected_tables = {
            ("main", "candles_daily"),
            ("main", "option_contracts"),
            ("main", "option_contract_snapshots"),
            ("main", "option_bars"),
            ("main", "option_bars_meta"),
            ("main", "options_flow"),
            ("meta", "ingestion_runs"),
            ("meta", "ingestion_run_assets"),
            ("meta", "asset_watermarks"),
            ("meta", "asset_checks"),
        }
        assert expected_tables.issubset(tables)

        candle_cols = _table_columns(conn, "candles_daily")
        assert {"vwap", "trade_count"}.issubset(candle_cols)

        contract_cols = _table_columns(conn, "option_contracts")
        assert {
            "contract_symbol",
            "underlying",
            "expiry",
            "option_type",
            "strike",
            "multiplier",
            "provider",
            "updated_at",
        }.issubset(contract_cols)

        snapshot_cols = _table_columns(conn, "option_contract_snapshots")
        assert {
            "contract_symbol",
            "as_of_date",
            "open_interest",
            "close_price",
            "provider",
            "updated_at",
        }.issubset(snapshot_cols)

        bar_cols = _table_columns(conn, "option_bars")
        assert {
            "contract_symbol",
            "interval",
            "ts",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
            "provider",
            "updated_at",
        }.issubset(bar_cols)

        meta_cols = _table_columns(conn, "option_bars_meta")
        assert {"status", "rows", "error_count"}.issubset(meta_cols)

        run_cols = _table_columns(conn, "ingestion_runs", schema="meta")
        assert {"run_id", "job_name", "triggered_by", "status", "started_at", "ended_at"}.issubset(
            run_cols
        )

        run_asset_cols = _table_columns(conn, "ingestion_run_assets", schema="meta")
        assert {"run_id", "asset_key", "partition_key", "status"}.issubset(run_asset_cols)

        watermark_cols = _table_columns(conn, "asset_watermarks", schema="meta")
        assert {"asset_key", "scope_key", "watermark_ts", "updated_at"}.issubset(watermark_cols)

        check_cols = _table_columns(conn, "asset_checks", schema="meta")
        assert {"check_id", "asset_key", "check_name", "severity", "status", "checked_at"}.issubset(
            check_cols
        )

        flow_cols = _table_columns(conn, "options_flow")
        assert {
            "symbol",
            "as_of",
            "from_date",
            "to_date",
            "window_size",
            "group_by",
            "row_key",
            "delta_oi",
            "delta_oi_notional",
            "volume_notional",
            "delta_notional",
            "n_pairs",
        }.issubset(flow_cols)

        ingestion_run_indexes = _index_names(conn, "ingestion_runs", schema="meta")
        assert {
            "idx_ingestion_runs_job_started",
            "idx_ingestion_runs_status_started",
            "idx_ingestion_runs_started_at",
        }.issubset(ingestion_run_indexes)

        asset_check_indexes = _index_names(conn, "asset_checks", schema="meta")
        assert {
            "idx_asset_checks_status_checked_at",
            "idx_asset_checks_asset_checked_at",
            "idx_asset_checks_run_id",
        }.issubset(asset_check_indexes)

        options_flow_indexes = _index_names(conn, "options_flow")
        assert {
            "idx_options_flow_symbol_dates",
            "idx_options_flow_as_of",
            "idx_options_flow_group_by",
            "idx_options_flow_expiry",
        }.issubset(options_flow_indexes)
    finally:
        conn.close()


def test_duckdb_migrations_upgrade_v2_to_v3(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)
    _bootstrap_v2_database(wh)

    assert current_schema_version(wh) == 2

    info = ensure_schema(wh)
    assert info.schema_version == 3
    assert current_schema_version(wh) == 3

    # Idempotent repeat should not duplicate migration rows.
    assert ensure_schema(wh).schema_version == 3

    conn = wh.connect(read_only=True)
    try:
        versions = [
            row[0]
            for row in conn.execute(
                """
                SELECT schema_version
                FROM schema_migrations
                ORDER BY schema_version;
                """
            ).fetchall()
        ]
        assert versions == [1, 2, 3]
    finally:
        conn.close()
