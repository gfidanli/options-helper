from __future__ import annotations

from options_helper.db.migrations import current_schema_version, ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _table_columns(conn, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {row[1] for row in rows}


def test_duckdb_migrations_v2(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)

    info = ensure_schema(wh)
    assert info.schema_version == 2
    assert info.path == db_path
    assert current_schema_version(wh) == 2

    conn = wh.connect(read_only=True)
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        expected_tables = {
            "candles_daily",
            "option_contracts",
            "option_contract_snapshots",
            "option_bars",
            "option_bars_meta",
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
    finally:
        conn.close()
