from __future__ import annotations

from options_helper.db.migrations import current_schema_version, ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def test_duckdb_migrations_v1(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)

    assert current_schema_version(wh) == 0
    info = ensure_schema(wh)
    assert info.schema_version == 3
    assert info.path == db_path

    assert current_schema_version(wh) == 3


def test_duckdb_migrations_are_idempotent(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)

    first = ensure_schema(wh)
    second = ensure_schema(wh)
    assert first.schema_version == 3
    assert second.schema_version == 3
    assert current_schema_version(wh) == 3

    conn = wh.connect(read_only=True)
    try:
        versions = [
            row[0]
            for row in conn.execute(
                "SELECT schema_version FROM schema_migrations ORDER BY schema_version"
            ).fetchall()
        ]
        assert versions == [1, 2, 3]
    finally:
        conn.close()
