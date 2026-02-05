from __future__ import annotations

from options_helper.db.migrations import current_schema_version, ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def test_duckdb_migrations_v1(tmp_path):
    db_path = tmp_path / "options.duckdb"
    wh = DuckDBWarehouse(db_path)

    assert current_schema_version(wh) == 0
    info = ensure_schema(wh)
    assert info.schema_version == 2
    assert info.path == db_path

    assert current_schema_version(wh) == 2
