from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from options_helper.db.warehouse import DuckDBWarehouse


@dataclass(frozen=True)
class SchemaInfo:
    path: Path
    schema_version: int


def _ensure_migrations_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          schema_version INTEGER PRIMARY KEY,
          applied_at TIMESTAMP NOT NULL DEFAULT current_timestamp
        );
        """
    )


def current_schema_version(warehouse: DuckDBWarehouse) -> int:
    """Return the current schema version (0 if uninitialized)."""
    with warehouse.transaction() as tx:
        _ensure_migrations_table(tx)
        row = tx.execute("SELECT max(schema_version) AS v FROM schema_migrations").fetchone()
        if not row or row[0] is None:
            return 0
        return int(row[0])


def _apply_schema_migration(warehouse: DuckDBWarehouse, *, schema_version: int, schema_path: Path) -> None:
    schema_sql = schema_path.read_text(encoding="utf-8")
    with warehouse.transaction() as tx:
        _ensure_migrations_table(tx)
        tx.execute(schema_sql)
        tx.execute(
            """
            INSERT INTO schema_migrations(schema_version)
            SELECT ?
            WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = ?);
            """,
            [schema_version, schema_version],
        )


def ensure_schema(warehouse: DuckDBWarehouse) -> SchemaInfo:
    """Idempotently apply schema migrations up to the latest known version."""
    migration_paths = (
        (1, Path(__file__).with_name("schema_v1.sql")),
        (2, Path(__file__).with_name("schema_v2.sql")),
        (3, Path(__file__).with_name("schema_v3.sql")),
        (4, Path(__file__).with_name("schema_v4.sql")),
        (5, Path(__file__).with_name("schema_v5.sql")),
    )
    v = current_schema_version(warehouse)
    for schema_version, schema_path in migration_paths:
        if v >= schema_version:
            continue
        _apply_schema_migration(warehouse, schema_version=schema_version, schema_path=schema_path)
        v = schema_version

    return SchemaInfo(path=warehouse.path, schema_version=v)
