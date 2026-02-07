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


def ensure_schema(warehouse: DuckDBWarehouse) -> SchemaInfo:
    """Idempotently apply schema migrations up to the latest known version."""
    v1_path = Path(__file__).with_name("schema_v1.sql")
    v2_path = Path(__file__).with_name("schema_v2.sql")
    v3_path = Path(__file__).with_name("schema_v3.sql")
    v4_path = Path(__file__).with_name("schema_v4.sql")
    v5_path = Path(__file__).with_name("schema_v5.sql")
    v = current_schema_version(warehouse)

    if v < 1:
        schema_sql = v1_path.read_text(encoding="utf-8")
        with warehouse.transaction() as tx:
            _ensure_migrations_table(tx)
            tx.execute(schema_sql)

            # Record that v1 is applied (idempotent).
            tx.execute(
                """
                INSERT INTO schema_migrations(schema_version)
                SELECT 1
                WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = 1);
                """
            )
        v = 1

    if v < 2:
        schema_sql = v2_path.read_text(encoding="utf-8")
        with warehouse.transaction() as tx:
            _ensure_migrations_table(tx)
            tx.execute(schema_sql)

            # Record that v2 is applied (idempotent).
            tx.execute(
                """
                INSERT INTO schema_migrations(schema_version)
                SELECT 2
                WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = 2);
                """
            )
        v = 2

    if v < 3:
        schema_sql = v3_path.read_text(encoding="utf-8")
        with warehouse.transaction() as tx:
            _ensure_migrations_table(tx)
            tx.execute(schema_sql)

            # Record that v3 is applied (idempotent).
            tx.execute(
                """
                INSERT INTO schema_migrations(schema_version)
                SELECT 3
                WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = 3);
                """
            )
        v = 3

    if v < 4:
        schema_sql = v4_path.read_text(encoding="utf-8")
        with warehouse.transaction() as tx:
            _ensure_migrations_table(tx)
            tx.execute(schema_sql)

            # Record that v4 is applied (idempotent).
            tx.execute(
                """
                INSERT INTO schema_migrations(schema_version)
                SELECT 4
                WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = 4);
                """
            )
        v = 4

    if v < 5:
        schema_sql = v5_path.read_text(encoding="utf-8")
        with warehouse.transaction() as tx:
            _ensure_migrations_table(tx)
            tx.execute(schema_sql)

            # Record that v5 is applied (idempotent).
            tx.execute(
                """
                INSERT INTO schema_migrations(schema_version)
                SELECT 5
                WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE schema_version = 5);
                """
            )
        v = 5

    return SchemaInfo(path=warehouse.path, schema_version=v)
