from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb


@dataclass(frozen=True)
class DuckDBWarehouse:
    """Thin wrapper around a DuckDB file connection.

    - Designed for *embedded* use (single machine).
    - Callers should prefer `transaction()` for multi-statement operations.
    """

    path: Path

    def connect(self, *, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(self.path), read_only=read_only)

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = self.connect(read_only=False)
        try:
            conn.execute("BEGIN TRANSACTION")
            yield conn
            conn.execute("COMMIT")
        except Exception:  # noqa: BLE001
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> None:
        conn = self.connect(read_only=False)
        try:
            if params is None:
                conn.execute(sql)
            else:
                conn.execute(sql, params)
        finally:
            conn.close()

    def fetch_df(self, sql: str, params: Sequence[Any] | None = None):
        conn = self.connect(read_only=True)
        try:
            if params is None:
                return conn.execute(sql).df()
            return conn.execute(sql, params).df()
        finally:
            conn.close()
