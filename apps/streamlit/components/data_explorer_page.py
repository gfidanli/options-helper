from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_DUCKDB_PATH = Path("data/options_helper.duckdb")
_TABLE_COLUMNS = ["table_schema", "table_name", "table_type", "row_count"]
_COLUMN_SCHEMA = ["column_name", "data_type", "is_nullable", "column_default", "ordinal_position"]


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    raw = "" if database_path is None else str(database_path).strip()
    candidate = DEFAULT_DUCKDB_PATH if not raw else Path(raw)
    return candidate.expanduser().resolve()


def list_database_schemas(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    df, note = _run_query_safe(
        """
        SELECT schema_name
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
        ORDER BY schema_name ASC
        """,
        database_path=database_path,
    )
    if note:
        return [], note
    if df.empty:
        return [], None
    values = [str(value).strip() for value in df["schema_name"].tolist() if str(value).strip()]
    return values, None


def list_database_tables(
    *,
    schema: str | None = None,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    conn, note = _open_connection(database_path=database_path)
    if note:
        return _empty_tables(), note

    assert conn is not None
    try:
        if schema and str(schema).strip():
            metadata = conn.execute(
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = ?
                ORDER BY table_name ASC
                """,
                [str(schema).strip()],
            ).df()
        else:
            metadata = conn.execute(
                """
                SELECT table_schema, table_name, table_type
                FROM information_schema.tables
                WHERE table_schema NOT IN ('information_schema', 'pg_catalog')
                ORDER BY table_schema ASC, table_name ASC
                """
            ).df()

        if metadata.empty:
            return _empty_tables(), None

        rows: list[dict[str, Any]] = []
        for item in metadata.to_dict(orient="records"):
            table_schema = str(item.get("table_schema") or "").strip()
            table_name = str(item.get("table_name") or "").strip()
            table_type = str(item.get("table_type") or "").strip()
            row_count = _try_count_rows(conn, schema=table_schema, table=table_name)
            rows.append(
                {
                    "table_schema": table_schema,
                    "table_name": table_name,
                    "table_type": table_type,
                    "row_count": row_count,
                }
            )
        out = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
        return out.sort_values(by=["table_schema", "table_name"], kind="stable").reset_index(drop=True), None
    except Exception as exc:  # noqa: BLE001
        return _empty_tables(), str(exc)
    finally:
        conn.close()


def load_table_columns(
    schema: str,
    table: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    schema_value = str(schema).strip()
    table_value = str(table).strip()
    if not schema_value or not table_value:
        return _empty_columns(), "Schema and table are required."

    df, note = _run_query_safe(
        """
        SELECT
          column_name,
          data_type,
          is_nullable,
          column_default,
          ordinal_position
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position ASC
        """,
        params=[schema_value, table_value],
        database_path=database_path,
    )
    if note:
        return _empty_columns(), note
    if df.empty:
        return _empty_columns(), None
    return df[_COLUMN_SCHEMA].reset_index(drop=True), None


def preview_table_rows(
    schema: str,
    table: str,
    *,
    limit: int = 50,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    schema_value = str(schema).strip()
    table_value = str(table).strip()
    if not schema_value or not table_value:
        return pd.DataFrame(), "Schema and table are required."

    conn, note = _open_connection(database_path=database_path)
    if note:
        return pd.DataFrame(), note

    assert conn is not None
    capped_limit = min(max(1, int(limit)), 500)
    try:
        qualified = _qualified_name(schema_value, table_value)
        frame = conn.execute(f"SELECT * FROM {qualified} LIMIT ?", [capped_limit]).df()
        return frame, None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), str(exc)
    finally:
        conn.close()


def build_select_sql(schema: str, table: str, *, limit: int) -> str:
    capped_limit = min(max(1, int(limit)), 500)
    return f"SELECT * FROM {_qualified_name(schema, table)} LIMIT {capped_limit};"


def _open_connection(
    *,
    database_path: str | Path | None = None,
) -> tuple[duckdb.DuckDBPyConnection | None, str | None]:
    path = resolve_duckdb_path(database_path)
    if not path.exists():
        return None, f"DuckDB database not found: {path}"
    try:
        return duckdb.connect(str(path), read_only=True), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _run_query_safe(
    sql: str,
    *,
    params: list[Any] | None = None,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    conn, note = _open_connection(database_path=database_path)
    if note:
        return pd.DataFrame(), note

    assert conn is not None
    try:
        frame = conn.execute(sql, params or []).df()
        return frame, None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), str(exc)
    finally:
        conn.close()


def _try_count_rows(conn: duckdb.DuckDBPyConnection, *, schema: str, table: str) -> int | None:
    try:
        qualified = _qualified_name(schema, table)
        value = conn.execute(f"SELECT COUNT(*) FROM {qualified}").fetchone()
    except Exception:  # noqa: BLE001
        return None
    if value is None:
        return None
    try:
        return int(value[0])
    except (TypeError, ValueError):
        return None


def _empty_tables() -> pd.DataFrame:
    return pd.DataFrame(columns=_TABLE_COLUMNS)


def _empty_columns() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMN_SCHEMA)


def _qualified_name(schema: str, table: str) -> str:
    return f"{_quote_identifier(schema)}.{_quote_identifier(table)}"


def _quote_identifier(identifier: str) -> str:
    escaped = str(identifier or "").replace('"', '""')
    return f'"{escaped}"'
