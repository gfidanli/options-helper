from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from options_helper.data.storage_runtime import get_default_duckdb_path


@dataclass(frozen=True)
class SymbolCoverageFrames:
    symbol: str
    database_path: Path
    database_exists: bool
    candles: pd.DataFrame
    snapshot_headers: pd.DataFrame
    contracts: pd.DataFrame
    contract_snapshots: pd.DataFrame
    option_bars_meta: pd.DataFrame
    notes: list[str]


def load_symbol_coverage_frames(
    symbol: str,
    *,
    duckdb_path: Path | None = None,
) -> SymbolCoverageFrames:
    sym = _normalize_symbol(symbol)
    path = _resolve_duckdb_path(duckdb_path)
    notes: list[str] = []

    if not path.exists():
        notes.append(f"DuckDB database not found: {path}")
        return SymbolCoverageFrames(
            symbol=sym,
            database_path=path,
            database_exists=False,
            candles=pd.DataFrame(),
            snapshot_headers=pd.DataFrame(),
            contracts=pd.DataFrame(),
            contract_snapshots=pd.DataFrame(),
            option_bars_meta=pd.DataFrame(),
            notes=notes,
        )

    conn: duckdb.DuckDBPyConnection | None = None
    try:
        conn = duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Failed to open DuckDB read-only: {exc}")
        return SymbolCoverageFrames(
            symbol=sym,
            database_path=path,
            database_exists=True,
            candles=pd.DataFrame(),
            snapshot_headers=pd.DataFrame(),
            contracts=pd.DataFrame(),
            contract_snapshots=pd.DataFrame(),
            option_bars_meta=pd.DataFrame(),
            notes=notes,
        )

    try:
        candles = _query_candles(conn, sym, notes)
        snapshot_headers = _query_snapshot_headers(conn, sym, notes)
        contracts = _query_contracts(conn, sym, notes)
        contract_snapshots = _query_contract_snapshots(conn, sym, notes)
        option_bars_meta = _query_option_bars_meta(conn, sym, notes)
    finally:
        conn.close()

    return SymbolCoverageFrames(
        symbol=sym,
        database_path=path,
        database_exists=True,
        candles=candles,
        snapshot_headers=snapshot_headers,
        contracts=contracts,
        contract_snapshots=contract_snapshots,
        option_bars_meta=option_bars_meta,
        notes=notes,
    )


def list_candidate_symbols(*, duckdb_path: Path | None = None) -> tuple[list[str], str | None]:
    path = _resolve_duckdb_path(duckdb_path)
    if not path.exists():
        return [], f"DuckDB database not found: {path}"
    try:
        conn = duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)

    try:
        symbols: set[str] = set()
        if _table_exists(conn, "candles_daily"):
            frame = conn.execute(
                """
                SELECT DISTINCT UPPER(symbol) AS symbol
                FROM candles_daily
                WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
                """
            ).df()
            symbols.update(_symbols_from_frame(frame))

        if _table_exists(conn, "options_snapshot_headers"):
            frame = conn.execute(
                """
                SELECT DISTINCT UPPER(symbol) AS symbol
                FROM options_snapshot_headers
                WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
                """
            ).df()
            symbols.update(_symbols_from_frame(frame))

        if _table_exists(conn, "option_contracts"):
            frame = conn.execute(
                """
                SELECT DISTINCT UPPER(underlying) AS symbol
                FROM option_contracts
                WHERE underlying IS NOT NULL AND TRIM(underlying) <> ''
                """
            ).df()
            symbols.update(_symbols_from_frame(frame))

        return sorted(symbols), None
    except Exception as exc:  # noqa: BLE001
        return [], str(exc)
    finally:
        conn.close()


def _query_candles(conn: duckdb.DuckDBPyConnection, symbol: str, notes: list[str]) -> pd.DataFrame:
    if not _table_exists(conn, "candles_daily"):
        notes.append("Table missing: candles_daily")
        return pd.DataFrame()
    try:
        return conn.execute(
            """
            WITH ranked AS (
              SELECT
                ts::TIMESTAMP AS ts,
                open,
                high,
                low,
                close,
                volume,
                vwap,
                trade_count,
                ROW_NUMBER() OVER (
                  PARTITION BY ts
                  ORDER BY CAST(auto_adjust AS INTEGER) DESC, CAST(back_adjust AS INTEGER) ASC
                ) AS row_num
              FROM candles_daily
              WHERE UPPER(symbol) = ?
                AND interval = '1d'
            )
            SELECT ts, open, high, low, close, volume, vwap, trade_count
            FROM ranked
            WHERE row_num = 1
            ORDER BY ts ASC
            """,
            [symbol],
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"candles_daily query failed: {exc}")
        return pd.DataFrame()


def _query_snapshot_headers(conn: duckdb.DuckDBPyConnection, symbol: str, notes: list[str]) -> pd.DataFrame:
    if not _table_exists(conn, "options_snapshot_headers"):
        notes.append("Table missing: options_snapshot_headers")
        return pd.DataFrame()
    try:
        return conn.execute(
            """
            SELECT snapshot_date, provider, contracts, updated_at
            FROM options_snapshot_headers
            WHERE UPPER(symbol) = ?
            ORDER BY snapshot_date ASC
            """,
            [symbol],
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"options_snapshot_headers query failed: {exc}")
        return pd.DataFrame()


def _query_contracts(conn: duckdb.DuckDBPyConnection, symbol: str, notes: list[str]) -> pd.DataFrame:
    if not _table_exists(conn, "option_contracts"):
        notes.append("Table missing: option_contracts")
        return pd.DataFrame()
    try:
        return conn.execute(
            """
            SELECT
              contract_symbol,
              underlying,
              expiry,
              option_type,
              strike,
              multiplier,
              provider,
              updated_at
            FROM option_contracts
            WHERE UPPER(underlying) = ?
            ORDER BY expiry ASC, strike ASC, option_type ASC, contract_symbol ASC
            """,
            [symbol],
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"option_contracts query failed: {exc}")
        return pd.DataFrame()


def _query_contract_snapshots(conn: duckdb.DuckDBPyConnection, symbol: str, notes: list[str]) -> pd.DataFrame:
    if not _table_exists(conn, "option_contract_snapshots"):
        notes.append("Table missing: option_contract_snapshots")
        return pd.DataFrame()
    if not _table_exists(conn, "option_contracts"):
        notes.append("Table missing: option_contracts (required for symbol filter)")
        return pd.DataFrame()
    try:
        return conn.execute(
            """
            SELECT
              s.contract_symbol,
              s.as_of_date,
              s.open_interest,
              s.open_interest_date,
              s.close_price,
              s.close_price_date,
              s.provider,
              s.updated_at
            FROM option_contract_snapshots s
            JOIN option_contracts c ON c.contract_symbol = s.contract_symbol
            WHERE UPPER(c.underlying) = ?
            ORDER BY s.as_of_date ASC, s.contract_symbol ASC
            """,
            [symbol],
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"option_contract_snapshots query failed: {exc}")
        return pd.DataFrame()


def _query_option_bars_meta(conn: duckdb.DuckDBPyConnection, symbol: str, notes: list[str]) -> pd.DataFrame:
    if not _table_exists(conn, "option_bars_meta"):
        notes.append("Table missing: option_bars_meta")
        return pd.DataFrame()
    if not _table_exists(conn, "option_contracts"):
        notes.append("Table missing: option_contracts (required for bars coverage)")
        return pd.DataFrame()
    try:
        return conn.execute(
            """
            SELECT
              m.contract_symbol,
              m.interval,
              m.provider,
              m.status,
              m.rows,
              m.start_ts,
              m.end_ts,
              m.last_success_at,
              m.last_attempt_at,
              m.last_error,
              m.error_count
            FROM option_bars_meta m
            JOIN option_contracts c ON c.contract_symbol = m.contract_symbol
            WHERE UPPER(c.underlying) = ?
              AND m.interval = '1d'
            ORDER BY m.contract_symbol ASC
            """,
            [symbol],
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(f"option_bars_meta query failed: {exc}")
        return pd.DataFrame()


def _table_exists(conn: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        row = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = ?
            LIMIT 1
            """,
            [table],
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return row is not None


def _symbols_from_frame(frame: pd.DataFrame) -> set[str]:
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return set()
    out: set[str] = set()
    for value in frame["symbol"].tolist():
        sym = _normalize_symbol(value)
        if sym:
            out.add(sym)
    return out


def _normalize_symbol(value: object) -> str:
    text = str(value or "").strip().upper()
    return "".join(ch for ch in text if ch.isalnum() or ch in {".", "-", "_"})


def _resolve_duckdb_path(duckdb_path: Path | None) -> Path:
    if duckdb_path is not None:
        return Path(duckdb_path)
    return get_default_duckdb_path()
