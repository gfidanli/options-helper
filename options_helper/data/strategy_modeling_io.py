from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Mapping, Sequence

import duckdb
import pandas as pd

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.storage_runtime import get_default_duckdb_path
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig

AdjustedDataFallbackMode = Literal["warn_and_skip_symbol", "use_unadjusted_ohlc"]
DailySourceMode = Literal["adjusted", "unadjusted"]

_DAILY_COLUMNS: tuple[str, ...] = (
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
)
_INTRADAY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "trade_count",
    "symbol",
    "session_date",
)
_ALLOWED_FALLBACK_MODES: set[str] = {"warn_and_skip_symbol", "use_unadjusted_ohlc"}
_INTRADAY_KIND = "stocks"
_INTRADAY_DATASET = "bars"


@dataclass(frozen=True)
class StrategyModelingUniverseLoadResult:
    symbols: list[str]
    notes: list[str]
    database_path: Path
    database_exists: bool


@dataclass(frozen=True)
class StrategyModelingDailyLoadResult:
    candles_by_symbol: dict[str, pd.DataFrame]
    source_by_symbol: dict[str, DailySourceMode]
    skipped_symbols: list[str]
    missing_symbols: list[str]
    notes: list[str]


@dataclass(frozen=True)
class IntradayCoverageBySymbol:
    symbol: str
    required_days: tuple[date, ...]
    covered_days: tuple[date, ...]
    missing_days: tuple[date, ...]

    @property
    def is_complete(self) -> bool:
        return not self.missing_days


@dataclass(frozen=True)
class StrategyModelingIntradayPreflightResult:
    require_intraday_bars: bool
    coverage_by_symbol: dict[str, IntradayCoverageBySymbol]
    blocked_symbols: list[str]
    notes: list[str]

    @property
    def is_blocked(self) -> bool:
        return bool(self.blocked_symbols)


@dataclass(frozen=True)
class StrategyModelingIntradayLoadResult:
    bars_by_symbol: dict[str, pd.DataFrame]
    preflight: StrategyModelingIntradayPreflightResult
    notes: list[str]


def normalize_symbol(value: object) -> str:
    raw = str(value or "").strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    if database_path is not None:
        return Path(database_path)
    return get_default_duckdb_path()


def list_strategy_modeling_universe(
    *,
    database_path: str | Path | None = None,
) -> StrategyModelingUniverseLoadResult:
    path = resolve_duckdb_path(database_path)
    notes: list[str] = []
    if not path.exists():
        notes.append(f"DuckDB database not found: {path}")
        return StrategyModelingUniverseLoadResult(
            symbols=[],
            notes=notes,
            database_path=path,
            database_exists=False,
        )

    conn = _connect_read_only(path, notes)
    if conn is None:
        return StrategyModelingUniverseLoadResult(
            symbols=[],
            notes=notes,
            database_path=path,
            database_exists=True,
        )

    try:
        if not _table_exists(conn, "candles_daily"):
            notes.append("candles_daily table not found. Run `options-helper ingest candles` first.")
            return StrategyModelingUniverseLoadResult(
                symbols=[],
                notes=notes,
                database_path=path,
                database_exists=True,
            )
        frame = conn.execute(
            """
            SELECT DISTINCT UPPER(symbol) AS symbol
            FROM candles_daily
            WHERE symbol IS NOT NULL
              AND TRIM(symbol) <> ''
              AND interval = '1d'
            ORDER BY symbol ASC
            """
        ).df()
    except Exception as exc:  # noqa: BLE001
        notes.append(str(exc))
        frame = pd.DataFrame()
    finally:
        conn.close()

    symbols = sorted({normalize_symbol(value) for value in frame.get("symbol", []) if normalize_symbol(value)})
    return StrategyModelingUniverseLoadResult(
        symbols=symbols,
        notes=notes,
        database_path=path,
        database_exists=True,
    )


def load_daily_ohlc_history(
    symbols: Sequence[str],
    *,
    database_path: str | Path | None = None,
    policy: StrategyModelingPolicyConfig | None = None,
    adjusted_data_fallback_mode: AdjustedDataFallbackMode = "warn_and_skip_symbol",
    start_date: date | None = None,
    end_date: date | None = None,
    interval: str = "1d",
) -> StrategyModelingDailyLoadResult:
    cfg = policy or StrategyModelingPolicyConfig()
    fallback_mode = _normalize_fallback_mode(adjusted_data_fallback_mode)
    requested_symbols = _dedupe_symbols(symbols)
    notes: list[str] = []

    if not requested_symbols:
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={},
            source_by_symbol={},
            skipped_symbols=[],
            missing_symbols=[],
            notes=notes,
        )

    path = resolve_duckdb_path(database_path)
    if not path.exists():
        notes.append(f"DuckDB database not found: {path}")
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={},
            source_by_symbol={},
            skipped_symbols=[],
            missing_symbols=requested_symbols,
            notes=notes,
        )

    conn = _connect_read_only(path, notes)
    if conn is None:
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={},
            source_by_symbol={},
            skipped_symbols=[],
            missing_symbols=requested_symbols,
            notes=notes,
        )

    if not _table_exists(conn, "candles_daily"):
        conn.close()
        notes.append("candles_daily table not found. Run `options-helper ingest candles` first.")
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={},
            source_by_symbol={},
            skipped_symbols=[],
            missing_symbols=requested_symbols,
            notes=notes,
        )

    candles_by_symbol: dict[str, pd.DataFrame] = {}
    source_by_symbol: dict[str, DailySourceMode] = {}
    skipped_symbols: list[str] = []
    missing_symbols: list[str] = []

    try:
        for symbol in requested_symbols:
            adjusted = _query_daily_history(
                conn,
                symbol,
                auto_adjust=True,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )
            if not adjusted.empty:
                candles_by_symbol[symbol] = adjusted
                source_by_symbol[symbol] = "adjusted"
                continue

            unadjusted = _query_daily_history(
                conn,
                symbol,
                auto_adjust=False,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
            )
            if unadjusted.empty:
                missing_symbols.append(symbol)
                notes.append(f"{symbol}: no daily candles found for requested date range.")
                continue

            if cfg.price_adjustment_policy == "adjusted_ohlc" and fallback_mode == "warn_and_skip_symbol":
                skipped_symbols.append(symbol)
                notes.append(
                    f"{symbol}: adjusted candles unavailable in requested date range; "
                    "skipping symbol (fallback=warn_and_skip_symbol)."
                )
                continue

            candles_by_symbol[symbol] = unadjusted
            source_by_symbol[symbol] = "unadjusted"
            notes.append(
                f"{symbol}: adjusted candles unavailable in requested date range; "
                "using unadjusted fallback."
            )
    finally:
        conn.close()

    return StrategyModelingDailyLoadResult(
        candles_by_symbol=candles_by_symbol,
        source_by_symbol=source_by_symbol,
        skipped_symbols=skipped_symbols,
        missing_symbols=missing_symbols,
        notes=notes,
    )


def build_required_intraday_sessions(
    daily_candles_by_symbol: Mapping[str, pd.DataFrame],
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, list[date]]:
    out: dict[str, list[date]] = {}
    for symbol in sorted(daily_candles_by_symbol):
        frame = daily_candles_by_symbol[symbol]
        if frame is None or frame.empty or "ts" not in frame.columns:
            out[symbol] = []
            continue

        ts = pd.to_datetime(frame["ts"], errors="coerce")
        ts = ts.loc[~ts.isna()]
        if ts.empty:
            out[symbol] = []
            continue

        days = sorted({value.date() for value in ts.tolist()})
        if start_date is not None:
            days = [day for day in days if day >= start_date]
        if end_date is not None:
            days = [day for day in days if day <= end_date]
        out[symbol] = days
    return out


def preflight_intraday_coverage(
    required_sessions_by_symbol: Mapping[str, Sequence[date]],
    *,
    intraday_dir: Path = Path("data/intraday"),
    timeframe: str = "1Min",
    require_intraday_bars: bool = True,
) -> StrategyModelingIntradayPreflightResult:
    store = IntradayStore(intraday_dir)
    notes: list[str] = []
    blocked_symbols: list[str] = []
    coverage_by_symbol: dict[str, IntradayCoverageBySymbol] = {}

    for symbol in sorted(required_sessions_by_symbol):
        required_days = tuple(sorted({day for day in required_sessions_by_symbol[symbol]}))
        covered_days: list[date] = []
        missing_days: list[date] = []

        for day in required_days:
            if _intraday_partition_has_rows(store, symbol=symbol, timeframe=timeframe, day=day):
                covered_days.append(day)
            else:
                missing_days.append(day)

        coverage = IntradayCoverageBySymbol(
            symbol=symbol,
            required_days=required_days,
            covered_days=tuple(covered_days),
            missing_days=tuple(missing_days),
        )
        coverage_by_symbol[symbol] = coverage

        if missing_days:
            note = (
                f"{symbol}: missing intraday coverage for {len(missing_days)}/{len(required_days)} "
                f"required sessions ({timeframe})."
            )
            notes.append(note)
            if require_intraday_bars:
                blocked_symbols.append(symbol)

    blocked_symbols = sorted(blocked_symbols)
    return StrategyModelingIntradayPreflightResult(
        require_intraday_bars=bool(require_intraday_bars),
        coverage_by_symbol=coverage_by_symbol,
        blocked_symbols=blocked_symbols,
        notes=notes,
    )


def load_required_intraday_bars(
    required_sessions_by_symbol: Mapping[str, Sequence[date]],
    *,
    intraday_dir: Path = Path("data/intraday"),
    timeframe: str = "1Min",
    require_intraday_bars: bool = True,
) -> StrategyModelingIntradayLoadResult:
    store = IntradayStore(intraday_dir)
    preflight = preflight_intraday_coverage(
        required_sessions_by_symbol,
        intraday_dir=intraday_dir,
        timeframe=timeframe,
        require_intraday_bars=require_intraday_bars,
    )

    notes = list(preflight.notes)
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol, coverage in preflight.coverage_by_symbol.items():
        if require_intraday_bars and coverage.missing_days:
            notes.append(f"{symbol}: intraday bars not loaded because coverage is incomplete.")
            continue

        parts: list[pd.DataFrame] = []
        for day in coverage.covered_days:
            frame = store.load_partition(_INTRADAY_KIND, _INTRADAY_DATASET, timeframe, symbol, day)
            norm = _normalize_intraday_frame(frame, symbol=symbol, session_date=day)
            if not norm.empty:
                parts.append(norm)

        if parts:
            merged = pd.concat(parts, ignore_index=True)
            merged = merged.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
            bars_by_symbol[symbol] = merged.reindex(columns=list(_INTRADAY_COLUMNS))
        else:
            bars_by_symbol[symbol] = pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    return StrategyModelingIntradayLoadResult(
        bars_by_symbol=bars_by_symbol,
        preflight=preflight,
        notes=notes,
    )


def _dedupe_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in symbols:
        symbol = normalize_symbol(value)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _normalize_fallback_mode(value: AdjustedDataFallbackMode) -> AdjustedDataFallbackMode:
    mode = str(value or "").strip()
    if mode not in _ALLOWED_FALLBACK_MODES:
        allowed = ", ".join(sorted(_ALLOWED_FALLBACK_MODES))
        raise ValueError(f"Invalid adjusted_data_fallback_mode={mode!r}; expected one of: {allowed}")
    return mode  # type: ignore[return-value]


def _connect_read_only(path: Path, notes: list[str]) -> duckdb.DuckDBPyConnection | None:
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Failed to open DuckDB read-only: {exc}")
        return None


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


def _query_daily_history(
    conn: duckdb.DuckDBPyConnection,
    symbol: str,
    *,
    auto_adjust: bool,
    interval: str,
    start_date: date | None,
    end_date: date | None,
) -> pd.DataFrame:
    where_parts = [
        "UPPER(symbol) = ?",
        "interval = ?",
        "auto_adjust = ?",
    ]
    params: list[object] = [symbol, interval, bool(auto_adjust)]

    if start_date is not None:
        where_parts.append("CAST(ts AS DATE) >= ?")
        params.append(start_date.isoformat())
    if end_date is not None:
        where_parts.append("CAST(ts AS DATE) <= ?")
        params.append(end_date.isoformat())

    where_sql = " AND ".join(where_parts)
    frame = conn.execute(
        f"""
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
              ORDER BY CAST(back_adjust AS INTEGER) ASC
            ) AS row_num
          FROM candles_daily
          WHERE {where_sql}
        )
        SELECT ts, open, high, low, close, volume, vwap, trade_count
        FROM ranked
        WHERE row_num = 1
        ORDER BY ts ASC
        """,
        params,
    ).df()
    return _normalize_daily_frame(frame)


def _normalize_daily_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_DAILY_COLUMNS))

    out = frame.copy()
    out["ts"] = pd.to_datetime(out.get("ts"), errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = pd.Series([None] * len(out))
    out = out.dropna(subset=["ts", "open", "high", "low", "close"])
    out = out.sort_values(by="ts", kind="stable").reset_index(drop=True)
    return out.reindex(columns=list(_DAILY_COLUMNS))


def _intraday_partition_has_rows(
    store: IntradayStore,
    *,
    symbol: str,
    timeframe: str,
    day: date,
) -> bool:
    meta = store.load_meta(_INTRADAY_KIND, _INTRADAY_DATASET, timeframe, symbol, day)
    if meta:
        try:
            if int(meta.get("rows") or 0) > 0:
                return True
        except Exception:  # noqa: BLE001
            pass
    frame = store.load_partition(_INTRADAY_KIND, _INTRADAY_DATASET, timeframe, symbol, day)
    return frame is not None and not frame.empty


def _normalize_intraday_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    session_date: date,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out = frame.copy()
    ts_col = "timestamp" if "timestamp" in out.columns else "ts" if "ts" in out.columns else None
    if ts_col is None:
        return pd.DataFrame(columns=list(_INTRADAY_COLUMNS))

    out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True)
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
        else:
            out[column] = pd.Series([None] * len(out))

    out = out.dropna(subset=["timestamp", "open", "high", "low", "close"])
    out["symbol"] = symbol
    out["session_date"] = pd.Timestamp(session_date)
    out = out.sort_values(by="timestamp", kind="stable").reset_index(drop=True)
    return out.reindex(columns=list(_INTRADAY_COLUMNS))


__all__ = [
    "AdjustedDataFallbackMode",
    "DailySourceMode",
    "IntradayCoverageBySymbol",
    "StrategyModelingDailyLoadResult",
    "StrategyModelingIntradayLoadResult",
    "StrategyModelingIntradayPreflightResult",
    "StrategyModelingUniverseLoadResult",
    "build_required_intraday_sessions",
    "list_strategy_modeling_universe",
    "load_daily_ohlc_history",
    "load_required_intraday_bars",
    "normalize_symbol",
    "preflight_intraday_coverage",
    "resolve_duckdb_path",
]
