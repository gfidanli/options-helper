from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path
from apps.streamlit.components.queries import run_query
from options_helper.analysis.tail_risk import TailRiskConfig, TailRiskResult, compute_tail_risk

_CANDLES_COLUMNS = ["ts", "close"]
_DERIVED_COLUMNS = [
    "date",
    "spot",
    "atm_iv_near",
    "rv_20d",
    "rv_60d",
    "iv_rv_20d",
    "atm_iv_near_percentile",
    "iv_term_slope",
]


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def list_candle_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    df, note = _run_query_safe(
        """
        SELECT DISTINCT UPPER(symbol) AS symbol
        FROM candles_daily
        WHERE symbol IS NOT NULL
          AND TRIM(symbol) <> ''
          AND interval = '1d'
        ORDER BY symbol ASC
        """,
        database_path=database_path,
    )
    if note is not None:
        return [], note
    if df.empty:
        return [], None
    symbols = [normalize_symbol(value) for value in df["symbol"].tolist()]
    return sorted(symbol for symbol in symbols if symbol), None


def load_candles_close(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    limit: int = 4000,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH ranked AS (
          SELECT
            ts::TIMESTAMP AS ts,
            close,
            ROW_NUMBER() OVER (
              PARTITION BY ts
              ORDER BY CAST(auto_adjust AS INTEGER) DESC, CAST(back_adjust AS INTEGER) ASC
            ) AS row_num
          FROM candles_daily
          WHERE UPPER(symbol) = ?
            AND interval = '1d'
        )
        SELECT ts, close
        FROM ranked
        WHERE row_num = 1
        ORDER BY ts DESC
        LIMIT ?
        """,
        params=[sym, max(1, int(limit))],
        database_path=database_path,
    )
    if note is not None:
        return pd.DataFrame(columns=_CANDLES_COLUMNS), note
    if df.empty:
        return pd.DataFrame(columns=_CANDLES_COLUMNS), None

    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["ts", "close"])
    out = out.sort_values(by="ts", kind="stable").reset_index(drop=True)
    return out.reindex(columns=_CANDLES_COLUMNS), None


def load_latest_derived_row(
    symbol: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        f"""
        SELECT {", ".join(_DERIVED_COLUMNS)}
        FROM derived_daily
        WHERE UPPER(symbol) = ?
        ORDER BY date DESC
        LIMIT 1
        """,
        params=[sym],
        database_path=database_path,
    )
    if note is not None:
        return None, note
    if df.empty:
        return None, None
    row = df.iloc[0].to_dict()
    row["date"] = str(row.get("date") or "")
    for column in _DERIVED_COLUMNS:
        if column != "date" and column in row:
            row[column] = _safe_float(row.get(column))
    return row, None


@st.cache_data(ttl=60, show_spinner=False)
def _compute_tail_risk_cached(
    *,
    symbol: str,
    database_path: str | None,
    lookback_days: int,
    horizon_days: int,
    num_simulations: int,
    seed: int,
    var_confidence: float,
) -> tuple[TailRiskResult | None, str | None]:
    candles_df, candles_note = load_candles_close(
        symbol,
        database_path=database_path,
        limit=max(lookback_days + 300, 365),
    )
    if candles_note is not None:
        return None, candles_note
    if candles_df.empty:
        return None, f"No candles found for {normalize_symbol(symbol)}."

    close = pd.Series(
        candles_df["close"].tolist(),
        index=pd.to_datetime(candles_df["ts"], errors="coerce"),
        dtype="float64",
    )
    close = close.dropna()
    if close.empty:
        return None, f"No usable close prices found for {normalize_symbol(symbol)}."

    config = TailRiskConfig(
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        num_simulations=num_simulations,
        seed=seed,
        var_confidence=var_confidence,
    )
    result = compute_tail_risk(close, config=config)
    return result, None


def compute_tail_risk_for_symbol(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    config: TailRiskConfig,
) -> tuple[TailRiskResult | None, str | None]:
    resolved = resolve_duckdb_path(database_path)
    return _compute_tail_risk_cached(
        symbol=normalize_symbol(symbol),
        database_path=str(resolved),
        lookback_days=int(config.lookback_days),
        horizon_days=int(config.horizon_days),
        num_simulations=int(config.num_simulations),
        seed=int(config.seed),
        var_confidence=float(config.var_confidence),
    )


def compute_end_return_percentile(end_returns: pd.Series | Any, realized_return: float) -> float | None:
    try:
        values = pd.to_numeric(pd.Series(end_returns), errors="coerce").dropna()
    except Exception:  # noqa: BLE001
        return None
    if values.empty:
        return None
    try:
        return float((values <= float(realized_return)).mean() * 100.0)
    except Exception:  # noqa: BLE001
        return None


def _run_query_safe(
    sql: str,
    *,
    params: list[Any] | None = None,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    path = resolve_duckdb_path(database_path)
    if not path.exists():
        return pd.DataFrame(), f"DuckDB database not found: {path}"
    try:
        frame = run_query(sql=sql, params=params or [], database_path=str(path))
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), _friendly_error(str(exc))
    return frame, None


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if "candles_daily" in lowered and "does not exist" in lowered:
        return "candles_daily table not found. Run `options-helper ingest candles` first."
    if "derived_daily" in lowered and "does not exist" in lowered:
        return "derived_daily table not found. Run `options-helper derived update` first."
    return message


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number

