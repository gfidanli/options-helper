from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_DUCKDB_PATH = Path("data/options_helper.duckdb")


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    raw = "" if database_path is None else str(database_path).strip()
    candidate = DEFAULT_DUCKDB_PATH if not raw else Path(raw)
    return candidate.expanduser().resolve()


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def resolve_symbol_selection(
    available_symbols: Sequence[str],
    *,
    query_symbol: str | None = None,
    user_symbol: str | None = None,
    default_symbol: str = "SPY",
) -> str:
    normalized_symbols = [normalize_symbol(symbol, default=default_symbol) for symbol in available_symbols]
    normalized_symbols = [symbol for symbol in normalized_symbols if symbol]
    desired = normalize_symbol(user_symbol or query_symbol or default_symbol, default=default_symbol)
    if desired and desired in normalized_symbols:
        return desired
    if normalized_symbols:
        return normalized_symbols[0]
    return desired


def sync_symbol_query_param(
    *,
    symbol: str | None,
    query_params: MutableMapping[str, Any],
    name: str = "symbol",
    default_symbol: str = "SPY",
) -> str:
    normalized_default = normalize_symbol(default_symbol, default=default_symbol)
    normalized = normalize_symbol(symbol, default=normalized_default)
    current = _read_query_param(query_params, name=name)
    if normalized == normalized_default:
        query_params.pop(name, None)
    elif current != normalized:
        query_params[name] = normalized
    return normalized


def list_available_symbols(
    *,
    database_path: str | Path | None = None,
) -> tuple[list[str], list[str]]:
    symbols: set[str] = set()
    notes: list[str] = []
    for table_name in ("candles_daily", "options_snapshot_headers", "derived_daily"):
        table_df, note = _run_query_safe(
            f"""
            SELECT DISTINCT UPPER(symbol) AS symbol
            FROM {table_name}
            WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
            ORDER BY symbol ASC
            """,
            database_path=database_path,
        )
        if note:
            notes.append(note)
            continue
        if table_df.empty or "symbol" not in table_df.columns:
            continue
        symbols.update(normalize_symbol(value) for value in table_df["symbol"].tolist())
    cleaned = sorted(symbol for symbol in symbols if symbol)
    return cleaned, notes


def load_candles_history(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    limit: int = 365,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH ranked AS (
          SELECT
            ts::TIMESTAMP AS ts,
            open,
            high,
            low,
            close,
            volume,
            ROW_NUMBER() OVER (
              PARTITION BY ts
              ORDER BY CAST(auto_adjust AS INTEGER) DESC, CAST(back_adjust AS INTEGER) ASC
            ) AS row_num
          FROM candles_daily
          WHERE UPPER(symbol) = ?
            AND interval = '1d'
        )
        SELECT ts, open, high, low, close, volume
        FROM ranked
        WHERE row_num = 1
        ORDER BY ts DESC
        LIMIT ?
        """,
        params=[sym, max(1, int(limit))],
        database_path=database_path,
    )
    if note:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]), note
    if df.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"]), None
    out = df.copy()
    out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
    out = out.dropna(subset=["ts"]).sort_values(by="ts", kind="stable").reset_index(drop=True)
    return out, None


def load_latest_snapshot_header(
    symbol: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        SELECT
          symbol,
          snapshot_date,
          provider,
          chain_path,
          spot,
          risk_free_rate,
          contracts,
          updated_at
        FROM options_snapshot_headers
        WHERE UPPER(symbol) = ?
        ORDER BY snapshot_date DESC, updated_at DESC
        LIMIT 1
        """,
        params=[sym],
        database_path=database_path,
    )
    if note:
        return None, note
    if df.empty:
        return None, None
    return df.iloc[0].to_dict(), None


def load_snapshot_chain(
    chain_path: str | Path,
    *,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    resolved_path = Path(str(chain_path)).expanduser().resolve()
    if not resolved_path.exists():
        return pd.DataFrame(), f"Snapshot chain file not found: {resolved_path}"

    escaped = str(resolved_path).replace("'", "''")
    df, note = _run_query_safe(
        f"SELECT * FROM read_parquet('{escaped}')",
        database_path=database_path,
    )
    if note:
        return pd.DataFrame(), note
    return df, None


def summarize_snapshot(
    *,
    snapshot_header: dict[str, Any] | None,
    chain_df: pd.DataFrame,
) -> dict[str, Any]:
    option_col = _resolve_column(chain_df, ["optionType", "option_type", "type"])
    oi_col = _resolve_column(chain_df, ["openInterest", "open_interest", "oi"])
    iv_col = _resolve_column(chain_df, ["impliedVolatility", "implied_volatility", "iv"])
    strike_col = _resolve_column(chain_df, ["strike"])
    volume_col = _resolve_column(chain_df, ["volume"])

    options = (
        pd.Series([], dtype="string")
        if option_col is None
        else chain_df[option_col].astype("string").str.lower().fillna("")
    )
    oi = pd.Series([], dtype="float64") if oi_col is None else pd.to_numeric(chain_df[oi_col], errors="coerce")
    iv = pd.Series([], dtype="float64") if iv_col is None else pd.to_numeric(chain_df[iv_col], errors="coerce")
    strikes = (
        pd.Series([], dtype="float64")
        if strike_col is None
        else pd.to_numeric(chain_df[strike_col], errors="coerce")
    )
    volume = (
        pd.Series([], dtype="float64")
        if volume_col is None
        else pd.to_numeric(chain_df[volume_col], errors="coerce")
    )

    call_mask = options.str.startswith("c")
    put_mask = options.str.startswith("p")

    call_oi = float(oi.where(call_mask).fillna(0.0).sum()) if not oi.empty else None
    put_oi = float(oi.where(put_mask).fillna(0.0).sum()) if not oi.empty else None
    total_oi = float(oi.fillna(0.0).sum()) if not oi.empty else None
    total_volume = float(volume.fillna(0.0).sum()) if not volume.empty else None
    positive_iv = iv[iv > 0] if not iv.empty else pd.Series([], dtype="float64")

    spot = _safe_float(None if snapshot_header is None else snapshot_header.get("spot"))
    atm_iv = None
    if spot is not None and not positive_iv.empty and not strikes.empty:
        temp = pd.DataFrame({"strike": strikes, "iv": iv}).dropna(subset=["strike", "iv"])
        temp = temp[temp["iv"] > 0]
        if not temp.empty:
            temp["spot_dist"] = (temp["strike"] - spot).abs()
            atm_iv = _safe_float(temp.sort_values(by=["spot_dist", "strike"], kind="stable").iloc[0]["iv"])

    return {
        "snapshot_date": None if snapshot_header is None else str(snapshot_header.get("snapshot_date") or ""),
        "provider": None if snapshot_header is None else snapshot_header.get("provider"),
        "contracts": int(len(chain_df)),
        "spot": spot,
        "total_open_interest": _none_if_nan(total_oi),
        "call_open_interest": _none_if_nan(call_oi),
        "put_open_interest": _none_if_nan(put_oi),
        "put_call_oi_ratio": None
        if call_oi in (None, 0.0) or put_oi is None
        else (put_oi / call_oi),
        "total_volume": _none_if_nan(total_volume),
        "avg_implied_volatility": None if positive_iv.empty else _safe_float(positive_iv.mean()),
        "atm_implied_volatility": atm_iv,
    }


def build_snapshot_strike_table(chain_df: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    if chain_df.empty:
        return pd.DataFrame(columns=["strike", "call_oi", "put_oi", "total_oi", "total_volume", "avg_iv"])

    strike_col = _resolve_column(chain_df, ["strike"])
    option_col = _resolve_column(chain_df, ["optionType", "option_type", "type"])
    oi_col = _resolve_column(chain_df, ["openInterest", "open_interest", "oi"])
    volume_col = _resolve_column(chain_df, ["volume"])
    iv_col = _resolve_column(chain_df, ["impliedVolatility", "implied_volatility", "iv"])
    if strike_col is None:
        return pd.DataFrame(columns=["strike", "call_oi", "put_oi", "total_oi", "total_volume", "avg_iv"])

    temp = pd.DataFrame(
        {
            "strike": pd.to_numeric(chain_df[strike_col], errors="coerce"),
            "option_type": ""
            if option_col is None
            else chain_df[option_col].astype("string").str.lower().fillna(""),
            "oi": 0.0 if oi_col is None else pd.to_numeric(chain_df[oi_col], errors="coerce").fillna(0.0),
            "volume": 0.0
            if volume_col is None
            else pd.to_numeric(chain_df[volume_col], errors="coerce").fillna(0.0),
            "iv": None if iv_col is None else pd.to_numeric(chain_df[iv_col], errors="coerce"),
        }
    )
    temp = temp.dropna(subset=["strike"])
    if temp.empty:
        return pd.DataFrame(columns=["strike", "call_oi", "put_oi", "total_oi", "total_volume", "avg_iv"])

    grouped = temp.groupby("strike", as_index=False).agg(
        call_oi=("oi", lambda s: float(s[temp.loc[s.index, "option_type"].str.startswith("c")].sum())),
        put_oi=("oi", lambda s: float(s[temp.loc[s.index, "option_type"].str.startswith("p")].sum())),
        total_oi=("oi", "sum"),
        total_volume=("volume", "sum"),
        avg_iv=("iv", lambda s: _safe_float(s[s > 0].mean()) if (s > 0).any() else None),
    )
    grouped["total_oi"] = pd.to_numeric(grouped["total_oi"], errors="coerce")
    grouped["total_volume"] = pd.to_numeric(grouped["total_volume"], errors="coerce")
    grouped = grouped.sort_values(by=["total_oi", "strike"], ascending=[False, True], kind="stable")
    return grouped.head(max(1, int(top_n))).reset_index(drop=True)


def load_derived_history(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    limit: int = 90,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        SELECT
          date,
          spot,
          pc_oi,
          pc_vol,
          atm_iv_near,
          iv_rv_20d,
          atm_iv_near_percentile,
          iv_term_slope
        FROM derived_daily
        WHERE UPPER(symbol) = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        params=[sym, max(1, int(limit))],
        database_path=database_path,
    )
    if note:
        return pd.DataFrame(), note
    if df.empty:
        return pd.DataFrame(), None
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(by="date", kind="stable").reset_index(drop=True)
    return out, None


def build_derived_snippet(derived_df: pd.DataFrame) -> dict[str, Any] | None:
    if derived_df.empty:
        return None
    latest = derived_df.iloc[-1]
    previous = derived_df.iloc[-2] if len(derived_df) > 1 else None

    latest_spot = _safe_float(latest.get("spot"))
    prev_spot = _safe_float(None if previous is None else previous.get("spot"))
    spot_change_1d = None
    if prev_spot not in (None, 0.0) and latest_spot is not None:
        spot_change_1d = (latest_spot - prev_spot) / prev_spot

    latest_iv = _safe_float(latest.get("atm_iv_near"))
    prev_iv = _safe_float(None if previous is None else previous.get("atm_iv_near"))
    iv_change_1d = None
    if latest_iv is not None and prev_iv is not None:
        iv_change_1d = latest_iv - prev_iv

    latest_date = latest.get("date")
    as_of = latest_date.date().isoformat() if hasattr(latest_date, "date") else str(latest_date)

    return {
        "as_of": as_of,
        "spot": latest_spot,
        "pc_oi": _safe_float(latest.get("pc_oi")),
        "pc_vol": _safe_float(latest.get("pc_vol")),
        "atm_iv_near": latest_iv,
        "iv_rv_20d": _safe_float(latest.get("iv_rv_20d")),
        "atm_iv_near_percentile": _safe_float(latest.get("atm_iv_near_percentile")),
        "iv_term_slope": _safe_float(latest.get("iv_term_slope")),
        "spot_change_1d": spot_change_1d,
        "atm_iv_change_1d": iv_change_1d,
    }


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
        conn = duckdb.connect(str(path), read_only=True)
        try:
            frame = conn.execute(sql, params or []).df()
        finally:
            conn.close()
        return frame, None
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), str(exc)


def _read_query_param(query_params: MutableMapping[str, Any], *, name: str) -> str | None:
    value = query_params.get(name)
    if isinstance(value, list):
        if not value:
            return None
        return str(value[0])
    if value is None:
        return None
    return str(value)


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    if df.empty:
        return None
    by_lower = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found:
            return found
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)
