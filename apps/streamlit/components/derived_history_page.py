from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_DUCKDB_PATH = Path("data/options_helper.duckdb")
_DERIVED_COLUMNS = [
    "date",
    "spot",
    "pc_oi",
    "pc_vol",
    "call_wall",
    "put_wall",
    "gamma_peak_strike",
    "atm_iv_near",
    "em_near_pct",
    "skew_near_pp",
    "rv_20d",
    "rv_60d",
    "iv_rv_20d",
    "atm_iv_near_percentile",
    "iv_term_slope",
]


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    raw = "" if database_path is None else str(database_path).strip()
    candidate = DEFAULT_DUCKDB_PATH if not raw else Path(raw)
    return candidate.expanduser().resolve()


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def list_derived_symbols(*, database_path: str | Path | None = None) -> tuple[list[str], str | None]:
    df, note = _run_query_safe(
        """
        SELECT DISTINCT UPPER(symbol) AS symbol
        FROM derived_daily
        WHERE symbol IS NOT NULL AND TRIM(symbol) <> ''
        ORDER BY symbol ASC
        """,
        database_path=database_path,
    )
    if note:
        return [], note
    if df.empty:
        return [], None
    symbols = [normalize_symbol(value) for value in df["symbol"].tolist()]
    return sorted(symbol for symbol in symbols if symbol), None


def load_derived_history(
    symbol: str,
    *,
    database_path: str | Path | None = None,
    limit: int = 2000,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        f"""
        SELECT {", ".join(_DERIVED_COLUMNS)}
        FROM derived_daily
        WHERE UPPER(symbol) = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        params=[sym, max(1, int(limit))],
        database_path=database_path,
    )
    if note:
        return _empty_derived(), note
    if df.empty:
        return _empty_derived(), None

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    for column in _DERIVED_COLUMNS:
        if column != "date" and column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.sort_values(by="date", kind="stable").reset_index(drop=True)
    return out, None


def slice_derived_history_window(derived_df: pd.DataFrame, *, window_days: int) -> pd.DataFrame:
    if derived_df.empty:
        return _empty_derived()

    window = max(1, int(window_days))
    temp = derived_df.copy()
    temp["date"] = pd.to_datetime(temp["date"], errors="coerce")
    temp = temp.dropna(subset=["date"])
    if temp.empty:
        return _empty_derived()

    latest_date = temp["date"].max()
    cutoff = latest_date - pd.to_timedelta(window - 1, unit="D")
    out = temp[temp["date"] >= cutoff]
    return out.sort_values(by="date", kind="stable").reset_index(drop=True)


def build_derived_latest_summary(derived_df: pd.DataFrame) -> dict[str, Any] | None:
    if derived_df.empty:
        return None

    latest = derived_df.iloc[-1]
    previous = derived_df.iloc[-2] if len(derived_df) > 1 else None

    latest_spot = _safe_float(latest.get("spot"))
    previous_spot = _safe_float(None if previous is None else previous.get("spot"))
    spot_change_1d = None
    if latest_spot is not None and previous_spot not in (None, 0.0):
        spot_change_1d = (latest_spot - previous_spot) / previous_spot

    latest_iv = _safe_float(latest.get("atm_iv_near"))
    previous_iv = _safe_float(None if previous is None else previous.get("atm_iv_near"))
    iv_change_1d = None
    if latest_iv is not None and previous_iv is not None:
        iv_change_1d = latest_iv - previous_iv

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
        "sample_rows": int(len(derived_df)),
    }


def _empty_derived() -> pd.DataFrame:
    return pd.DataFrame(columns=_DERIVED_COLUMNS)


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
        return pd.DataFrame(), _friendly_error(str(exc))


def _friendly_error(message: str) -> str:
    lowered = message.lower()
    if "derived_daily" in lowered and "does not exist" in lowered:
        return "derived_daily table not found. Run `options-helper derived update` first."
    return message


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


def _coerce_date(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
