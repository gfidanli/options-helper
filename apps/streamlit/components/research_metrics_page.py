from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from apps.streamlit.components.duckdb_path import resolve_duckdb_path as _resolve_duckdb_path

_IV_TENOR_COLUMNS = [
    "as_of",
    "tenor_target_dte",
    "atm_iv",
    "expected_move_pct",
    "straddle_mark",
    "contracts_used",
    "updated_at",
]
_IV_DELTA_COLUMNS = [
    "as_of",
    "tenor_target_dte",
    "option_type",
    "delta_bucket",
    "avg_iv",
    "median_iv",
    "n_contracts",
    "updated_at",
]
_EXPOSURE_COLUMNS = [
    "as_of",
    "strike",
    "call_oi",
    "put_oi",
    "call_gex",
    "put_gex",
    "net_gex",
    "expiry_rows",
]
_INTRADAY_STRIKE_COLUMNS = [
    "market_date",
    "strike",
    "option_type",
    "delta_bucket",
    "contract_rows",
    "trade_count",
    "buy_notional",
    "sell_notional",
    "net_notional",
]
_INTRADAY_CONTRACT_COLUMNS = [
    "market_date",
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "source_rows",
    "trade_count",
    "buy_notional",
    "sell_notional",
    "net_notional",
    "avg_unknown_trade_share",
    "avg_quote_coverage_pct",
]


def resolve_duckdb_path(database_path: str | Path | None = None) -> Path:
    return _resolve_duckdb_path(database_path)


def normalize_symbol(value: Any, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def load_iv_surface_tenor_history(
    symbol: str,
    *,
    limit_dates: int = 90,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_days AS (
          SELECT DISTINCT as_of
          FROM iv_surface_tenor
          WHERE UPPER(symbol) = ?
          ORDER BY as_of DESC
          LIMIT ?
        ),
        ranked AS (
          SELECT
            as_of,
            tenor_target_dte,
            atm_iv,
            expected_move_pct,
            straddle_mark,
            contracts_used,
            updated_at,
            ROW_NUMBER() OVER (
              PARTITION BY as_of, tenor_target_dte
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM iv_surface_tenor
          WHERE UPPER(symbol) = ?
            AND as_of IN (SELECT as_of FROM latest_days)
        )
        SELECT
          as_of,
          tenor_target_dte,
          atm_iv,
          expected_move_pct,
          straddle_mark,
          contracts_used,
          updated_at
        FROM ranked
        WHERE row_num = 1
        ORDER BY as_of ASC, tenor_target_dte ASC
        """,
        params=[sym, max(1, int(limit_dates)), sym],
        database_path=database_path,
    )
    if note:
        return _empty_iv_tenor(), note
    if df.empty:
        return _empty_iv_tenor(), None

    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
    for column in ("tenor_target_dte", "contracts_used", "atm_iv", "expected_move_pct", "straddle_mark"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["as_of", "tenor_target_dte"])
    out = out.sort_values(by=["as_of", "tenor_target_dte"], kind="stable").reset_index(drop=True)
    return out.reindex(columns=_IV_TENOR_COLUMNS), None


def load_iv_surface_delta_history(
    symbol: str,
    *,
    limit_dates: int = 90,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_days AS (
          SELECT DISTINCT as_of
          FROM iv_surface_delta_buckets
          WHERE UPPER(symbol) = ?
          ORDER BY as_of DESC
          LIMIT ?
        ),
        ranked AS (
          SELECT
            as_of,
            tenor_target_dte,
            option_type,
            delta_bucket,
            avg_iv,
            median_iv,
            n_contracts,
            updated_at,
            ROW_NUMBER() OVER (
              PARTITION BY as_of, tenor_target_dte, option_type, delta_bucket
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM iv_surface_delta_buckets
          WHERE UPPER(symbol) = ?
            AND as_of IN (SELECT as_of FROM latest_days)
        )
        SELECT
          as_of,
          tenor_target_dte,
          option_type,
          delta_bucket,
          avg_iv,
          median_iv,
          n_contracts,
          updated_at
        FROM ranked
        WHERE row_num = 1
        ORDER BY as_of ASC, tenor_target_dte ASC, option_type ASC, delta_bucket ASC
        """,
        params=[sym, max(1, int(limit_dates)), sym],
        database_path=database_path,
    )
    if note:
        return _empty_iv_delta(), note
    if df.empty:
        return _empty_iv_delta(), None

    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    out["updated_at"] = pd.to_datetime(out["updated_at"], errors="coerce")
    for column in ("tenor_target_dte", "avg_iv", "median_iv", "n_contracts"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["option_type"] = out["option_type"].astype("string").str.lower()
    out["delta_bucket"] = out["delta_bucket"].astype("string")
    out = out.dropna(subset=["as_of", "tenor_target_dte"])
    out = out.sort_values(
        by=["as_of", "tenor_target_dte", "option_type", "delta_bucket"],
        kind="stable",
    ).reset_index(drop=True)
    return out.reindex(columns=_IV_DELTA_COLUMNS), None


def build_iv_tenor_sparkline(tenor_history_df: pd.DataFrame) -> pd.DataFrame:
    if tenor_history_df.empty:
        return pd.DataFrame()

    temp = tenor_history_df.copy()
    temp["as_of"] = pd.to_datetime(temp["as_of"], errors="coerce")
    temp["tenor_target_dte"] = pd.to_numeric(temp["tenor_target_dte"], errors="coerce")
    temp["atm_iv"] = pd.to_numeric(temp["atm_iv"], errors="coerce")
    temp = temp.dropna(subset=["as_of", "tenor_target_dte"])
    if temp.empty:
        return pd.DataFrame()

    temp["tenor_label"] = temp["tenor_target_dte"].map(lambda value: f"{int(value)}d")
    spark = temp.pivot_table(index="as_of", columns="tenor_label", values="atm_iv", aggfunc="mean")
    if spark.empty:
        return pd.DataFrame()

    ordered_labels = sorted(spark.columns, key=_tenor_sort_key)
    return spark.sort_index(kind="stable").reindex(columns=ordered_labels)


def build_iv_delta_bucket_sparkline(
    delta_history_df: pd.DataFrame,
    *,
    max_series: int = 8,
) -> pd.DataFrame:
    if delta_history_df.empty:
        return pd.DataFrame()

    temp = delta_history_df.copy()
    temp["as_of"] = pd.to_datetime(temp["as_of"], errors="coerce")
    temp["tenor_target_dte"] = pd.to_numeric(temp["tenor_target_dte"], errors="coerce")
    temp["avg_iv"] = pd.to_numeric(temp["avg_iv"], errors="coerce")
    temp["n_contracts"] = pd.to_numeric(temp["n_contracts"], errors="coerce").fillna(0.0)
    temp = temp.dropna(subset=["as_of", "tenor_target_dte"])
    if temp.empty:
        return pd.DataFrame()

    temp["option_type"] = temp["option_type"].astype("string").str.lower()
    temp["delta_bucket"] = temp["delta_bucket"].astype("string").str.lower()
    temp["series_label"] = temp.apply(_iv_delta_label, axis=1)

    latest_as_of = temp["as_of"].max()
    latest_rows = temp[temp["as_of"] == latest_as_of]
    labels = (
        latest_rows.sort_values(
            by=["n_contracts", "tenor_target_dte", "option_type", "delta_bucket"],
            ascending=[False, True, True, True],
            kind="stable",
        )["series_label"]
        .drop_duplicates()
        .head(max(1, int(max_series)))
        .tolist()
    )
    if not labels:
        labels = temp["series_label"].drop_duplicates().head(max(1, int(max_series))).tolist()
    if not labels:
        return pd.DataFrame()

    filtered = temp[temp["series_label"].isin(labels)]
    spark = filtered.pivot_table(index="as_of", columns="series_label", values="avg_iv", aggfunc="mean")
    if spark.empty:
        return pd.DataFrame()

    ordered_labels = [label for label in labels if label in spark.columns]
    return spark.sort_index(kind="stable").reindex(columns=ordered_labels)


def build_latest_iv_tenor_table(tenor_history_df: pd.DataFrame) -> pd.DataFrame:
    if tenor_history_df.empty:
        return _empty_iv_tenor()
    latest_as_of = pd.to_datetime(tenor_history_df["as_of"], errors="coerce").max()
    if pd.isna(latest_as_of):
        return _empty_iv_tenor()
    out = tenor_history_df[pd.to_datetime(tenor_history_df["as_of"], errors="coerce") == latest_as_of].copy()
    if out.empty:
        return _empty_iv_tenor()
    return out.sort_values(by=["tenor_target_dte"], kind="stable").reset_index(drop=True)


def build_latest_iv_delta_table(
    delta_history_df: pd.DataFrame,
    *,
    top_n: int = 60,
) -> pd.DataFrame:
    if delta_history_df.empty:
        return _empty_iv_delta()
    latest_as_of = pd.to_datetime(delta_history_df["as_of"], errors="coerce").max()
    if pd.isna(latest_as_of):
        return _empty_iv_delta()
    out = delta_history_df[pd.to_datetime(delta_history_df["as_of"], errors="coerce") == latest_as_of].copy()
    if out.empty:
        return _empty_iv_delta()
    out["n_contracts"] = pd.to_numeric(out["n_contracts"], errors="coerce")
    out = out.sort_values(
        by=["tenor_target_dte", "option_type", "delta_bucket", "n_contracts"],
        ascending=[True, True, True, False],
        kind="stable",
    )
    return out.head(max(1, int(top_n))).reset_index(drop=True)


def load_exposure_by_strike_latest(
    symbol: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_day AS (
          SELECT MAX(as_of) AS as_of
          FROM dealer_exposure_strikes
          WHERE UPPER(symbol) = ?
        ),
        ranked AS (
          SELECT
            as_of,
            expiry,
            strike,
            call_oi,
            put_oi,
            call_gex,
            put_gex,
            net_gex,
            ROW_NUMBER() OVER (
              PARTITION BY as_of, expiry, strike
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM dealer_exposure_strikes
          WHERE UPPER(symbol) = ?
            AND as_of = (SELECT as_of FROM latest_day)
        )
        SELECT
          as_of,
          strike,
          SUM(COALESCE(call_oi, 0.0)) AS call_oi,
          SUM(COALESCE(put_oi, 0.0)) AS put_oi,
          SUM(COALESCE(call_gex, 0.0)) AS call_gex,
          SUM(COALESCE(put_gex, 0.0)) AS put_gex,
          SUM(COALESCE(net_gex, 0.0)) AS net_gex,
          COUNT(*)::BIGINT AS expiry_rows
        FROM ranked
        WHERE row_num = 1
        GROUP BY as_of, strike
        ORDER BY strike ASC
        """,
        params=[sym, sym],
        database_path=database_path,
    )
    if note:
        return _empty_exposure(), note
    if df.empty:
        return _empty_exposure(), None

    out = df.copy()
    out["as_of"] = pd.to_datetime(out["as_of"], errors="coerce")
    for column in ("strike", "call_oi", "put_oi", "call_gex", "put_gex", "net_gex", "expiry_rows"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["as_of", "strike"])
    out = out.sort_values(by=["strike"], kind="stable").reset_index(drop=True)
    return out.reindex(columns=_EXPOSURE_COLUMNS), None


def compute_exposure_flip_strike(exposure_by_strike_df: pd.DataFrame) -> float | None:
    if exposure_by_strike_df.empty:
        return None
    temp = exposure_by_strike_df.copy()
    temp["strike"] = pd.to_numeric(temp["strike"], errors="coerce")
    temp["net_gex"] = pd.to_numeric(temp["net_gex"], errors="coerce").fillna(0.0)
    temp = temp.dropna(subset=["strike"]).sort_values(by="strike", kind="stable").reset_index(drop=True)
    if len(temp) < 2:
        return None

    temp["cum_net_gex"] = temp["net_gex"].cumsum()
    strikes = [float(value) for value in temp["strike"].tolist()]
    cumulative = [float(value) for value in temp["cum_net_gex"].tolist()]

    for strike, value in zip(strikes, cumulative, strict=False):
        if value == 0.0 and isfinite(strike):
            return strike

    for index in range(1, len(cumulative)):
        prev_value = cumulative[index - 1]
        cur_value = cumulative[index]
        prev_strike = strikes[index - 1]
        cur_strike = strikes[index]
        if not (isfinite(prev_value) and isfinite(cur_value) and isfinite(prev_strike) and isfinite(cur_strike)):
            continue
        if prev_value == 0.0:
            return prev_strike
        if cur_value == 0.0:
            return cur_strike
        if (prev_value < 0.0 < cur_value) or (prev_value > 0.0 > cur_value):
            span = abs(prev_value) + abs(cur_value)
            if span <= 0.0:
                return cur_strike
            weight = abs(prev_value) / span
            return float(prev_strike + ((cur_strike - prev_strike) * weight))
    return None


def build_exposure_top_strikes(exposure_by_strike_df: pd.DataFrame, *, top_n: int = 20) -> pd.DataFrame:
    if exposure_by_strike_df.empty:
        return _empty_exposure()
    out = exposure_by_strike_df.copy()
    out["net_gex"] = pd.to_numeric(out["net_gex"], errors="coerce")
    out["abs_net_gex"] = out["net_gex"].abs()
    out = out.sort_values(by=["abs_net_gex", "strike"], ascending=[False, True], kind="stable")
    return out.head(max(1, int(top_n))).reset_index(drop=True)


def load_intraday_flow_summary(
    symbol: str,
    *,
    database_path: str | Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_day AS (
          SELECT MAX(market_date) AS market_date
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
        ),
        ranked AS (
          SELECT
            market_date,
            source,
            contract_symbol,
            buy_volume,
            sell_volume,
            unknown_volume,
            buy_notional,
            sell_notional,
            net_notional,
            trade_count,
            unknown_trade_share,
            quote_coverage_pct,
            ROW_NUMBER() OVER (
              PARTITION BY market_date, source, contract_symbol
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
            AND market_date = (SELECT market_date FROM latest_day)
        )
        SELECT
          market_date,
          COUNT(*)::BIGINT AS contracts,
          COUNT(DISTINCT source)::BIGINT AS sources,
          SUM(COALESCE(trade_count, 0))::BIGINT AS trade_count,
          SUM(COALESCE(buy_volume, 0.0)) AS buy_volume,
          SUM(COALESCE(sell_volume, 0.0)) AS sell_volume,
          SUM(COALESCE(unknown_volume, 0.0)) AS unknown_volume,
          SUM(COALESCE(buy_notional, 0.0)) AS buy_notional,
          SUM(COALESCE(sell_notional, 0.0)) AS sell_notional,
          SUM(COALESCE(net_notional, 0.0)) AS net_notional,
          AVG(unknown_trade_share) AS avg_unknown_trade_share,
          AVG(quote_coverage_pct) AS avg_quote_coverage_pct
        FROM ranked
        WHERE row_num = 1
        GROUP BY market_date
        """,
        params=[sym, sym],
        database_path=database_path,
    )
    if note:
        return None, note
    if df.empty:
        return None, None

    row = df.iloc[0].to_dict()
    return {
        "market_date": _date_to_iso(row.get("market_date")),
        "contracts": _safe_int(row.get("contracts")) or 0,
        "sources": _safe_int(row.get("sources")) or 0,
        "trade_count": _safe_int(row.get("trade_count")) or 0,
        "buy_volume": _safe_float(row.get("buy_volume")),
        "sell_volume": _safe_float(row.get("sell_volume")),
        "unknown_volume": _safe_float(row.get("unknown_volume")),
        "buy_notional": _safe_float(row.get("buy_notional")),
        "sell_notional": _safe_float(row.get("sell_notional")),
        "net_notional": _safe_float(row.get("net_notional")),
        "avg_unknown_trade_share": _safe_float(row.get("avg_unknown_trade_share")),
        "avg_quote_coverage_pct": _safe_float(row.get("avg_quote_coverage_pct")),
    }, None


def load_intraday_flow_top_strikes(
    symbol: str,
    *,
    top_n: int = 15,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_day AS (
          SELECT MAX(market_date) AS market_date
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
        ),
        ranked AS (
          SELECT
            market_date,
            source,
            contract_symbol,
            strike,
            option_type,
            delta_bucket,
            buy_notional,
            sell_notional,
            net_notional,
            trade_count,
            ROW_NUMBER() OVER (
              PARTITION BY market_date, source, contract_symbol
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
            AND market_date = (SELECT market_date FROM latest_day)
        )
        SELECT
          market_date,
          strike,
          option_type,
          delta_bucket,
          COUNT(*)::BIGINT AS contract_rows,
          SUM(COALESCE(trade_count, 0))::BIGINT AS trade_count,
          SUM(COALESCE(buy_notional, 0.0)) AS buy_notional,
          SUM(COALESCE(sell_notional, 0.0)) AS sell_notional,
          SUM(COALESCE(net_notional, 0.0)) AS net_notional
        FROM ranked
        WHERE row_num = 1
          AND strike IS NOT NULL
        GROUP BY market_date, strike, option_type, delta_bucket
        ORDER BY ABS(SUM(COALESCE(net_notional, 0.0))) DESC, strike ASC, option_type ASC
        LIMIT ?
        """,
        params=[sym, sym, max(1, int(top_n))],
        database_path=database_path,
    )
    if note:
        return _empty_intraday_strikes(), note
    if df.empty:
        return _empty_intraday_strikes(), None

    out = df.copy()
    out["market_date"] = pd.to_datetime(out["market_date"], errors="coerce")
    for column in ("strike", "contract_rows", "trade_count", "buy_notional", "sell_notional", "net_notional"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["option_type"] = out["option_type"].astype("string").str.lower()
    out["delta_bucket"] = out["delta_bucket"].astype("string").str.lower()
    out = out.dropna(subset=["market_date", "strike"])
    out = out.reset_index(drop=True)
    return out.reindex(columns=_INTRADAY_STRIKE_COLUMNS), None


def load_intraday_flow_top_contracts(
    symbol: str,
    *,
    top_n: int = 20,
    database_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str | None]:
    sym = normalize_symbol(symbol)
    df, note = _run_query_safe(
        """
        WITH latest_day AS (
          SELECT MAX(market_date) AS market_date
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
        ),
        ranked AS (
          SELECT
            market_date,
            source,
            contract_symbol,
            expiry,
            option_type,
            strike,
            buy_notional,
            sell_notional,
            net_notional,
            trade_count,
            unknown_trade_share,
            quote_coverage_pct,
            ROW_NUMBER() OVER (
              PARTITION BY market_date, source, contract_symbol
              ORDER BY updated_at DESC, provider ASC
            ) AS row_num
          FROM intraday_option_flow
          WHERE UPPER(symbol) = ?
            AND market_date = (SELECT market_date FROM latest_day)
        )
        SELECT
          market_date,
          contract_symbol,
          expiry,
          option_type,
          strike,
          COUNT(*)::BIGINT AS source_rows,
          SUM(COALESCE(trade_count, 0))::BIGINT AS trade_count,
          SUM(COALESCE(buy_notional, 0.0)) AS buy_notional,
          SUM(COALESCE(sell_notional, 0.0)) AS sell_notional,
          SUM(COALESCE(net_notional, 0.0)) AS net_notional,
          AVG(unknown_trade_share) AS avg_unknown_trade_share,
          AVG(quote_coverage_pct) AS avg_quote_coverage_pct
        FROM ranked
        WHERE row_num = 1
        GROUP BY market_date, contract_symbol, expiry, option_type, strike
        ORDER BY ABS(SUM(COALESCE(net_notional, 0.0))) DESC, contract_symbol ASC
        LIMIT ?
        """,
        params=[sym, sym, max(1, int(top_n))],
        database_path=database_path,
    )
    if note:
        return _empty_intraday_contracts(), note
    if df.empty:
        return _empty_intraday_contracts(), None

    out = df.copy()
    out["market_date"] = pd.to_datetime(out["market_date"], errors="coerce")
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce")
    for column in (
        "strike",
        "source_rows",
        "trade_count",
        "buy_notional",
        "sell_notional",
        "net_notional",
        "avg_unknown_trade_share",
        "avg_quote_coverage_pct",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["option_type"] = out["option_type"].astype("string").str.lower()
    out = out.dropna(subset=["market_date"])
    out = out.reset_index(drop=True)
    return out.reindex(columns=_INTRADAY_CONTRACT_COLUMNS), None


def _empty_iv_tenor() -> pd.DataFrame:
    return pd.DataFrame(columns=_IV_TENOR_COLUMNS)


def _empty_iv_delta() -> pd.DataFrame:
    return pd.DataFrame(columns=_IV_DELTA_COLUMNS)


def _empty_exposure() -> pd.DataFrame:
    return pd.DataFrame(columns=_EXPOSURE_COLUMNS)


def _empty_intraday_strikes() -> pd.DataFrame:
    return pd.DataFrame(columns=_INTRADAY_STRIKE_COLUMNS)


def _empty_intraday_contracts() -> pd.DataFrame:
    return pd.DataFrame(columns=_INTRADAY_CONTRACT_COLUMNS)


def _tenor_sort_key(value: Any) -> tuple[int, str]:
    text = str(value or "").strip().lower()
    if text.endswith("d"):
        text = text[:-1]
    try:
        return (int(text), "")
    except ValueError:
        return (10**9, str(value))


def _iv_delta_label(row: pd.Series) -> str:
    try:
        tenor = int(float(row.get("tenor_target_dte")))
    except (TypeError, ValueError):
        tenor = 0
    option_type = str(row.get("option_type") or "").strip().lower()
    delta_bucket = str(row.get("delta_bucket") or "").strip().lower()
    return f"{tenor}d {option_type} {delta_bucket}".strip()


def _date_to_iso(value: Any) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


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


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
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
    if "iv_surface_tenor" in lowered and "does not exist" in lowered:
        return "iv_surface_tenor table not found. Run `options-helper market-analysis iv-surface --persist` first."
    if "iv_surface_delta_buckets" in lowered and "does not exist" in lowered:
        return (
            "iv_surface_delta_buckets table not found. "
            "Run `options-helper market-analysis iv-surface --persist` first."
        )
    if "dealer_exposure_strikes" in lowered and "does not exist" in lowered:
        return (
            "dealer_exposure_strikes table not found. "
            "Run `options-helper market-analysis exposure --persist` first."
        )
    if "intraday_option_flow" in lowered and "does not exist" in lowered:
        return "intraday_option_flow table not found. Run `options-helper intraday flow --persist` first."
    return message
