from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from options_helper.analysis.osi import parse_contract_symbol


_CANDLE_CHECK_SPECS = [
    ("candles_unique_symbol_date", "error"),
    ("candles_monotonic_date", "error"),
    ("candles_no_negative_prices", "error"),
    ("candles_gap_days_last_30", "warn"),
]
_OPTIONS_BARS_CHECK_SPECS = [
    ("options_bars_monotonic_ts", "error"),
    ("options_bars_no_negative_prices", "error"),
    ("options_bars_duplicate_pk", "error"),
]
_SNAPSHOT_CHECK_SPECS = [("snapshot_parseable_contract_symbol", "warn")]
_FLOW_CHECK_SPECS = [("flow_no_null_primary_keys", "error")]
_DERIVED_CHECK_SPECS = [("derived_no_duplicate_keys", "error")]


@dataclass(frozen=True)
class QualityCheckResult:
    asset_key: str
    check_name: str
    severity: str
    status: str
    scope_key: str = "ALL"
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str | None = None


def _normalize_scope_key(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "ALL"


def _normalize_symbol_list(symbols: Iterable[str] | None) -> list[str]:
    if symbols is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in symbols:
        sym = str(value or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _coerce_to_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except Exception:  # noqa: BLE001
        return None


def _clean_text(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    text = str(value).strip()
    return text or None


def _skip_results(
    *,
    asset_key: str,
    specs: list[tuple[str, str]],
    reason: str,
    scope_key: str = "ALL",
    metrics: dict[str, Any] | None = None,
) -> list[QualityCheckResult]:
    payload = {"reason": reason}
    if metrics:
        payload.update(metrics)
    return [
        QualityCheckResult(
            asset_key=asset_key,
            check_name=check_name,
            severity=severity,
            status="skip",
            scope_key=scope_key,
            metrics=dict(payload),
            message=f"check skipped ({reason})",
        )
        for check_name, severity in specs
    ]


def evaluate_candle_checks_for_symbol(
    symbol: str,
    history: pd.DataFrame,
    *,
    gap_lookback_days: int = 30,
) -> list[QualityCheckResult]:
    sym = _normalize_scope_key(str(symbol).upper())
    if history is None or history.empty:
        return _skip_results(
            asset_key="candles_daily",
            specs=_CANDLE_CHECK_SPECS,
            reason="no_rows",
            scope_key=sym,
            metrics={"rows_checked": 0},
        )

    index_raw = pd.to_datetime(pd.Index(history.index), errors="coerce", utc=True)
    invalid_date_rows = int(pd.isna(index_raw).sum())
    valid_dates = index_raw[~pd.isna(index_raw)]
    valid_dates = valid_dates.tz_convert(None).normalize() if len(valid_dates) else valid_dates

    duplicate_day_rows = int(pd.Index(valid_dates).duplicated(keep=False).sum())
    date_series = pd.Series(valid_dates)
    out_of_order_pairs = (
        int((date_series.diff().dropna() < pd.Timedelta(0)).sum()) if len(date_series) > 1 else 0
    )

    price_columns = [
        col
        for col in history.columns
        if str(col).strip().lower() in {"open", "high", "low", "close"}
    ]
    if price_columns:
        prices = history[price_columns].apply(pd.to_numeric, errors="coerce")
        negative_mask = prices < 0
        negative_cells = int(negative_mask.sum().sum())
        negative_rows = int(negative_mask.any(axis=1).sum())
        negative_status = "fail" if negative_cells > 0 else "pass"
        negative_message = (
            f"{negative_cells} negative price value(s) across {negative_rows} row(s)."
            if negative_cells > 0
            else "no negative price values found"
        )
    else:
        negative_cells = 0
        negative_rows = 0
        negative_status = "skip"
        negative_message = "price columns unavailable"

    if len(valid_dates) == 0:
        gap_status = "skip"
        gap_days = 0
        missing_business_days: list[str] = []
        latest_date = None
        gap_message = "no valid candle dates"
    else:
        latest_date = valid_dates.max().date()
        window_days = max(int(gap_lookback_days), 1)
        window_start = latest_date - timedelta(days=window_days - 1)
        expected_days = [d.date() for d in pd.bdate_range(start=window_start, end=latest_date)]
        observed_days = {
            d.date()
            for d in valid_dates
            if window_start <= d.date() <= latest_date and d.weekday() < 5
        }
        missing = [day for day in expected_days if day not in observed_days]
        missing_business_days = [day.isoformat() for day in missing[:10]]
        gap_days = len(missing)
        gap_status = "fail" if gap_days > 0 else "pass"
        gap_message = (
            f"{gap_days} missing business day(s) in trailing {window_days}-day window."
            if gap_days > 0
            else f"no missing business days in trailing {window_days}-day window"
        )

    uniqueness_status = "fail" if duplicate_day_rows > 0 or invalid_date_rows > 0 else "pass"
    monotonic_status = "fail" if out_of_order_pairs > 0 or invalid_date_rows > 0 else "pass"

    return [
        QualityCheckResult(
            asset_key="candles_daily",
            check_name="candles_unique_symbol_date",
            severity="error",
            status=uniqueness_status,
            scope_key=sym,
            metrics={
                "rows_checked": int(len(history)),
                "duplicate_date_rows": duplicate_day_rows,
                "invalid_date_rows": invalid_date_rows,
            },
            message=(
                "duplicate or invalid candle dates found"
                if uniqueness_status == "fail"
                else "no duplicate candle dates found"
            ),
        ),
        QualityCheckResult(
            asset_key="candles_daily",
            check_name="candles_monotonic_date",
            severity="error",
            status=monotonic_status,
            scope_key=sym,
            metrics={
                "rows_checked": int(len(history)),
                "out_of_order_pairs": out_of_order_pairs,
                "invalid_date_rows": invalid_date_rows,
            },
            message=(
                "candle dates are not monotonic"
                if monotonic_status == "fail"
                else "candle dates are monotonic"
            ),
        ),
        QualityCheckResult(
            asset_key="candles_daily",
            check_name="candles_no_negative_prices",
            severity="error",
            status=negative_status,
            scope_key=sym,
            metrics={
                "rows_checked": int(len(history)),
                "negative_rows": negative_rows,
                "negative_cells": negative_cells,
                "price_columns": [str(col) for col in price_columns],
            },
            message=negative_message,
        ),
        QualityCheckResult(
            asset_key="candles_daily",
            check_name="candles_gap_days_last_30",
            severity="warn",
            status=gap_status,
            scope_key=sym,
            metrics={
                "rows_checked": int(len(history)),
                "latest_date": None if latest_date is None else latest_date.isoformat(),
                "gap_days": gap_days,
                "missing_business_days_sample": missing_business_days,
            },
            message=gap_message,
        ),
    ]


def run_candle_quality_checks(
    *,
    candle_store: Any | None,
    symbols: Iterable[str] | None,
    gap_lookback_days: int = 30,
    skip_reason: str | None = None,
) -> list[QualityCheckResult]:
    symbols_norm = _normalize_symbol_list(symbols)
    if skip_reason is not None:
        return _skip_results(asset_key="candles_daily", specs=_CANDLE_CHECK_SPECS, reason=skip_reason)
    if not symbols_norm:
        return _skip_results(asset_key="candles_daily", specs=_CANDLE_CHECK_SPECS, reason="no_symbols")
    if candle_store is None or not hasattr(candle_store, "load"):
        return _skip_results(
            asset_key="candles_daily",
            specs=_CANDLE_CHECK_SPECS,
            reason="store_not_queryable",
        )

    results: list[QualityCheckResult] = []
    for symbol in symbols_norm:
        history = candle_store.load(symbol)
        results.extend(
            evaluate_candle_checks_for_symbol(
                symbol,
                history,
                gap_lookback_days=gap_lookback_days,
            )
        )
    return results


def evaluate_options_bars_checks_from_frame(
    bars_frame: pd.DataFrame,
    *,
    scope_key: str = "ALL",
) -> list[QualityCheckResult]:
    frame = bars_frame if bars_frame is not None else pd.DataFrame()
    scope = _normalize_scope_key(scope_key)

    contract_col = "contract_symbol" if "contract_symbol" in frame.columns else "contractSymbol"
    interval_col = "interval"
    provider_col = "provider"
    ts_col = "ts"

    has_group_cols = all(
        col in frame.columns for col in {contract_col, interval_col, provider_col, ts_col}
    )

    if has_group_cols:
        work = pd.DataFrame(
            {
                "contract_symbol": frame[contract_col].map(_clean_text),
                "interval": frame[interval_col].map(_clean_text),
                "provider": frame[provider_col].map(_clean_text),
                "ts": pd.to_datetime(frame[ts_col], errors="coerce", utc=True),
            }
        )
        invalid_ts_rows = int(work["ts"].isna().sum())
        monotonic_groups = 0
        valid = work.dropna(subset=["contract_symbol", "interval", "provider", "ts"]).copy()
        if not valid.empty:
            valid["ts"] = valid["ts"].dt.tz_convert(None)
            for _, group in valid.groupby(["contract_symbol", "interval", "provider"], sort=False):
                if not group["ts"].is_monotonic_increasing:
                    monotonic_groups += 1
        monotonic_status = "fail" if invalid_ts_rows > 0 or monotonic_groups > 0 else "pass"
        monotonic_message = (
            "non-monotonic or invalid option bar timestamps found"
            if monotonic_status == "fail"
            else "option bar timestamps are monotonic"
        )
    else:
        invalid_ts_rows = 0
        monotonic_groups = 0
        monotonic_status = "skip"
        monotonic_message = "required grouping/timestamp columns unavailable"

    price_columns = [
        col
        for col in frame.columns
        if str(col).strip().lower() in {"open", "high", "low", "close"}
    ]
    if price_columns:
        prices = frame[price_columns].apply(pd.to_numeric, errors="coerce")
        negative_mask = prices < 0
        negative_cells = int(negative_mask.sum().sum())
        negative_rows = int(negative_mask.any(axis=1).sum())
        negative_status = "fail" if negative_cells > 0 else "pass"
        negative_message = (
            f"{negative_cells} negative price value(s) across {negative_rows} row(s)."
            if negative_cells > 0
            else "no negative price values found"
        )
    else:
        negative_cells = 0
        negative_rows = 0
        negative_status = "skip"
        negative_message = "price columns unavailable"

    if has_group_cols:
        pk_df = pd.DataFrame(
            {
                "contract_symbol": frame[contract_col].map(_clean_text),
                "interval": frame[interval_col].map(_clean_text),
                "provider": frame[provider_col].map(_clean_text),
                "ts": pd.to_datetime(frame[ts_col], errors="coerce", utc=True).dt.tz_convert(None),
            }
        )
        duplicate_mask = pk_df.duplicated(
            subset=["contract_symbol", "interval", "provider", "ts"],
            keep=False,
        )
        duplicate_pk_rows = int(duplicate_mask.sum())
        duplicate_pk_keys = int(
            pk_df.loc[duplicate_mask, ["contract_symbol", "interval", "provider", "ts"]]
            .drop_duplicates()
            .shape[0]
        )
        duplicate_status = "fail" if duplicate_pk_rows > 0 else "pass"
        duplicate_message = (
            "duplicate option_bars primary keys found"
            if duplicate_status == "fail"
            else "no duplicate option_bars primary keys found"
        )
    else:
        duplicate_pk_rows = 0
        duplicate_pk_keys = 0
        duplicate_status = "skip"
        duplicate_message = "required primary key columns unavailable"

    row_count = int(len(frame))
    return [
        QualityCheckResult(
            asset_key="options_bars",
            check_name="options_bars_monotonic_ts",
            severity="error",
            status=monotonic_status,
            scope_key=scope,
            metrics={
                "rows_checked": row_count,
                "invalid_ts_rows": invalid_ts_rows,
                "non_monotonic_groups": monotonic_groups,
            },
            message=monotonic_message,
        ),
        QualityCheckResult(
            asset_key="options_bars",
            check_name="options_bars_no_negative_prices",
            severity="error",
            status=negative_status,
            scope_key=scope,
            metrics={
                "rows_checked": row_count,
                "negative_rows": negative_rows,
                "negative_cells": negative_cells,
            },
            message=negative_message,
        ),
        QualityCheckResult(
            asset_key="options_bars",
            check_name="options_bars_duplicate_pk",
            severity="error",
            status=duplicate_status,
            scope_key=scope,
            metrics={
                "rows_checked": row_count,
                "duplicate_pk_rows": duplicate_pk_rows,
                "duplicate_pk_keys": duplicate_pk_keys,
            },
            message=duplicate_message,
        ),
    ]


def _warehouse_from_store(store: Any | None) -> Any | None:
    if store is None:
        return None
    warehouse = getattr(store, "warehouse", None)
    if warehouse is None:
        return None
    if not hasattr(warehouse, "fetch_df"):
        return None
    return warehouse


def _query_option_bars_frame(
    *,
    bars_store: Any | None,
    contract_symbols: Iterable[str] | None,
    interval: str | None,
    provider: str | None,
) -> pd.DataFrame | None:
    warehouse = _warehouse_from_store(bars_store)
    if warehouse is None:
        return None

    where_clauses: list[str] = []
    params: list[Any] = []

    symbols_norm = _normalize_symbol_list(contract_symbols)
    if symbols_norm:
        placeholders = ", ".join("?" for _ in symbols_norm)
        where_clauses.append(f"contract_symbol IN ({placeholders})")
        params.extend(symbols_norm)

    interval_text = _clean_text(interval)
    if interval_text is not None:
        where_clauses.append("interval = ?")
        params.append(interval_text.lower())

    provider_text = _clean_text(provider)
    if provider_text is not None:
        where_clauses.append("provider = ?")
        params.append(provider_text.lower())

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        "SELECT contract_symbol, interval, provider, ts, open, high, low, close "
        f"FROM option_bars {where_sql} "
        "ORDER BY contract_symbol ASC, interval ASC, provider ASC, ts ASC"
    )
    df = warehouse.fetch_df(sql, params)
    if df is None:
        return pd.DataFrame(
            columns=["contract_symbol", "interval", "provider", "ts", "open", "high", "low", "close"]
        )
    return df


def run_options_bars_quality_checks(
    *,
    bars_store: Any | None,
    contract_symbols: Iterable[str] | None,
    interval: str | None = "1d",
    provider: str | None = "alpaca",
    dry_run: bool = False,
    skip_reason: str | None = None,
) -> list[QualityCheckResult]:
    if skip_reason is not None:
        return _skip_results(asset_key="options_bars", specs=_OPTIONS_BARS_CHECK_SPECS, reason=skip_reason)
    if dry_run:
        return _skip_results(asset_key="options_bars", specs=_OPTIONS_BARS_CHECK_SPECS, reason="dry_run")

    frame = _query_option_bars_frame(
        bars_store=bars_store,
        contract_symbols=contract_symbols,
        interval=interval,
        provider=provider,
    )
    if frame is None:
        return _skip_results(
            asset_key="options_bars",
            specs=_OPTIONS_BARS_CHECK_SPECS,
            reason="store_not_queryable",
        )
    return evaluate_options_bars_checks_from_frame(frame)


def evaluate_snapshot_parseable_contract_symbol_check(
    symbol: str,
    snapshot_date: date,
    snapshot_frame: pd.DataFrame,
) -> QualityCheckResult:
    sym = _normalize_scope_key(str(symbol).upper())
    scope = f"{sym}|{snapshot_date.isoformat()}"
    frame = snapshot_frame if snapshot_frame is not None else pd.DataFrame()

    if frame.empty:
        return _skip_results(
            asset_key="options_snapshots",
            specs=_SNAPSHOT_CHECK_SPECS,
            reason="no_rows",
            scope_key=scope,
            metrics={"rows_checked": 0},
        )[0]

    if "contractSymbol" not in frame.columns:
        return QualityCheckResult(
            asset_key="options_snapshots",
            check_name="snapshot_parseable_contract_symbol",
            severity="warn",
            status="fail",
            scope_key=scope,
            metrics={
                "rows_checked": int(len(frame)),
                "total_contracts": 0,
                "parseable_contracts": 0,
                "unparseable_contracts": 0,
            },
            message="missing contractSymbol column",
        )

    symbols = [
        str(value).strip().upper()
        for value in frame["contractSymbol"].tolist()
        if _clean_text(value) is not None
    ]
    total_contracts = len(symbols)
    if total_contracts == 0:
        return _skip_results(
            asset_key="options_snapshots",
            specs=_SNAPSHOT_CHECK_SPECS,
            reason="empty_contract_symbols",
            scope_key=scope,
            metrics={"rows_checked": int(len(frame))},
        )[0]

    parseable_contracts = sum(1 for raw in symbols if parse_contract_symbol(raw) is not None)
    unparseable_contracts = total_contracts - parseable_contracts
    status = "fail" if unparseable_contracts > 0 else "pass"

    return QualityCheckResult(
        asset_key="options_snapshots",
        check_name="snapshot_parseable_contract_symbol",
        severity="warn",
        status=status,
        scope_key=scope,
        metrics={
            "rows_checked": int(len(frame)),
            "total_contracts": total_contracts,
            "parseable_contracts": parseable_contracts,
            "unparseable_contracts": unparseable_contracts,
            "unparseable_ratio": float(unparseable_contracts / total_contracts),
        },
        message=(
            f"{unparseable_contracts} contract symbol(s) were not parseable"
            if unparseable_contracts > 0
            else "all contract symbols are parseable"
        ),
    )


def run_snapshot_quality_checks(
    *,
    snapshot_store: Any | None,
    snapshot_dates_by_symbol: dict[str, date],
    skip_reason: str | None = None,
) -> list[QualityCheckResult]:
    if skip_reason is not None:
        return _skip_results(
            asset_key="options_snapshots",
            specs=_SNAPSHOT_CHECK_SPECS,
            reason=skip_reason,
        )
    if not snapshot_dates_by_symbol:
        return _skip_results(
            asset_key="options_snapshots",
            specs=_SNAPSHOT_CHECK_SPECS,
            reason="no_snapshot_dates",
        )
    if snapshot_store is None or not hasattr(snapshot_store, "load_day"):
        return _skip_results(
            asset_key="options_snapshots",
            specs=_SNAPSHOT_CHECK_SPECS,
            reason="store_not_queryable",
        )

    results: list[QualityCheckResult] = []
    for symbol, snapshot_day in sorted(snapshot_dates_by_symbol.items()):
        frame = snapshot_store.load_day(symbol, snapshot_day)
        results.append(evaluate_snapshot_parseable_contract_symbol_check(symbol, snapshot_day, frame))
    return results


def evaluate_flow_pk_null_guard_from_frame(
    flow_frame: pd.DataFrame,
    *,
    scope_key: str = "ALL",
) -> QualityCheckResult:
    frame = flow_frame if flow_frame is not None else pd.DataFrame()
    scope = _normalize_scope_key(scope_key)
    if frame.empty:
        return QualityCheckResult(
            asset_key="options_flow",
            check_name="flow_no_null_primary_keys",
            severity="error",
            status="pass",
            scope_key=scope,
            metrics={"rows_checked": 0, "null_pk_rows": 0},
            message="no persisted flow rows to validate",
        )

    symbol_col = "symbol"
    from_col = "from_date"
    to_col = "to_date"
    window_col = "window_size" if "window_size" in frame.columns else "window"
    group_col = "group_by"
    row_key_col = "row_key"

    required = [symbol_col, from_col, to_col, window_col, group_col, row_key_col]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return QualityCheckResult(
            asset_key="options_flow",
            check_name="flow_no_null_primary_keys",
            severity="error",
            status="fail",
            scope_key=scope,
            metrics={"rows_checked": int(len(frame)), "missing_columns": missing},
            message="required flow primary key columns unavailable",
        )

    symbol_values = frame[symbol_col].map(_clean_text)
    from_values = pd.to_datetime(frame[from_col], errors="coerce")
    to_values = pd.to_datetime(frame[to_col], errors="coerce")
    window_values = pd.to_numeric(frame[window_col], errors="coerce")
    group_values = frame[group_col].map(_clean_text)
    row_key_values = frame[row_key_col].map(_clean_text)

    null_mask = (
        symbol_values.isna()
        | from_values.isna()
        | to_values.isna()
        | window_values.isna()
        | group_values.isna()
        | row_key_values.isna()
    )
    null_pk_rows = int(null_mask.sum())
    status = "fail" if null_pk_rows > 0 else "pass"

    return QualityCheckResult(
        asset_key="options_flow",
        check_name="flow_no_null_primary_keys",
        severity="error",
        status=status,
        scope_key=scope,
        metrics={"rows_checked": int(len(frame)), "null_pk_rows": null_pk_rows},
        message=(
            "flow rows with null/blank primary key values found"
            if status == "fail"
            else "all flow primary key fields are populated"
        ),
    )


def _query_flow_rows_frame(
    *,
    flow_store: Any | None,
    symbols: Iterable[str] | None,
) -> pd.DataFrame | None:
    warehouse = _warehouse_from_store(flow_store)
    if warehouse is None:
        return None

    where_clauses: list[str] = []
    params: list[Any] = []

    symbols_norm = _normalize_symbol_list(symbols)
    if symbols_norm:
        placeholders = ", ".join("?" for _ in symbols_norm)
        where_clauses.append(f"symbol IN ({placeholders})")
        params.extend(symbols_norm)

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = (
        "SELECT symbol, from_date, to_date, window_size, group_by, row_key "
        f"FROM options_flow {where_sql}"
    )
    df = warehouse.fetch_df(sql, params)
    if df is None:
        return pd.DataFrame(columns=["symbol", "from_date", "to_date", "window_size", "group_by", "row_key"])
    return df


def run_flow_quality_checks(
    *,
    flow_store: Any | None,
    symbols: Iterable[str] | None,
    skip_reason: str | None = None,
) -> list[QualityCheckResult]:
    if skip_reason is not None:
        return _skip_results(asset_key="options_flow", specs=_FLOW_CHECK_SPECS, reason=skip_reason)

    frame = _query_flow_rows_frame(flow_store=flow_store, symbols=symbols)
    if frame is None:
        return _skip_results(asset_key="options_flow", specs=_FLOW_CHECK_SPECS, reason="store_not_queryable")
    return [evaluate_flow_pk_null_guard_from_frame(frame)]


def evaluate_derived_duplicate_guard_from_frame(
    symbol: str,
    derived_frame: pd.DataFrame,
) -> QualityCheckResult:
    sym = _normalize_scope_key(str(symbol).upper())
    frame = derived_frame if derived_frame is not None else pd.DataFrame()

    if frame.empty:
        return _skip_results(
            asset_key="derived_daily",
            specs=_DERIVED_CHECK_SPECS,
            reason="no_rows",
            scope_key=sym,
            metrics={"rows_checked": 0},
        )[0]

    if "date" not in frame.columns:
        return QualityCheckResult(
            asset_key="derived_daily",
            check_name="derived_no_duplicate_keys",
            severity="error",
            status="fail",
            scope_key=sym,
            metrics={"rows_checked": int(len(frame)), "missing_columns": ["date"]},
            message="required derived key column unavailable",
        )

    date_series = frame["date"].astype(str).str.strip()
    duplicate_mask = date_series.duplicated(keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_dates = int(date_series[duplicate_mask].nunique())
    status = "fail" if duplicate_rows > 0 else "pass"

    return QualityCheckResult(
        asset_key="derived_daily",
        check_name="derived_no_duplicate_keys",
        severity="error",
        status=status,
        scope_key=sym,
        metrics={
            "rows_checked": int(len(frame)),
            "duplicate_rows": duplicate_rows,
            "duplicate_dates": duplicate_dates,
        },
        message=(
            "duplicate derived rows found for symbol/date"
            if status == "fail"
            else "no duplicate derived rows found"
        ),
    )


def run_derived_quality_checks(
    *,
    derived_store: Any | None,
    symbol: str,
    skip_reason: str | None = None,
) -> list[QualityCheckResult]:
    if skip_reason is not None:
        return _skip_results(
            asset_key="derived_daily",
            specs=_DERIVED_CHECK_SPECS,
            reason=skip_reason,
            scope_key=_normalize_scope_key(str(symbol).upper()),
        )

    if derived_store is None or not hasattr(derived_store, "load"):
        return _skip_results(
            asset_key="derived_daily",
            specs=_DERIVED_CHECK_SPECS,
            reason="store_not_queryable",
            scope_key=_normalize_scope_key(str(symbol).upper()),
        )

    frame = derived_store.load(symbol)
    return [evaluate_derived_duplicate_guard_from_frame(symbol, frame)]


def persist_quality_checks(*, run_logger: Any, checks: Iterable[QualityCheckResult]) -> None:
    for check in checks:
        run_logger.log_check(
            asset_key=check.asset_key,
            check_name=check.check_name,
            severity=check.severity,
            status=check.status,
            partition_key=_normalize_scope_key(check.scope_key),
            metrics=check.metrics,
            message=check.message,
        )
