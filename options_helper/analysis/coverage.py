from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

import pandas as pd


DEFAULT_OI_DELTA_LAGS = (1, 3, 5)


@dataclass(frozen=True)
class CandleCoverage:
    rows_total: int
    rows_lookback: int
    start_date: str | None
    end_date: str | None
    expected_business_days: int
    missing_business_days: int
    missing_business_dates: list[str]
    missing_value_cells: int
    missing_values_by_column: dict[str, int]


@dataclass(frozen=True)
class SnapshotCoverage:
    days_present_total: int
    days_present_lookback: int
    start_date: str | None
    end_date: str | None
    expected_business_days: int
    missing_business_days: int
    missing_business_dates: list[str]
    avg_contracts_per_day: float | None
    median_contracts_per_day: float | None
    min_contracts_per_day: float | None
    max_contracts_per_day: float | None
    non_zero_contract_days: int


@dataclass(frozen=True)
class OIDeltaCoverage:
    lag_days: int
    contracts_with_oi: int
    contracts_with_delta: int
    pair_count: int
    coverage_ratio: float


@dataclass(frozen=True)
class ContractOICoverage:
    contracts_total: int
    contracts_with_snapshots: int
    contracts_with_oi: int
    expected_contract_days: int
    observed_contract_days: int
    observed_oi_contract_days: int
    snapshot_day_coverage_ratio: float
    oi_day_coverage_ratio: float
    lookback_start_date: str | None
    lookback_end_date: str | None
    snapshot_days_present: int
    snapshot_days_missing: int
    snapshot_missing_dates: list[str]
    per_contract_oi_days_median: float | None
    per_contract_oi_days_p90: float | None
    oi_delta_coverage: list[OIDeltaCoverage]


@dataclass(frozen=True)
class OptionBarsCoverage:
    contracts_total: int
    contracts_with_rows: int
    rows_total: int
    status_counts: dict[str, int]
    start_date: str | None
    end_date: str | None
    contracts_covering_lookback_end: int
    covering_lookback_end_ratio: float


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def compute_candle_coverage(
    candles: pd.DataFrame,
    *,
    lookback_days: int,
) -> CandleCoverage:
    frame = _normalize_daily_rows(candles, date_col_candidates=("ts", "date"))
    if frame.empty:
        return CandleCoverage(
            rows_total=0,
            rows_lookback=0,
            start_date=None,
            end_date=None,
            expected_business_days=0,
            missing_business_days=0,
            missing_business_dates=[],
            missing_value_cells=0,
            missing_values_by_column={},
        )

    day_values = frame["_date"].dropna().astype("object").tolist()
    unique_days = sorted({d for d in day_values if isinstance(d, date)})
    if not unique_days:
        return CandleCoverage(
            rows_total=int(len(frame)),
            rows_lookback=0,
            start_date=None,
            end_date=None,
            expected_business_days=0,
            missing_business_days=0,
            missing_business_dates=[],
            missing_value_cells=0,
            missing_values_by_column={},
        )

    start_day = unique_days[0]
    end_day = unique_days[-1]
    expected_days = _expected_business_days(end_day=end_day, lookback_days=lookback_days, start_floor=start_day)

    day_set = set(unique_days)
    missing_days = [d for d in expected_days if d not in day_set]
    rows_lookback = int(sum(1 for d in unique_days if d in set(expected_days)))

    missing_by_column: dict[str, int] = {}
    missing_cells = 0
    for column in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if column not in frame.columns:
            continue
        count = int(pd.to_numeric(frame[column], errors="coerce").isna().sum())
        if count <= 0:
            continue
        missing_by_column[column] = count
        missing_cells += count

    return CandleCoverage(
        rows_total=int(len(frame)),
        rows_lookback=rows_lookback,
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        expected_business_days=int(len(expected_days)),
        missing_business_days=int(len(missing_days)),
        missing_business_dates=[d.isoformat() for d in missing_days],
        missing_value_cells=missing_cells,
        missing_values_by_column=missing_by_column,
    )


def compute_snapshot_coverage(
    snapshot_headers: pd.DataFrame,
    *,
    lookback_days: int,
) -> SnapshotCoverage:
    frame = _normalize_daily_rows(snapshot_headers, date_col_candidates=("snapshot_date", "as_of_date", "date"))
    if frame.empty:
        return SnapshotCoverage(
            days_present_total=0,
            days_present_lookback=0,
            start_date=None,
            end_date=None,
            expected_business_days=0,
            missing_business_days=0,
            missing_business_dates=[],
            avg_contracts_per_day=None,
            median_contracts_per_day=None,
            min_contracts_per_day=None,
            max_contracts_per_day=None,
            non_zero_contract_days=0,
        )

    day_values = frame["_date"].dropna().astype("object").tolist()
    unique_days = sorted({d for d in day_values if isinstance(d, date)})
    if not unique_days:
        return SnapshotCoverage(
            days_present_total=0,
            days_present_lookback=0,
            start_date=None,
            end_date=None,
            expected_business_days=0,
            missing_business_days=0,
            missing_business_dates=[],
            avg_contracts_per_day=None,
            median_contracts_per_day=None,
            min_contracts_per_day=None,
            max_contracts_per_day=None,
            non_zero_contract_days=0,
        )

    start_day = unique_days[0]
    end_day = unique_days[-1]
    expected_days = _expected_business_days(end_day=end_day, lookback_days=lookback_days, start_floor=start_day)
    day_set = set(unique_days)
    missing_days = [d for d in expected_days if d not in day_set]

    contract_series = (
        pd.to_numeric(frame.get("contracts"), errors="coerce")
        if "contracts" in frame.columns
        else pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")
    )
    grouped_contracts = (
        pd.DataFrame({"_date": frame["_date"], "contracts": contract_series})
        .dropna(subset=["_date"])
        .groupby("_date", sort=True)["contracts"]
        .max()
    )
    grouped_contracts = grouped_contracts.dropna()

    contracts_values = grouped_contracts.tolist()
    avg_contracts = float(pd.Series(contracts_values).mean()) if contracts_values else None
    median_contracts = float(pd.Series(contracts_values).median()) if contracts_values else None
    min_contracts = float(min(contracts_values)) if contracts_values else None
    max_contracts = float(max(contracts_values)) if contracts_values else None
    non_zero_days = int(sum(1 for val in contracts_values if float(val) > 0))

    return SnapshotCoverage(
        days_present_total=int(len(unique_days)),
        days_present_lookback=int(sum(1 for d in unique_days if d in set(expected_days))),
        start_date=start_day.isoformat(),
        end_date=end_day.isoformat(),
        expected_business_days=int(len(expected_days)),
        missing_business_days=int(len(missing_days)),
        missing_business_dates=[d.isoformat() for d in missing_days],
        avg_contracts_per_day=avg_contracts,
        median_contracts_per_day=median_contracts,
        min_contracts_per_day=min_contracts,
        max_contracts_per_day=max_contracts,
        non_zero_contract_days=non_zero_days,
    )


def compute_contract_oi_coverage(
    contracts: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    lookback_days: int,
    as_of: date | None = None,
) -> ContractOICoverage:
    contract_symbols = _normalized_contract_symbols(contracts, symbol_col_candidates=("contract_symbol", "contractSymbol"))
    total_contracts = len(contract_symbols)

    if snapshots is None or snapshots.empty:
        lookback_end = as_of
        expected_days = _expected_business_days(end_day=lookback_end, lookback_days=lookback_days)
        return _empty_contract_oi_coverage(
            total_contracts=total_contracts,
            expected_days=expected_days,
            lookback_end=lookback_end,
        )

    frame = snapshots.copy()
    symbol_col = _first_available_column(frame, ("contract_symbol", "contractSymbol"))
    day_col = _first_available_column(frame, ("as_of_date", "snapshot_date", "date"))
    if symbol_col is None or day_col is None:
        lookback_end = as_of
        expected_days = _expected_business_days(end_day=lookback_end, lookback_days=lookback_days)
        return _empty_contract_oi_coverage(
            total_contracts=total_contracts,
            expected_days=expected_days,
            lookback_end=lookback_end,
        )

    frame["_contract_symbol"] = frame[symbol_col].map(_normalize_symbol)
    frame["_as_of_date"] = _coerce_date_series(frame[day_col])
    frame = frame.dropna(subset=["_contract_symbol", "_as_of_date"]).copy()

    if contract_symbols:
        frame = frame[frame["_contract_symbol"].isin(set(contract_symbols))].copy()

    if frame.empty:
        lookback_end = as_of
        expected_days = _expected_business_days(end_day=lookback_end, lookback_days=lookback_days)
        return _empty_contract_oi_coverage(
            total_contracts=total_contracts,
            expected_days=expected_days,
            lookback_end=lookback_end,
        )

    frame = frame.drop_duplicates(subset=["_contract_symbol", "_as_of_date"], keep="last")
    frame = frame.sort_values(["_as_of_date", "_contract_symbol"], kind="stable")

    observed_end = max(v for v in frame["_as_of_date"].tolist() if isinstance(v, date))
    lookback_end = as_of or observed_end
    expected_days = _expected_business_days(end_day=lookback_end, lookback_days=lookback_days)
    expected_set = set(expected_days)

    if expected_set:
        frame = frame[frame["_as_of_date"].isin(expected_set)].copy()

    if frame.empty:
        return _empty_contract_oi_coverage(
            total_contracts=total_contracts,
            expected_days=expected_days,
            lookback_end=lookback_end,
        )

    contracts_with_snapshots = int(frame["_contract_symbol"].nunique())

    if "open_interest" in frame.columns:
        oi_series = pd.to_numeric(frame["open_interest"], errors="coerce")
    elif "openInterest" in frame.columns:
        oi_series = pd.to_numeric(frame["openInterest"], errors="coerce")
    else:
        oi_series = pd.Series([float("nan")] * len(frame), index=frame.index, dtype="float64")

    frame["_open_interest"] = oi_series
    oi_frame = frame[frame["_open_interest"].notna()].copy()

    contracts_with_oi = int(oi_frame["_contract_symbol"].nunique()) if not oi_frame.empty else 0

    expected_contract_days = int(len(expected_days) * total_contracts) if total_contracts > 0 else 0
    observed_contract_days = int(len(frame))
    observed_oi_contract_days = int(len(oi_frame))

    snapshot_ratio = (
        float(observed_contract_days / expected_contract_days)
        if expected_contract_days > 0
        else 0.0
    )
    oi_ratio = (
        float(observed_oi_contract_days / expected_contract_days)
        if expected_contract_days > 0
        else 0.0
    )

    observed_snapshot_days = sorted(
        {
            v
            for v in frame["_as_of_date"].tolist()
            if isinstance(v, date)
        }
    )
    missing_snapshot_days = [d for d in expected_days if d not in set(observed_snapshot_days)]

    if not oi_frame.empty:
        per_contract = (
            oi_frame.groupby("_contract_symbol", sort=True)["_as_of_date"]
            .nunique()
            .astype("int64")
        )
        median_oi_days = float(per_contract.median()) if not per_contract.empty else None
        p90_oi_days = float(per_contract.quantile(0.9)) if not per_contract.empty else None
    else:
        median_oi_days = None
        p90_oi_days = None

    delta_coverage = [
        _compute_oi_delta_coverage(oi_frame, lag_days=lag, contracts_with_oi=contracts_with_oi)
        for lag in DEFAULT_OI_DELTA_LAGS
    ]

    return ContractOICoverage(
        contracts_total=total_contracts,
        contracts_with_snapshots=contracts_with_snapshots,
        contracts_with_oi=contracts_with_oi,
        expected_contract_days=expected_contract_days,
        observed_contract_days=observed_contract_days,
        observed_oi_contract_days=observed_oi_contract_days,
        snapshot_day_coverage_ratio=snapshot_ratio,
        oi_day_coverage_ratio=oi_ratio,
        lookback_start_date=expected_days[0].isoformat() if expected_days else None,
        lookback_end_date=lookback_end.isoformat() if lookback_end else None,
        snapshot_days_present=int(len(observed_snapshot_days)),
        snapshot_days_missing=int(len(missing_snapshot_days)),
        snapshot_missing_dates=[d.isoformat() for d in missing_snapshot_days],
        per_contract_oi_days_median=median_oi_days,
        per_contract_oi_days_p90=p90_oi_days,
        oi_delta_coverage=delta_coverage,
    )


def compute_option_bars_coverage(
    option_bars_meta: pd.DataFrame,
    *,
    lookback_days: int,
    as_of: date | None = None,
) -> OptionBarsCoverage:
    frame = option_bars_meta.copy() if option_bars_meta is not None else pd.DataFrame()
    if frame.empty:
        return OptionBarsCoverage(
            contracts_total=0,
            contracts_with_rows=0,
            rows_total=0,
            status_counts={},
            start_date=None,
            end_date=None,
            contracts_covering_lookback_end=0,
            covering_lookback_end_ratio=0.0,
        )

    symbol_col = _first_available_column(frame, ("contract_symbol", "contractSymbol"))
    if symbol_col is None:
        return OptionBarsCoverage(
            contracts_total=0,
            contracts_with_rows=0,
            rows_total=0,
            status_counts={},
            start_date=None,
            end_date=None,
            contracts_covering_lookback_end=0,
            covering_lookback_end_ratio=0.0,
        )

    frame["_contract_symbol"] = frame[symbol_col].map(_normalize_symbol)
    frame = frame.dropna(subset=["_contract_symbol"]).copy()
    frame = frame.drop_duplicates(subset=["_contract_symbol"], keep="last")
    if frame.empty:
        return OptionBarsCoverage(
            contracts_total=0,
            contracts_with_rows=0,
            rows_total=0,
            status_counts={},
            start_date=None,
            end_date=None,
            contracts_covering_lookback_end=0,
            covering_lookback_end_ratio=0.0,
        )

    status_series = frame.get("status")
    statuses = (
        status_series.map(lambda v: str(v or "").strip().lower() or "unknown")
        if status_series is not None
        else pd.Series(["unknown"] * len(frame), index=frame.index)
    )
    status_counts: dict[str, int] = {}
    for raw, count in statuses.value_counts(dropna=False).items():
        key = str(raw or "unknown").strip().lower() or "unknown"
        status_counts[key] = int(count)

    rows_series = pd.to_numeric(frame.get("rows"), errors="coerce") if "rows" in frame.columns else pd.Series([0] * len(frame))
    rows_series = rows_series.fillna(0.0)

    start_series = _coerce_timestamp_series(frame.get("start_ts"))
    end_series = _coerce_timestamp_series(frame.get("end_ts"))

    end_day = as_of
    if end_day is None and not end_series.dropna().empty:
        end_day = end_series.dropna().max().date()

    lookback_days_set = set(_expected_business_days(end_day=end_day, lookback_days=lookback_days))
    coverage_cutoff = max(lookback_days_set) if lookback_days_set else end_day

    covering_end = 0
    if coverage_cutoff is not None:
        for ts in end_series.tolist():
            if ts is None:
                continue
            try:
                if pd.isna(ts):
                    continue
            except Exception:  # noqa: BLE001
                pass
            if ts.date() >= coverage_cutoff:
                covering_end += 1

    contracts_total = int(len(frame))
    contracts_with_rows = int((rows_series > 0).sum())
    rows_total = int(rows_series.sum())

    start_date = start_series.dropna().min().date().isoformat() if not start_series.dropna().empty else None
    end_date = end_series.dropna().max().date().isoformat() if not end_series.dropna().empty else None

    return OptionBarsCoverage(
        contracts_total=contracts_total,
        contracts_with_rows=contracts_with_rows,
        rows_total=rows_total,
        status_counts=status_counts,
        start_date=start_date,
        end_date=end_date,
        contracts_covering_lookback_end=covering_end,
        covering_lookback_end_ratio=(float(covering_end / contracts_total) if contracts_total > 0 else 0.0),
    )


def _empty_contract_oi_coverage(
    *,
    total_contracts: int,
    expected_days: list[date],
    lookback_end: date | None,
) -> ContractOICoverage:
    return ContractOICoverage(
        contracts_total=total_contracts,
        contracts_with_snapshots=0,
        contracts_with_oi=0,
        expected_contract_days=int(len(expected_days) * total_contracts) if total_contracts > 0 else 0,
        observed_contract_days=0,
        observed_oi_contract_days=0,
        snapshot_day_coverage_ratio=0.0,
        oi_day_coverage_ratio=0.0,
        lookback_start_date=expected_days[0].isoformat() if expected_days else None,
        lookback_end_date=lookback_end.isoformat() if lookback_end else None,
        snapshot_days_present=0,
        snapshot_days_missing=int(len(expected_days)),
        snapshot_missing_dates=[d.isoformat() for d in expected_days],
        per_contract_oi_days_median=None,
        per_contract_oi_days_p90=None,
        oi_delta_coverage=[
            OIDeltaCoverage(
                lag_days=lag,
                contracts_with_oi=0,
                contracts_with_delta=0,
                pair_count=0,
                coverage_ratio=0.0,
            )
            for lag in DEFAULT_OI_DELTA_LAGS
        ],
    )


def _compute_oi_delta_coverage(
    oi_frame: pd.DataFrame,
    *,
    lag_days: int,
    contracts_with_oi: int,
) -> OIDeltaCoverage:
    if oi_frame is None or oi_frame.empty or contracts_with_oi <= 0:
        return OIDeltaCoverage(
            lag_days=int(lag_days),
            contracts_with_oi=max(0, int(contracts_with_oi)),
            contracts_with_delta=0,
            pair_count=0,
            coverage_ratio=0.0,
        )

    work = oi_frame[["_contract_symbol", "_as_of_date"]].copy()
    work["_as_of_date"] = pd.to_datetime(work["_as_of_date"], errors="coerce")
    work = work.dropna(subset=["_contract_symbol", "_as_of_date"])
    if work.empty:
        return OIDeltaCoverage(
            lag_days=int(lag_days),
            contracts_with_oi=max(0, int(contracts_with_oi)),
            contracts_with_delta=0,
            pair_count=0,
            coverage_ratio=0.0,
        )

    work = work.drop_duplicates(subset=["_contract_symbol", "_as_of_date"], keep="last")
    current = work.copy()
    current["_prev_date"] = (current["_as_of_date"] - pd.offsets.BDay(int(lag_days))).dt.date

    previous = work.copy()
    previous["_prev_date"] = previous["_as_of_date"].dt.date
    merged = current.merge(
        previous,
        on=["_contract_symbol", "_prev_date"],
        how="inner",
        suffixes=("_curr", "_prev"),
    )

    contracts_with_delta = int(merged["_contract_symbol"].nunique()) if not merged.empty else 0
    pair_count = int(len(merged))
    coverage_ratio = float(contracts_with_delta / contracts_with_oi) if contracts_with_oi > 0 else 0.0

    return OIDeltaCoverage(
        lag_days=int(lag_days),
        contracts_with_oi=int(contracts_with_oi),
        contracts_with_delta=contracts_with_delta,
        pair_count=pair_count,
        coverage_ratio=coverage_ratio,
    )


def _expected_business_days(
    *,
    end_day: date | None,
    lookback_days: int,
    start_floor: date | None = None,
) -> list[date]:
    if end_day is None:
        return []
    days = max(1, int(lookback_days))
    values = [
        ts.date()
        for ts in pd.bdate_range(end=pd.Timestamp(end_day), periods=days).tolist()
    ]
    if start_floor is not None:
        values = [d for d in values if d >= start_floor]
    return values


def _normalize_daily_rows(
    frame: pd.DataFrame,
    *,
    date_col_candidates: tuple[str, ...],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    date_col = _first_available_column(out, date_col_candidates)
    if date_col is None:
        return pd.DataFrame()
    out["_date"] = _coerce_date_series(out[date_col])
    out = out.dropna(subset=["_date"]).copy()
    if out.empty:
        return pd.DataFrame()
    out = out.sort_values("_date", kind="stable")
    out = out.drop_duplicates(subset=["_date"], keep="last")
    return out.reset_index(drop=True)


def _coerce_date_series(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        try:
            parsed = parsed.dt.tz_convert(None)
        except Exception:  # noqa: BLE001
            pass
        return parsed.dt.date
    return pd.Series(dtype="object")


def _coerce_timestamp_series(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    if isinstance(parsed, pd.Series):
        try:
            parsed = parsed.dt.tz_convert(None)
        except Exception:  # noqa: BLE001
            pass
        return parsed
    return pd.Series(dtype="datetime64[ns]")


def _first_available_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _normalize_symbol(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text


def _normalized_contract_symbols(
    frame: pd.DataFrame,
    *,
    symbol_col_candidates: tuple[str, ...],
) -> list[str]:
    if frame is None or frame.empty:
        return []
    symbol_col = _first_available_column(frame, symbol_col_candidates)
    if symbol_col is None:
        return []
    cleaned = [_normalize_symbol(v) for v in frame[symbol_col].tolist()]
    return sorted({c for c in cleaned if c})
