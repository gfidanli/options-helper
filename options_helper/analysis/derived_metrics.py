from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from options_helper.analysis.chain_metrics import ChainReport


class DerivedRow(BaseModel):
    """
    A compact per-day row of derived metrics for a symbol.

    Notes:
    - This is intended to be persisted to a per-symbol CSV file (see `DerivedStore`).
    - Fields are best-effort: missing snapshot data should surface as nulls.
    """

    date: str
    spot: float
    pc_oi: float | None = None
    pc_vol: float | None = None
    call_wall: float | None = None
    put_wall: float | None = None
    gamma_peak_strike: float | None = None
    atm_iv_near: float | None = None
    em_near_pct: float | None = None
    skew_near_pp: float | None = None

    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_chain_report(cls, report: ChainReport) -> "DerivedRow":
        warnings: list[str] = []

        call_wall = report.walls_overall.calls[0].strike if report.walls_overall.calls else None
        put_wall = report.walls_overall.puts[0].strike if report.walls_overall.puts else None

        near = report.expiries[0] if report.expiries else None
        if near is None:
            warnings.append("missing_near_expiry")

        return cls(
            date=report.as_of,
            spot=report.spot,
            pc_oi=report.totals.pc_oi_ratio,
            pc_vol=report.totals.pc_volume_ratio,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_peak_strike=report.gamma.peak_strike,
            atm_iv_near=None if near is None else near.atm_iv,
            em_near_pct=None if near is None else near.expected_move_pct,
            skew_near_pp=None if near is None else near.skew_25d_pp,
            warnings=warnings + list(report.warnings),
        )


TrendDirection = Literal["up", "down", "flat"]


class DerivedMetricStat(BaseModel):
    name: str
    value: float | None = None
    percentile: float | None = None
    percentile_n: int = 0
    trend_direction: TrendDirection | None = None
    trend_n: int = 0
    trend_delta: float | None = None
    trend_delta_pct: float | None = None


class DerivedStatsReport(BaseModel):
    schema_version: int = 1
    symbol: str
    as_of: str
    window: int
    trend_window: int
    metrics: list[DerivedMetricStat] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _to_float(val: object) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _percentile_rank_last(values: pd.Series) -> float | None:
    """
    Percentile rank (0-100) for the last value in the series, using an "average rank"
    method that yields 50th percentile when all values are equal.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None

    n = int(len(values))
    if n == 1:
        return 100.0

    ranks = values.rank(method="average")
    r = float(ranks.iloc[-1])  # 1..n
    return float((r - 1.0) / (n - 1.0) * 100.0)


def compute_derived_stats(
    df: pd.DataFrame,
    *,
    symbol: str,
    as_of: str = "latest",
    window: int = 60,
    trend_window: int = 5,
    metric_columns: list[str] | None = None,
) -> DerivedStatsReport:
    """
    Compute percentile ranks (vs last N rows) and trend flags for derived metrics.

    Pure function: deterministic for a given DataFrame and inputs.
    """
    if df is None or df.empty:
        raise ValueError("empty derived data")
    if "date" not in df.columns:
        raise ValueError("derived data missing 'date' column")

    if window < 1 or trend_window < 1:
        raise ValueError("window and trend_window must be >= 1")

    data = df.copy()
    data["date"] = data["date"].astype(str)
    data = data.sort_values(["date"], ascending=True, na_position="last")

    as_of_norm = as_of.strip().lower()
    as_of_date = str(data["date"].iloc[-1]) if as_of_norm == "latest" else as_of.strip()

    if as_of_date not in set(data["date"].tolist()):
        raise ValueError(f"date not found in derived data: {as_of_date}")

    data = data[data["date"] <= as_of_date].copy()
    data = data.sort_values(["date"], ascending=True, na_position="last")

    current_rows = data[data["date"] == as_of_date]
    if current_rows.empty:
        raise ValueError(f"no row for date: {as_of_date}")
    current = current_rows.iloc[-1]

    if metric_columns is None:
        metric_columns = [c for c in data.columns if c != "date"]

    warnings: list[str] = []
    metrics: list[DerivedMetricStat] = []

    for col in metric_columns:
        if col not in data.columns:
            warnings.append(f"missing_column:{col}")
            metrics.append(DerivedMetricStat(name=col))
            continue

        series = pd.to_numeric(data[col], errors="coerce")
        value = _to_float(current.get(col))

        pct_values = series.dropna().tail(window)
        percentile = None if value is None else _percentile_rank_last(pct_values)

        trend_values = series.dropna().tail(trend_window)
        trend_n = int(len(trend_values))
        trend_direction: TrendDirection | None = None
        trend_delta = trend_delta_pct = None
        if value is not None and trend_n >= 2:
            start = float(trend_values.iloc[0])
            end = float(trend_values.iloc[-1])
            trend_delta = float(end - start)
            if start != 0:
                trend_delta_pct = float(trend_delta / abs(start) * 100.0)
            if abs(trend_delta) <= 1e-12:
                trend_direction = "flat"
            elif trend_delta > 0:
                trend_direction = "up"
            else:
                trend_direction = "down"

        metrics.append(
            DerivedMetricStat(
                name=col,
                value=value,
                percentile=percentile,
                percentile_n=int(len(pct_values)),
                trend_direction=trend_direction,
                trend_n=trend_n,
                trend_delta=trend_delta,
                trend_delta_pct=trend_delta_pct,
            )
        )

    return DerivedStatsReport(
        symbol=symbol.upper(),
        as_of=as_of_date,
        window=window,
        trend_window=trend_window,
        metrics=metrics,
        warnings=warnings,
    )
