from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from options_helper.analysis.chain_metrics import ChainReport
from options_helper.analysis.volatility import realized_vol


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
    rv_20d: float | None = None
    rv_60d: float | None = None
    iv_rv_20d: float | None = None
    atm_iv_near_percentile: float | None = None
    iv_term_slope: float | None = None

    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def from_chain_report(
        cls,
        report: ChainReport,
        *,
        candles: pd.DataFrame | None = None,
        derived_history: pd.DataFrame | None = None,
    ) -> "DerivedRow":
        warnings: list[str] = []

        call_wall = report.walls_overall.calls[0].strike if report.walls_overall.calls else None
        put_wall = report.walls_overall.puts[0].strike if report.walls_overall.puts else None

        near = report.expiries[0] if report.expiries else None
        next_exp = report.expiries[1] if len(report.expiries) > 1 else None
        if near is None:
            warnings.append("missing_near_expiry")

        rv_20d = _rv_at_asof(candles, as_of=report.as_of, window=20)
        rv_60d = _rv_at_asof(candles, as_of=report.as_of, window=60)

        atm_iv_near = None if near is None else near.atm_iv
        atm_iv_next = None if next_exp is None else next_exp.atm_iv
        iv_rv_20d = None if (atm_iv_near is None or rv_20d is None or rv_20d <= 0) else atm_iv_near / rv_20d

        iv_term_slope = None
        if atm_iv_near is not None and atm_iv_next is not None:
            iv_term_slope = atm_iv_next - atm_iv_near

        atm_iv_near_percentile = _percentile_from_history(
            derived_history,
            date_str=report.as_of,
            value=atm_iv_near,
            column="atm_iv_near",
        )

        return cls(
            date=report.as_of,
            spot=report.spot,
            pc_oi=report.totals.pc_oi_ratio,
            pc_vol=report.totals.pc_volume_ratio,
            call_wall=call_wall,
            put_wall=put_wall,
            gamma_peak_strike=report.gamma.peak_strike,
            atm_iv_near=atm_iv_near,
            em_near_pct=None if near is None else near.expected_move_pct,
            skew_near_pp=None if near is None else near.skew_25d_pp,
            rv_20d=rv_20d,
            rv_60d=rv_60d,
            iv_rv_20d=iv_rv_20d,
            atm_iv_near_percentile=atm_iv_near_percentile,
            iv_term_slope=iv_term_slope,
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


def _select_close(history: pd.DataFrame | None) -> pd.Series:
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    if "Adj Close" in history.columns:
        return pd.to_numeric(history["Adj Close"], errors="coerce")
    if "Close" in history.columns:
        return pd.to_numeric(history["Close"], errors="coerce")
    return pd.Series(dtype="float64")


def _rv_at_asof(history: pd.DataFrame | None, *, as_of: str, window: int) -> float | None:
    close = _select_close(history)
    if close.empty:
        return None

    try:
        as_of_ts = pd.to_datetime(as_of, errors="coerce")
    except Exception:  # noqa: BLE001
        as_of_ts = pd.NaT

    if not pd.isna(as_of_ts):
        close = close.loc[close.index <= as_of_ts]
        if close.empty:
            return None

    rv_series = realized_vol(close, window)
    if rv_series.empty:
        return None
    rv_clean = rv_series.dropna()
    if rv_clean.empty:
        return None
    return float(rv_clean.iloc[-1])


def _percentile_from_history(
    derived_history: pd.DataFrame | None,
    *,
    date_str: str,
    value: float | None,
    column: str,
) -> float | None:
    if value is None:
        return None

    values = pd.Series(dtype="float64")
    if derived_history is not None and not derived_history.empty:
        history = derived_history.copy()
        if "date" in history.columns:
            history = history[history["date"].astype(str) != str(date_str)]
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce").dropna()

    combined = pd.concat([values, pd.Series([value])], ignore_index=True)
    return _percentile_rank_last(combined)


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
    _validate_derived_stat_windows(window=window, trend_window=trend_window)
    data, as_of_date, current = _prepare_derived_stats_inputs(df=df, as_of=as_of)
    selected_columns = metric_columns or [column for column in data.columns if column != "date"]
    warnings: list[str] = []
    metrics: list[DerivedMetricStat] = []
    for column in selected_columns:
        stat, warning = _build_metric_stat(
            data=data,
            current=current,
            column=column,
            window=window,
            trend_window=trend_window,
        )
        if warning is not None:
            warnings.append(warning)
        metrics.append(stat)
    return DerivedStatsReport(
        symbol=symbol.upper(),
        as_of=as_of_date,
        window=window,
        trend_window=trend_window,
        metrics=metrics,
        warnings=warnings,
    )


def _prepare_derived_stats_inputs(
    *,
    df: pd.DataFrame,
    as_of: str,
) -> tuple[pd.DataFrame, str, pd.Series]:
    if df is None or df.empty:
        raise ValueError("empty derived data")
    if "date" not in df.columns:
        raise ValueError("derived data missing 'date' column")
    data = df.copy()
    data["date"] = data["date"].astype(str)
    data = data.sort_values(["date"], ascending=True, na_position="last")
    as_of_date = _resolve_derived_as_of_date(data=data, as_of=as_of)
    if as_of_date not in set(data["date"].tolist()):
        raise ValueError(f"date not found in derived data: {as_of_date}")
    data = data[data["date"] <= as_of_date].copy()
    data = data.sort_values(["date"], ascending=True, na_position="last")
    current_rows = data[data["date"] == as_of_date]
    if current_rows.empty:
        raise ValueError(f"no row for date: {as_of_date}")
    return data, as_of_date, current_rows.iloc[-1]


def _resolve_derived_as_of_date(*, data: pd.DataFrame, as_of: str) -> str:
    as_of_norm = as_of.strip().lower()
    if as_of_norm == "latest":
        return str(data["date"].iloc[-1])
    return as_of.strip()


def _build_metric_stat(
    *,
    data: pd.DataFrame,
    current: pd.Series,
    column: str,
    window: int,
    trend_window: int,
) -> tuple[DerivedMetricStat, str | None]:
    if column not in data.columns:
        return DerivedMetricStat(name=column), f"missing_column:{column}"
    series = pd.to_numeric(data[column], errors="coerce")
    value = _to_float(current.get(column))
    percentile_values = series.dropna().tail(window)
    percentile = None if value is None else _percentile_rank_last(percentile_values)
    trend = _compute_trend_stats(series=series, value=value, trend_window=trend_window)
    return (
        DerivedMetricStat(
            name=column,
            value=value,
            percentile=percentile,
            percentile_n=int(len(percentile_values)),
            trend_direction=trend["trend_direction"],
            trend_n=trend["trend_n"],
            trend_delta=trend["trend_delta"],
            trend_delta_pct=trend["trend_delta_pct"],
        ),
        None,
    )


def _validate_derived_stat_windows(*, window: int, trend_window: int) -> None:
    if window < 1 or trend_window < 1:
        raise ValueError("window and trend_window must be >= 1")


def _compute_trend_stats(
    *,
    series: pd.Series,
    value: float | None,
    trend_window: int,
) -> dict[str, TrendDirection | float | int | None]:
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
    return {
        "trend_direction": trend_direction,
        "trend_n": trend_n,
        "trend_delta": trend_delta,
        "trend_delta_pct": trend_delta_pct,
    }
