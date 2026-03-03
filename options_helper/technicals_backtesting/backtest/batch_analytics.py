from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping, Sequence

import pandas as pd

from options_helper.technicals_backtesting.backtest.batch_runner import SymbolBacktestOutcome


PeriodFrequency = Literal["M", "Y"]


@dataclass(frozen=True)
class BatchAnalyticsResult:
    aggregate_curve: pd.DataFrame
    benchmark_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    yearly_returns: pd.DataFrame
    summary_metrics: dict[str, dict[str, float | int | str | None]]


def _normalize_daily_index(index: pd.Index | pd.DatetimeIndex) -> pd.DatetimeIndex:
    parsed = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    parsed = parsed[~parsed.isna()]
    if parsed.empty:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    normalized = parsed.normalize()
    return normalized.sort_values().unique()


def _coerce_daily_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")
    values = pd.to_numeric(series, errors="coerce")
    parsed_index = pd.DatetimeIndex(pd.to_datetime(series.index, errors="coerce"))
    frame = pd.DataFrame({"value": values}, index=parsed_index)
    frame = frame.loc[~frame.index.isna()]
    frame = frame.dropna(subset=["value"])
    if frame.empty:
        return pd.Series(dtype="float64")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    deduped = frame.groupby(level=0).last().sort_index()
    return deduped["value"].astype("float64")


def _coerce_daily_returns(series: pd.Series) -> pd.Series:
    cleaned = _coerce_daily_series(series)
    if cleaned.empty:
        return cleaned
    sanitized = cleaned.replace([math.inf, -math.inf], float("nan"))
    return sanitized.astype("float64")


def daily_returns_from_equity_curve(
    equity_curve: pd.DataFrame,
    *,
    equity_col: str = "Equity",
) -> pd.Series:
    if equity_col not in equity_curve.columns:
        raise ValueError(f"Equity curve is missing required column '{equity_col}'.")
    equity = _coerce_daily_series(equity_curve[equity_col])
    if equity.empty:
        return pd.Series(dtype="float64")
    returns = equity.pct_change().replace([math.inf, -math.inf], float("nan"))
    returns = returns.dropna().astype("float64")
    returns.name = "daily_return"
    return returns


def extract_symbol_daily_returns(
    outcomes: Sequence[SymbolBacktestOutcome],
    *,
    equity_col: str = "Equity",
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for outcome in outcomes:
        if not outcome.ok or outcome.equity_curve is None:
            continue
        daily_returns = daily_returns_from_equity_curve(
            outcome.equity_curve,
            equity_col=equity_col,
        )
        if daily_returns.empty:
            continue
        out[str(outcome.symbol).upper()] = daily_returns
    return out


def build_equal_weight_aggregate_curve(
    symbol_daily_returns: Mapping[str, pd.Series],
    *,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be > 0.")
    aligned_inputs: dict[str, pd.Series] = {}
    for symbol, returns in symbol_daily_returns.items():
        cleaned = _coerce_daily_returns(returns)
        if cleaned.empty:
            continue
        aligned_inputs[str(symbol).upper()] = cleaned
    if not aligned_inputs:
        raise ValueError("symbol_daily_returns must include at least one non-empty return series.")

    merged = pd.concat(aligned_inputs, axis=1).sort_index()
    active_symbols = merged.notna().sum(axis=1).astype("int64")
    daily_return = merged.mean(axis=1, skipna=True).astype("float64")
    equity = float(initial_equity) * (1.0 + daily_return.fillna(0.0)).cumprod()

    result = pd.DataFrame(
        {
            "daily_return": daily_return,
            "equity": equity.astype("float64"),
            "active_symbols": active_symbols,
        }
    )
    result.index.name = "date"
    return result


def _resolve_benchmark_close_series(benchmark_close: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(benchmark_close, pd.DataFrame):
        if "Close" not in benchmark_close.columns:
            raise ValueError("benchmark_close dataframe must include a 'Close' column.")
        raw = benchmark_close["Close"]
    else:
        raw = benchmark_close
    return _coerce_daily_series(raw)


def build_aligned_benchmark_curve(
    benchmark_close: pd.Series | pd.DataFrame,
    *,
    analysis_dates: pd.Index | pd.DatetimeIndex,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be > 0.")
    analysis_index = _normalize_daily_index(analysis_dates)
    if analysis_index.empty:
        return pd.DataFrame(columns=["daily_return", "equity", "has_data"])

    close = _resolve_benchmark_close_series(benchmark_close)
    returns = close.pct_change().replace([math.inf, -math.inf], float("nan"))
    windowed = returns.loc[(returns.index >= analysis_index[0]) & (returns.index <= analysis_index[-1])]
    aligned_returns = windowed.reindex(analysis_index)
    has_data = aligned_returns.notna()
    equity = float(initial_equity) * (1.0 + aligned_returns.fillna(0.0)).cumprod()

    result = pd.DataFrame(
        {
            "daily_return": aligned_returns.astype("float64"),
            "equity": equity.astype("float64"),
            "has_data": has_data,
        },
        index=analysis_index,
    )
    result.index.name = "date"
    return result


def _compound_returns_by_period(daily_returns: pd.Series, *, frequency: PeriodFrequency) -> pd.Series:
    valid = _coerce_daily_returns(daily_returns).dropna()
    if valid.empty:
        return pd.Series(dtype="float64")
    grouped = (1.0 + valid).groupby(valid.index.to_period(frequency)).prod() - 1.0
    return grouped.astype("float64")


def build_period_return_table(
    *,
    aggregate_daily_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    frequency: PeriodFrequency,
) -> pd.DataFrame:
    aggregate_period = _compound_returns_by_period(aggregate_daily_returns, frequency=frequency)
    benchmark_period = _compound_returns_by_period(benchmark_daily_returns, frequency=frequency)
    if aggregate_period.empty and benchmark_period.empty:
        return pd.DataFrame(columns=["period", "aggregate_return", "benchmark_return"])

    periods = aggregate_period.index.union(benchmark_period.index).sort_values()
    frame = pd.DataFrame(
        {
            "aggregate_return": aggregate_period.reindex(periods),
            "benchmark_return": benchmark_period.reindex(periods),
        }
    )
    if frequency == "M":
        labels = [f"{period.year:04d}-{period.month:02d}" for period in periods]
    else:
        labels = [f"{period.year:04d}" for period in periods]
    frame.insert(0, "period", labels)
    return frame.reset_index(drop=True)


def compute_summary_metrics(
    daily_returns: pd.Series,
    *,
    initial_equity: float = 1.0,
) -> dict[str, float | int | str | None]:
    if initial_equity <= 0.0:
        raise ValueError("initial_equity must be > 0.")

    valid_returns = _coerce_daily_returns(daily_returns).dropna()
    if valid_returns.empty:
        return {
            "start_date": None,
            "end_date": None,
            "trading_days": 0,
            "total_return": 0.0,
            "ending_equity": float(initial_equity),
            "cagr": None,
            "max_drawdown": 0.0,
            "annualized_volatility": None,
            "sharpe": None,
        }

    equity = float(initial_equity) * (1.0 + valid_returns).cumprod()
    ending_equity = float(equity.iloc[-1])
    total_return = (ending_equity / float(initial_equity)) - 1.0
    day_span = int((valid_returns.index[-1] - valid_returns.index[0]).days)
    cagr: float | None = None
    if day_span > 0:
        years = day_span / 365.25
        if years > 0.0:
            cagr = (ending_equity / float(initial_equity)) ** (1.0 / years) - 1.0

    drawdown = (equity / equity.cummax()) - 1.0
    volatility: float | None = None
    sharpe: float | None = None
    if len(valid_returns) > 1:
        vol = float(valid_returns.std(ddof=0))
        volatility = vol * math.sqrt(252.0)
        if vol > 0.0:
            sharpe = float(valid_returns.mean()) / vol * math.sqrt(252.0)

    return {
        "start_date": valid_returns.index[0].date().isoformat(),
        "end_date": valid_returns.index[-1].date().isoformat(),
        "trading_days": int(len(valid_returns)),
        "total_return": float(total_return),
        "ending_equity": ending_equity,
        "cagr": None if cagr is None else float(cagr),
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": None if volatility is None else float(volatility),
        "sharpe": None if sharpe is None else float(sharpe),
    }


def compute_batch_analytics(
    *,
    symbol_daily_returns: Mapping[str, pd.Series],
    benchmark_close: pd.Series | pd.DataFrame,
    initial_equity: float = 1.0,
) -> BatchAnalyticsResult:
    aggregate_curve = build_equal_weight_aggregate_curve(
        symbol_daily_returns,
        initial_equity=initial_equity,
    )
    benchmark_curve = build_aligned_benchmark_curve(
        benchmark_close,
        analysis_dates=aggregate_curve.index,
        initial_equity=initial_equity,
    )
    monthly_returns = build_period_return_table(
        aggregate_daily_returns=aggregate_curve["daily_return"],
        benchmark_daily_returns=benchmark_curve["daily_return"],
        frequency="M",
    )
    yearly_returns = build_period_return_table(
        aggregate_daily_returns=aggregate_curve["daily_return"],
        benchmark_daily_returns=benchmark_curve["daily_return"],
        frequency="Y",
    )
    summary_metrics = {
        "aggregate": compute_summary_metrics(
            aggregate_curve["daily_return"],
            initial_equity=initial_equity,
        ),
        "benchmark": compute_summary_metrics(
            benchmark_curve["daily_return"],
            initial_equity=initial_equity,
        ),
    }
    return BatchAnalyticsResult(
        aggregate_curve=aggregate_curve,
        benchmark_curve=benchmark_curve,
        monthly_returns=monthly_returns,
        yearly_returns=yearly_returns,
        summary_metrics=summary_metrics,
    )


__all__ = [
    "BatchAnalyticsResult",
    "build_aligned_benchmark_curve",
    "build_equal_weight_aggregate_curve",
    "build_period_return_table",
    "compute_batch_analytics",
    "compute_summary_metrics",
    "daily_returns_from_equity_curve",
    "extract_symbol_daily_returns",
]
