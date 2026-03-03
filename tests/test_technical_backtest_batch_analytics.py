from __future__ import annotations

import pandas as pd
import pytest

from options_helper.technicals_backtesting.backtest.batch_analytics import (
    build_aligned_benchmark_curve,
    build_equal_weight_aggregate_curve,
    build_period_return_table,
    compute_batch_analytics,
    compute_summary_metrics,
    extract_symbol_daily_returns,
)
from options_helper.technicals_backtesting.backtest.batch_runner import SymbolBacktestOutcome


def _outcome(
    *,
    symbol: str,
    ok: bool,
    equity_curve: pd.DataFrame | None,
) -> SymbolBacktestOutcome:
    return SymbolBacktestOutcome(
        symbol=symbol,
        ok=ok,
        stats={"Return [%]": 1.0} if ok else None,
        equity_curve=equity_curve,
        trades=None,
        warnings=(),
        error=None if ok else "failed",
        stage_timings={},
    )


def test_extract_symbol_daily_returns_and_aggregate_use_active_symbol_denominator() -> None:
    outcomes = (
        _outcome(
            symbol="SPY",
            ok=True,
            equity_curve=pd.DataFrame(
                {"Equity": [100.0, 110.0, 121.0]},
                index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
            ),
        ),
        _outcome(
            symbol="QQQ",
            ok=True,
            equity_curve=pd.DataFrame(
                {"Equity": [200.0, 220.0]},
                index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
            ),
        ),
        _outcome(symbol="IWM", ok=False, equity_curve=None),
    )

    symbol_returns = extract_symbol_daily_returns(outcomes)

    assert set(symbol_returns) == {"SPY", "QQQ"}
    aggregate = build_equal_weight_aggregate_curve(symbol_returns, initial_equity=1.0)

    assert list(aggregate.index.strftime("%Y-%m-%d")) == ["2025-01-02", "2025-01-03"]
    assert aggregate.loc[pd.Timestamp("2025-01-02"), "daily_return"] == pytest.approx(0.10)
    assert aggregate.loc[pd.Timestamp("2025-01-03"), "daily_return"] == pytest.approx(0.10)
    assert aggregate.loc[pd.Timestamp("2025-01-02"), "active_symbols"] == 1
    assert aggregate.loc[pd.Timestamp("2025-01-03"), "active_symbols"] == 2
    assert aggregate.loc[pd.Timestamp("2025-01-03"), "equity"] == pytest.approx(1.21)


def test_compute_batch_analytics_handles_sparse_benchmark_edge_alignment() -> None:
    symbol_daily_returns = {
        "AAA": pd.Series(
            [0.10, -0.05, 0.02],
            index=pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
        ),
        "BBB": pd.Series(
            [0.03, 0.04, -0.01],
            index=pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]),
        ),
    }
    benchmark_close = pd.Series(
        [100.0, 101.0, 100.0, 102.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-06", "2025-01-08"]),
    )

    result = compute_batch_analytics(
        symbol_daily_returns=symbol_daily_returns,
        benchmark_close=benchmark_close,
        initial_equity=1.0,
    )

    aggregate = result.aggregate_curve
    assert list(aggregate.index.strftime("%Y-%m-%d")) == [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
    ]
    assert aggregate.loc[pd.Timestamp("2025-01-03"), "daily_return"] == pytest.approx(-0.01)
    assert aggregate.loc[pd.Timestamp("2025-01-06"), "daily_return"] == pytest.approx(0.03)
    assert aggregate.loc[pd.Timestamp("2025-01-07"), "daily_return"] == pytest.approx(-0.01)

    benchmark = result.benchmark_curve
    assert list(benchmark.index.strftime("%Y-%m-%d")) == list(aggregate.index.strftime("%Y-%m-%d"))
    assert pd.isna(benchmark.loc[pd.Timestamp("2025-01-02"), "daily_return"])
    assert benchmark.loc[pd.Timestamp("2025-01-03"), "daily_return"] == pytest.approx(0.01)
    assert benchmark.loc[pd.Timestamp("2025-01-06"), "daily_return"] == pytest.approx(-0.009900990099)
    assert pd.isna(benchmark.loc[pd.Timestamp("2025-01-07"), "daily_return"])
    assert benchmark["has_data"].tolist() == [False, True, True, False]
    assert benchmark["equity"].tolist() == pytest.approx([1.0, 1.01, 1.0, 1.0])

    monthly = result.monthly_returns
    assert monthly.to_dict(orient="records") == [
        {
            "period": "2025-01",
            "aggregate_return": pytest.approx(0.1104533),
            "benchmark_return": pytest.approx(0.0),
        }
    ]
    yearly = result.yearly_returns
    assert yearly.to_dict(orient="records") == [
        {
            "period": "2025",
            "aggregate_return": pytest.approx(0.1104533),
            "benchmark_return": pytest.approx(0.0),
        }
    ]

    summary = result.summary_metrics
    assert summary["aggregate"]["start_date"] == "2025-01-02"
    assert summary["aggregate"]["end_date"] == "2025-01-07"
    assert summary["aggregate"]["trading_days"] == 4
    assert summary["aggregate"]["total_return"] == pytest.approx(0.1104533)

    assert summary["benchmark"]["start_date"] == "2025-01-03"
    assert summary["benchmark"]["end_date"] == "2025-01-06"
    assert summary["benchmark"]["trading_days"] == 2
    assert summary["benchmark"]["total_return"] == pytest.approx(0.0)


def test_benchmark_alignment_no_overlap_produces_flat_equity_and_empty_summary_window() -> None:
    analysis_dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    benchmark_close = pd.Series(
        [200.0, 204.0],
        index=pd.to_datetime(["2025-02-01", "2025-02-03"]),
    )

    curve = build_aligned_benchmark_curve(
        benchmark_close,
        analysis_dates=analysis_dates,
        initial_equity=1.0,
    )

    assert curve["daily_return"].isna().all()
    assert curve["has_data"].tolist() == [False, False, False]
    assert curve["equity"].tolist() == pytest.approx([1.0, 1.0, 1.0])

    summary = compute_summary_metrics(curve["daily_return"], initial_equity=1.0)
    assert summary == {
        "start_date": None,
        "end_date": None,
        "trading_days": 0,
        "total_return": 0.0,
        "ending_equity": 1.0,
        "cagr": None,
        "max_drawdown": 0.0,
        "annualized_volatility": None,
        "sharpe": None,
    }


def test_equal_weight_aggregate_uses_active_denominator_with_sparse_overlap_gaps() -> None:
    symbol_daily_returns = {
        "AAA": pd.Series(
            [0.10, -0.20],
            index=pd.to_datetime(["2025-01-02", "2025-01-06"]),
        ),
        "BBB": pd.Series(
            [0.04, 0.06],
            index=pd.to_datetime(["2025-01-03", "2025-01-06"]),
        ),
        "CCC": pd.Series(
            [0.10],
            index=pd.to_datetime(["2025-01-06"]),
        ),
    }

    curve = build_equal_weight_aggregate_curve(symbol_daily_returns, initial_equity=1.0)

    assert list(curve.index.strftime("%Y-%m-%d")) == ["2025-01-02", "2025-01-03", "2025-01-06"]
    assert curve["active_symbols"].tolist() == [1, 1, 3]
    assert curve.loc[pd.Timestamp("2025-01-02"), "daily_return"] == pytest.approx(0.10)
    assert curve.loc[pd.Timestamp("2025-01-03"), "daily_return"] == pytest.approx(0.04)
    assert curve.loc[pd.Timestamp("2025-01-06"), "daily_return"] == pytest.approx((-0.20 + 0.06 + 0.10) / 3.0)
    assert curve.loc[pd.Timestamp("2025-01-06"), "equity"] == pytest.approx(1.1287466667)


def test_benchmark_alignment_includes_left_edge_return_and_trailing_gaps() -> None:
    analysis_dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"])
    benchmark_close = pd.Series(
        [100.0, 105.0, 103.0, 106.0],
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
    )

    curve = build_aligned_benchmark_curve(
        benchmark_close,
        analysis_dates=analysis_dates,
        initial_equity=1.0,
    )

    assert curve.loc[pd.Timestamp("2025-01-02"), "daily_return"] == pytest.approx(0.05)
    assert curve.loc[pd.Timestamp("2025-01-03"), "daily_return"] == pytest.approx(-0.0190476190)
    assert curve.loc[pd.Timestamp("2025-01-06"), "daily_return"] == pytest.approx(0.0291262136)
    assert pd.isna(curve.loc[pd.Timestamp("2025-01-07"), "daily_return"])
    assert curve["has_data"].tolist() == [True, True, True, False]
    assert curve["equity"].tolist() == pytest.approx([1.05, 1.03, 1.06, 1.06])


def test_period_return_table_generates_monthly_and_yearly_buckets() -> None:
    aggregate_returns = pd.Series(
        [0.10, -0.05, 0.02, 0.03],
        index=pd.to_datetime(["2025-01-31", "2025-02-03", "2025-02-28", "2026-01-02"]),
    )
    benchmark_returns = pd.Series(
        [0.01, 0.04, -0.02],
        index=pd.to_datetime(["2025-01-31", "2025-03-03", "2026-01-02"]),
    )

    monthly = build_period_return_table(
        aggregate_daily_returns=aggregate_returns,
        benchmark_daily_returns=benchmark_returns,
        frequency="M",
    )
    yearly = build_period_return_table(
        aggregate_daily_returns=aggregate_returns,
        benchmark_daily_returns=benchmark_returns,
        frequency="Y",
    )

    assert monthly.to_dict(orient="records") == [
        {
            "period": "2025-01",
            "aggregate_return": pytest.approx(0.10),
            "benchmark_return": pytest.approx(0.01),
        },
        {
            "period": "2025-02",
            "aggregate_return": pytest.approx(-0.031),
            "benchmark_return": pytest.approx(float("nan"), nan_ok=True),
        },
        {
            "period": "2025-03",
            "aggregate_return": pytest.approx(float("nan"), nan_ok=True),
            "benchmark_return": pytest.approx(0.04),
        },
        {
            "period": "2026-01",
            "aggregate_return": pytest.approx(0.03),
            "benchmark_return": pytest.approx(-0.02),
        },
    ]
    assert yearly.to_dict(orient="records") == [
        {
            "period": "2025",
            "aggregate_return": pytest.approx(0.0659),
            "benchmark_return": pytest.approx(0.0504),
        },
        {
            "period": "2026",
            "aggregate_return": pytest.approx(0.03),
            "benchmark_return": pytest.approx(-0.02),
        },
    ]
