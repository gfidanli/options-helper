from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from options_helper.reporting.technical_backtest_html import render_technical_backtest_html
from options_helper.schemas.technical_backtest_batch import (
    TechnicalBacktestBatchSummaryArtifact,
    validate_technical_backtest_batch_summary_payload,
)

_PER_SYMBOL_FIELDS: tuple[str, ...] = (
    "symbol",
    "period_start",
    "period_end",
    "starting_equity",
    "ending_equity",
    "total_return_pct",
    "annualized_return_pct",
    "max_drawdown_pct",
    "trade_count",
    "win_rate",
    "profit_factor",
    "invested_pct",
)
_EQUITY_FIELDS: tuple[str, ...] = (
    "session_date",
    "aggregate_equity",
    "benchmark_equity",
    "aggregate_drawdown_pct",
    "benchmark_drawdown_pct",
)
_MONTHLY_FIELDS: tuple[str, ...] = ("year", "month", "aggregate_return_pct", "benchmark_return_pct")
_YEARLY_FIELDS: tuple[str, ...] = ("year", "aggregate_return_pct", "benchmark_return_pct")
_FAILED_FIELDS: tuple[str, ...] = ("symbol", "stage", "error")


@dataclass(frozen=True)
class TechnicalBacktestBatchArtifactPaths:
    run_dir: Path
    summary_json: Path
    report_html: Path
    per_symbol_metrics_csv: Path
    equity_curve_csv: Path
    monthly_returns_csv: Path
    yearly_returns_csv: Path
    failed_symbols_csv: Path


def _coerce_summary(
    payload: TechnicalBacktestBatchSummaryArtifact | Mapping[str, Any],
) -> TechnicalBacktestBatchSummaryArtifact:
    if isinstance(payload, TechnicalBacktestBatchSummaryArtifact):
        return payload
    return validate_technical_backtest_batch_summary_payload(payload)


def build_technical_backtest_batch_artifact_paths(
    reports_dir: Path,
    *,
    run_id: str,
) -> TechnicalBacktestBatchArtifactPaths:
    run_dir = reports_dir / run_id
    return TechnicalBacktestBatchArtifactPaths(
        run_dir=run_dir,
        summary_json=run_dir / "summary.json",
        report_html=run_dir / "report.html",
        per_symbol_metrics_csv=run_dir / "per_symbol_metrics.csv",
        equity_curve_csv=run_dir / "equity_curve.csv",
        monthly_returns_csv=run_dir / "monthly_returns.csv",
        yearly_returns_csv=run_dir / "yearly_returns.csv",
        failed_symbols_csv=run_dir / "failed_symbols.csv",
    )


def _write_csv_rows(path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({str(name): row.get(name) for name in fieldnames})


def _per_symbol_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.per_symbol_metrics:
        metrics = item.metrics
        rows.append(
            {
                "symbol": item.symbol,
                "period_start": item.period_start.isoformat(),
                "period_end": item.period_end.isoformat(),
                "starting_equity": metrics.starting_equity,
                "ending_equity": metrics.ending_equity,
                "total_return_pct": metrics.total_return_pct,
                "annualized_return_pct": metrics.annualized_return_pct,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "trade_count": metrics.trade_count,
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "invested_pct": metrics.invested_pct,
            }
        )
    return rows


def _equity_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in summary.equity_curve:
        rows.append(
            {
                "session_date": item.session_date.isoformat(),
                "aggregate_equity": item.aggregate_equity,
                "benchmark_equity": item.benchmark_equity,
                "aggregate_drawdown_pct": item.aggregate_drawdown_pct,
                "benchmark_drawdown_pct": item.benchmark_drawdown_pct,
            }
        )
    return rows


def _monthly_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[dict[str, Any]]:
    return [
        {
            "year": item.year,
            "month": item.month,
            "aggregate_return_pct": item.aggregate_return_pct,
            "benchmark_return_pct": item.benchmark_return_pct,
        }
        for item in summary.monthly_returns
    ]


def _yearly_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[dict[str, Any]]:
    return [
        {
            "year": item.year,
            "aggregate_return_pct": item.aggregate_return_pct,
            "benchmark_return_pct": item.benchmark_return_pct,
        }
        for item in summary.yearly_returns
    ]


def _failed_rows(summary: TechnicalBacktestBatchSummaryArtifact) -> list[dict[str, Any]]:
    return [
        {
            "symbol": item.symbol,
            "stage": item.stage,
            "error": item.error,
        }
        for item in summary.failed_symbols
    ]


def write_technical_backtest_batch_artifacts(
    *,
    reports_dir: Path,
    summary_payload: TechnicalBacktestBatchSummaryArtifact | Mapping[str, Any],
) -> TechnicalBacktestBatchArtifactPaths:
    summary = _coerce_summary(summary_payload)
    paths = build_technical_backtest_batch_artifact_paths(reports_dir, run_id=summary.run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)

    paths.summary_json.write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    _write_csv_rows(
        paths.per_symbol_metrics_csv,
        fieldnames=_PER_SYMBOL_FIELDS,
        rows=_per_symbol_rows(summary),
    )
    _write_csv_rows(
        paths.equity_curve_csv,
        fieldnames=_EQUITY_FIELDS,
        rows=_equity_rows(summary),
    )
    _write_csv_rows(
        paths.monthly_returns_csv,
        fieldnames=_MONTHLY_FIELDS,
        rows=_monthly_rows(summary),
    )
    _write_csv_rows(
        paths.yearly_returns_csv,
        fieldnames=_YEARLY_FIELDS,
        rows=_yearly_rows(summary),
    )
    _write_csv_rows(
        paths.failed_symbols_csv,
        fieldnames=_FAILED_FIELDS,
        rows=_failed_rows(summary),
    )
    paths.report_html.write_text(
        render_technical_backtest_html(summary),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "TechnicalBacktestBatchArtifactPaths",
    "build_technical_backtest_batch_artifact_paths",
    "write_technical_backtest_batch_artifacts",
]
