from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from dagster import Failure, MaterializeResult

import options_helper.cli_deps as cli_deps
from options_helper.pipelines.visibility_jobs import VisibilityJobExecutionError

from .resources import DagsterPaths


@dataclass(frozen=True)
class DerivedMetricsRunSummary:
    successful_symbols: list[str]
    skipped_symbols: list[str]
    failed_symbols: list[str]
    bytes_written: int


def derived_metrics_run_args(
    *,
    partition_key: str,
    symbols: list[str],
    paths: DagsterPaths,
) -> dict[str, Any]:
    return {
        "partition_key": partition_key,
        "symbols": symbols,
        "cache_dir": str(paths.data_dir / "options_snapshots"),
        "derived_dir": str(paths.data_dir / "derived"),
        "candle_cache_dir": str(paths.data_dir / "candles"),
    }


def derived_metrics_no_symbols_result(
    *,
    run_logger: Any,
    partition_key: str,
    symbol_warnings: list[str],
) -> MaterializeResult:
    run_logger.log_asset_skipped(
        asset_key="derived_metrics",
        asset_kind="table",
        partition_key=partition_key,
        extra={"reason": "no_symbols", "warnings": symbol_warnings},
    )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "status": "skipped",
            "reason": "no_symbols",
        }
    )


def run_derived_metrics_symbols(
    *,
    symbols: list[str],
    partition_key: str,
    paths: DagsterPaths,
    run_logger: Any,
    run_job: Any,
) -> DerivedMetricsRunSummary:
    successful_symbols: list[str] = []
    skipped_symbols: list[str] = []
    failed_symbols: list[str] = []
    bytes_written = 0
    for sym in symbols:
        try:
            result = run_job(
                symbol=sym,
                as_of=partition_key,
                cache_dir=paths.data_dir / "options_snapshots",
                derived_dir=paths.data_dir / "derived",
                candle_cache_dir=paths.data_dir / "candles",
                snapshot_store_builder=cli_deps.build_snapshot_store,
                derived_store_builder=cli_deps.build_derived_store,
                candle_store_builder=cli_deps.build_candle_store,
                run_logger=run_logger,
            )
        except VisibilityJobExecutionError:
            skipped_symbols.append(sym)
            continue
        except Exception:  # noqa: BLE001
            failed_symbols.append(sym)
            continue
        successful_symbols.append(result.symbol.upper())
        if result.output_path.exists():
            bytes_written += result.output_path.stat().st_size
    return DerivedMetricsRunSummary(
        successful_symbols=successful_symbols,
        skipped_symbols=skipped_symbols,
        failed_symbols=failed_symbols,
        bytes_written=bytes_written,
    )


def _record_derived_metrics_success(
    *,
    run_logger: Any,
    partition_key: str,
    partition_day: date,
    summary: DerivedMetricsRunSummary,
    symbol_warnings: list[str],
) -> None:
    run_logger.log_asset_success(
        asset_key="derived_metrics",
        asset_kind="table",
        partition_key=partition_key,
        rows_inserted=len(summary.successful_symbols),
        bytes_written=summary.bytes_written,
        min_event_ts=partition_day,
        max_event_ts=partition_day,
        extra={
            "successful_symbols": summary.successful_symbols,
            "skipped_symbols": summary.skipped_symbols,
            "failed_symbols": summary.failed_symbols,
            "warnings": symbol_warnings,
        },
    )
    for symbol in summary.successful_symbols:
        run_logger.upsert_watermark(
            asset_key="derived_metrics",
            scope_key=symbol,
            watermark_ts=partition_day,
        )
    run_logger.upsert_watermark(
        asset_key="derived_metrics",
        scope_key="ALL",
        watermark_ts=partition_day,
    )


def record_derived_metrics_result(
    *,
    run_logger: Any,
    partition_key: str,
    partition_day: date,
    summary: DerivedMetricsRunSummary,
    symbol_warnings: list[str],
) -> MaterializeResult:
    if summary.failed_symbols and not summary.successful_symbols:
        run_logger.log_asset_failure(
            asset_key="derived_metrics",
            asset_kind="table",
            partition_key=partition_key,
            extra={
                "failed_symbols": summary.failed_symbols,
                "skipped_symbols": summary.skipped_symbols,
                "warnings": symbol_warnings,
            },
        )
        raise Failure(f"derived stage failed for all symbols in partition {partition_key}")
    if summary.successful_symbols:
        _record_derived_metrics_success(
            run_logger=run_logger,
            partition_key=partition_key,
            partition_day=partition_day,
            summary=summary,
            symbol_warnings=symbol_warnings,
        )
    else:
        run_logger.log_asset_skipped(
            asset_key="derived_metrics",
            asset_kind="table",
            partition_key=partition_key,
            extra={
                "reason": "no_symbols_succeeded",
                "skipped_symbols": summary.skipped_symbols,
                "warnings": symbol_warnings,
            },
        )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "successful_symbols": len(summary.successful_symbols),
            "skipped_symbols": len(summary.skipped_symbols),
            "failed_symbols": len(summary.failed_symbols),
        }
    )
