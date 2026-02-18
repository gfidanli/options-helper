from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from dagster import Failure, MaterializeResult

import options_helper.cli_deps as cli_deps
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS
from options_helper.pipelines.visibility_jobs import VisibilityJobParameterError

from .resources import DagsterPaths


@dataclass(frozen=True)
class CandlesDailyResultSummary:
    ok_items: list[Any]
    empty_items: list[Any]
    failed_items: list[Any]
    latest_data_date: date | None


def candles_daily_run_args(*, partition_key: str, paths: DagsterPaths) -> dict[str, Any]:
    return {
        "partition_key": partition_key,
        "watchlists_path": str(paths.watchlists_path),
        "watchlist": list(DEFAULT_WATCHLISTS),
        "candle_cache_dir": str(paths.data_dir / "candles"),
    }


def run_candles_daily_job(
    *,
    paths: DagsterPaths,
    partition_key: str,
    run_logger: Any,
    run_job: Any,
):
    try:
        return run_job(
            watchlists_path=paths.watchlists_path,
            watchlist=list(DEFAULT_WATCHLISTS),
            symbol=[],
            candle_cache_dir=paths.data_dir / "candles",
            provider_builder=cli_deps.build_provider,
            candle_store_builder=cli_deps.build_candle_store,
            run_logger=run_logger,
        )
    except VisibilityJobParameterError as exc:
        run_logger.log_asset_failure(
            asset_key="candles_daily",
            asset_kind="table",
            partition_key=partition_key,
            extra={"error": str(exc)},
        )
        raise Failure(str(exc)) from exc


def summarize_candles_daily_result(result: Any) -> CandlesDailyResultSummary:
    ok_items = [item for item in result.results if item.status == "ok"]
    empty_items = [item for item in result.results if item.status == "empty"]
    failed_items = [item for item in result.results if item.status == "error"]
    latest_data_date = max(
        (item.last_date for item in ok_items if item.last_date is not None),
        default=None,
    )
    return CandlesDailyResultSummary(
        ok_items=ok_items,
        empty_items=empty_items,
        failed_items=failed_items,
        latest_data_date=latest_data_date,
    )


def _candles_no_symbols_result(*, run_logger: Any, partition_key: str) -> MaterializeResult:
    run_logger.log_asset_skipped(
        asset_key="candles_daily",
        asset_kind="table",
        partition_key=partition_key,
        extra={"reason": "no_symbols"},
    )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "status": "skipped",
            "reason": "no_symbols",
        }
    )


def _raise_if_all_candles_failed(
    *,
    run_logger: Any,
    partition_key: str,
    result: Any,
    summary: CandlesDailyResultSummary,
) -> None:
    if not summary.failed_items or summary.ok_items:
        return
    run_logger.log_asset_failure(
        asset_key="candles_daily",
        asset_kind="table",
        partition_key=partition_key,
        extra={
            "warnings": result.warnings,
            "failed_symbols": [item.symbol for item in summary.failed_items],
        },
    )
    raise Failure(f"candle ingest failed for all symbols in partition {partition_key}")


def _log_candles_daily_result(
    *,
    run_logger: Any,
    partition_key: str,
    result: Any,
    summary: CandlesDailyResultSummary,
) -> None:
    if summary.ok_items:
        run_logger.log_asset_success(
            asset_key="candles_daily",
            asset_kind="table",
            partition_key=partition_key,
            rows_inserted=len(summary.ok_items),
            min_event_ts=summary.latest_data_date,
            max_event_ts=summary.latest_data_date,
            extra={
                "warnings": result.warnings,
                "empty_count": len(summary.empty_items),
                "failed_count": len(summary.failed_items),
            },
        )
        return
    run_logger.log_asset_skipped(
        asset_key="candles_daily",
        asset_kind="table",
        partition_key=partition_key,
        extra={
            "reason": "no_success_rows",
            "warnings": result.warnings,
        },
    )


def _upsert_candles_daily_watermarks(
    *,
    run_logger: Any,
    summary: CandlesDailyResultSummary,
) -> None:
    for item in summary.ok_items:
        if item.last_date is None:
            continue
        run_logger.upsert_watermark(
            asset_key="candles_daily",
            scope_key=item.symbol,
            watermark_ts=item.last_date,
        )
    if summary.latest_data_date is not None:
        run_logger.upsert_watermark(
            asset_key="candles_daily",
            scope_key="ALL",
            watermark_ts=summary.latest_data_date,
        )


def record_candles_daily_result(
    *,
    run_logger: Any,
    partition_key: str,
    result: Any,
) -> MaterializeResult:
    if result.no_symbols:
        return _candles_no_symbols_result(run_logger=run_logger, partition_key=partition_key)
    summary = summarize_candles_daily_result(result)
    _raise_if_all_candles_failed(
        run_logger=run_logger,
        partition_key=partition_key,
        result=result,
        summary=summary,
    )
    _log_candles_daily_result(
        run_logger=run_logger,
        partition_key=partition_key,
        result=result,
        summary=summary,
    )
    _upsert_candles_daily_watermarks(run_logger=run_logger, summary=summary)
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "ok_count": len(summary.ok_items),
            "empty_count": len(summary.empty_items),
            "failed_count": len(summary.failed_items),
        }
    )


def options_flow_run_args(*, partition_key: str, paths: DagsterPaths) -> dict[str, Any]:
    return {
        "partition_key": partition_key,
        "portfolio_path": str(paths.portfolio_path),
        "watchlists_path": str(paths.watchlists_path),
        "cache_dir": str(paths.data_dir / "options_snapshots"),
    }


def run_options_flow_job(
    *,
    paths: DagsterPaths,
    partition_key: str,
    run_logger: Any,
    run_job: Any,
    portfolio_loader: Any,
):
    try:
        return run_job(
            portfolio_path=paths.portfolio_path,
            symbol=None,
            watchlists_path=paths.watchlists_path,
            watchlist=[],
            all_watchlists=False,
            cache_dir=paths.data_dir / "options_snapshots",
            window=1,
            group_by="contract",
            top=10,
            out=paths.data_dir / "reports" / "dagster",
            strict=False,
            snapshot_store_builder=cli_deps.build_snapshot_store,
            flow_store_builder=cli_deps.build_flow_store,
            portfolio_loader=portfolio_loader,
            run_logger=run_logger,
        )
    except VisibilityJobParameterError as exc:
        run_logger.log_asset_failure(
            asset_key="options_flow",
            asset_kind="table",
            partition_key=partition_key,
            extra={"error": str(exc)},
        )
        raise Failure(str(exc)) from exc


def _options_flow_no_symbols_result(*, run_logger: Any, partition_key: str) -> MaterializeResult:
    run_logger.log_asset_skipped(
        asset_key="options_flow",
        asset_kind="table",
        partition_key=partition_key,
        extra={"reason": "no_symbols"},
    )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "status": "skipped",
            "reason": "no_symbols",
        }
    )


def _record_options_flow_success(
    *,
    run_logger: Any,
    partition_key: str,
    success_by_symbol: dict[str, date | None],
    skipped_symbols: set[str],
) -> date | None:
    latest_success = max((d for d in success_by_symbol.values() if d is not None), default=None)
    run_logger.log_asset_success(
        asset_key="options_flow",
        asset_kind="table",
        partition_key=partition_key,
        rows_inserted=len(success_by_symbol),
        min_event_ts=latest_success,
        max_event_ts=latest_success,
        extra={"skipped_symbols": sorted(skipped_symbols)},
    )
    for symbol, flow_end_date in sorted(success_by_symbol.items()):
        if flow_end_date is None:
            continue
        run_logger.upsert_watermark(
            asset_key="options_flow",
            scope_key=symbol,
            watermark_ts=flow_end_date,
        )
    if latest_success is not None:
        run_logger.upsert_watermark(
            asset_key="options_flow",
            scope_key="ALL",
            watermark_ts=latest_success,
        )
    return latest_success


def record_options_flow_result(
    *,
    run_logger: Any,
    partition_key: str,
    result: Any,
    success_by_symbol: dict[str, date | None],
    skipped_symbols: set[str],
) -> MaterializeResult:
    if result.no_symbols:
        return _options_flow_no_symbols_result(run_logger=run_logger, partition_key=partition_key)
    success_count = len(success_by_symbol)
    if success_count > 0:
        _record_options_flow_success(
            run_logger=run_logger,
            partition_key=partition_key,
            success_by_symbol=success_by_symbol,
            skipped_symbols=skipped_symbols,
        )
    elif skipped_symbols:
        run_logger.log_asset_skipped(
            asset_key="options_flow",
            asset_kind="table",
            partition_key=partition_key,
            extra={
                "reason": "insufficient_snapshots",
                "symbols": sorted(skipped_symbols),
            },
        )
    else:
        run_logger.log_asset_success(
            asset_key="options_flow",
            asset_kind="table",
            partition_key=partition_key,
            extra={"reason": "no_flow_rows"},
        )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "success_symbols": success_count,
            "skipped_symbols": len(skipped_symbols),
        }
    )
