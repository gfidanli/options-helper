from __future__ import annotations

from datetime import date
from typing import Any

from dagster import Failure, MaterializeResult

import options_helper.cli_deps as cli_deps
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS
from options_helper.pipelines.visibility_jobs import VisibilityJobParameterError

from .resources import DagsterPaths


def options_bars_run_args(*, partition_key: str, paths: DagsterPaths) -> dict[str, Any]:
    return {
        "partition_key": partition_key,
        "watchlists_path": str(paths.watchlists_path),
        "watchlist": list(DEFAULT_WATCHLISTS),
        "contracts_store_dir": str(paths.data_dir / "option_contracts"),
        "bars_store_dir": str(paths.data_dir / "option_bars"),
    }


def run_options_bars_job(
    *,
    paths: DagsterPaths,
    partition_day: date,
    partition_key: str,
    run_logger: Any,
    run_job: Any,
):
    try:
        return run_job(
            watchlists_path=paths.watchlists_path,
            watchlist=list(DEFAULT_WATCHLISTS),
            symbol=[],
            contracts_exp_start="2000-01-01",
            contracts_exp_end=None,
            lookback_years=10,
            page_limit=200,
            max_underlyings=None,
            max_contracts=None,
            max_expiries=None,
            contracts_max_requests_per_second=2.5,
            bars_concurrency=8,
            bars_max_requests_per_second=30.0,
            bars_write_batch_size=200,
            resume=True,
            dry_run=False,
            fail_fast=False,
            provider_builder=cli_deps.build_provider,
            contracts_store_builder=cli_deps.build_option_contracts_store,
            bars_store_builder=cli_deps.build_option_bars_store,
            contracts_store_dir=paths.data_dir / "option_contracts",
            bars_store_dir=paths.data_dir / "option_bars",
            today=partition_day,
            run_logger=run_logger,
        )
    except VisibilityJobParameterError as exc:
        run_logger.log_asset_failure(
            asset_key="options_bars",
            asset_kind="table",
            partition_key=partition_key,
            extra={"error": str(exc)},
        )
        raise Failure(str(exc)) from exc


def options_bars_skip(
    *,
    run_logger: Any,
    partition_key: str,
    reason: str,
) -> MaterializeResult:
    run_logger.log_asset_skipped(
        asset_key="options_bars",
        asset_kind="table",
        partition_key=partition_key,
        extra={"reason": reason},
    )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "status": "skipped",
            "reason": reason,
        }
    )


def finalize_options_bars_summary(
    *,
    summary: Any,
    run_logger: Any,
    partition_key: str,
    partition_day: date,
) -> None:
    if summary.error_contracts > 0 and summary.ok_contracts == 0:
        run_logger.log_asset_failure(
            asset_key="options_bars",
            asset_kind="table",
            partition_key=partition_key,
            rows_inserted=summary.bars_rows,
            extra={
                "ok_contracts": summary.ok_contracts,
                "error_contracts": summary.error_contracts,
                "skipped_contracts": summary.skipped_contracts,
            },
        )
        raise Failure(f"options bars backfill failed for partition {partition_key}")

    run_logger.log_asset_success(
        asset_key="options_bars",
        asset_kind="table",
        partition_key=partition_key,
        rows_inserted=summary.bars_rows,
        min_event_ts=partition_day,
        max_event_ts=partition_day,
        extra={
            "ok_contracts": summary.ok_contracts,
            "error_contracts": summary.error_contracts,
            "skipped_contracts": summary.skipped_contracts,
            "requests_attempted": summary.requests_attempted,
        },
    )
    if summary.ok_contracts > 0:
        run_logger.upsert_watermark(
            asset_key="options_bars",
            scope_key="ALL",
            watermark_ts=partition_day,
        )


def options_bars_materialize_result(*, partition_key: str, summary: Any) -> MaterializeResult:
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "ok_contracts": summary.ok_contracts,
            "error_contracts": summary.error_contracts,
            "bars_rows": summary.bars_rows,
        }
    )


def run_options_snapshot_file_job(
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
            cache_dir=paths.data_dir / "options_snapshots",
            candle_cache_dir=paths.data_dir / "candles",
            window_pct=1.0,
            spot_period="10d",
            require_data_date=partition_key,
            require_data_tz="America/Chicago",
            watchlists_path=paths.watchlists_path,
            watchlist=[],
            all_watchlists=False,
            all_expiries=True,
            full_chain=True,
            max_expiries=None,
            risk_free_rate=0.0,
            provider_builder=cli_deps.build_provider,
            snapshot_store_builder=cli_deps.build_snapshot_store,
            candle_store_builder=cli_deps.build_candle_store,
            portfolio_loader=portfolio_loader,
            run_logger=run_logger,
        )
    except VisibilityJobParameterError as exc:
        run_logger.log_asset_failure(
            asset_key="options_snapshot_file",
            asset_kind="file",
            partition_key=partition_key,
            extra={"error": str(exc)},
        )
        raise Failure(str(exc)) from exc


def options_snapshot_skip(
    *,
    run_logger: Any,
    partition_key: str,
    reason: str,
) -> MaterializeResult:
    run_logger.log_asset_skipped(
        asset_key="options_snapshot_file",
        asset_kind="file",
        partition_key=partition_key,
        extra={"reason": reason},
    )
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "status": "skipped",
            "reason": reason,
        }
    )


def resolve_options_snapshot_symbols(
    *,
    status_by_symbol: dict[str, str],
    dates_used: list[date],
) -> tuple[list[str], list[str], date | None]:
    success_symbols = sorted(sym for sym, status in status_by_symbol.items() if status == "success")
    failed_symbols = sorted(sym for sym, status in status_by_symbol.items() if status == "failed")
    max_data_date = max(dates_used) if dates_used else None
    return success_symbols, failed_symbols, max_data_date


def record_options_snapshot_file_result(
    *,
    run_logger: Any,
    partition_key: str,
    result: Any,
    success_symbols: list[str],
    failed_symbols: list[str],
    max_data_date: date | None,
) -> MaterializeResult:
    if failed_symbols and not success_symbols:
        run_logger.log_asset_failure(
            asset_key="options_snapshot_file",
            asset_kind="file",
            partition_key=partition_key,
            extra={
                "failed_symbols": failed_symbols,
                "warnings": [msg for msg in result.messages if "Warning" in msg],
            },
        )
        raise Failure(f"snapshot stage failed for all symbols in partition {partition_key}")

    if success_symbols:
        run_logger.log_asset_success(
            asset_key="options_snapshot_file",
            asset_kind="file",
            partition_key=partition_key,
            rows_inserted=len(success_symbols),
            min_event_ts=max_data_date,
            max_event_ts=max_data_date,
            extra={
                "success_symbols": success_symbols,
                "failed_symbols": failed_symbols,
            },
        )
    else:
        run_logger.log_asset_skipped(
            asset_key="options_snapshot_file",
            asset_kind="file",
            partition_key=partition_key,
            extra={"reason": "no_snapshots_saved"},
        )

    for symbol in success_symbols:
        if max_data_date is None:
            continue
        run_logger.upsert_watermark(
            asset_key="options_snapshot_file",
            scope_key=symbol,
            watermark_ts=max_data_date,
        )
    if max_data_date is not None:
        run_logger.upsert_watermark(
            asset_key="options_snapshot_file",
            scope_key="ALL",
            watermark_ts=max_data_date,
        )

    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "success_symbols": len(success_symbols),
            "failed_symbols": len(failed_symbols),
            "max_data_date": None if max_data_date is None else max_data_date.isoformat(),
        }
    )
