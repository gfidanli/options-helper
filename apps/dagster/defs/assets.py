from __future__ import annotations

from contextlib import contextmanager
from datetime import date
import os
import re
from typing import Any, Iterable

from dagster import AssetExecutionContext, DailyPartitionsDefinition, Failure, MaterializeResult, asset

import options_helper.cli_deps as cli_deps
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS, resolve_symbols
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.data.storage_runtime import (
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobExecutionError,
    VisibilityJobParameterError,
    run_briefing_job,
    run_derived_update_job,
    run_flow_report_job,
    run_ingest_candles_job,
    run_ingest_options_bars_job,
    run_snapshot_options_job,
)
from options_helper.storage import load_portfolio

from .resources import DagsterPaths, DagsterRuntimeConfig


_FLOW_ARROW_PATTERN = r"(?:->|\u2192)"
_FLOW_HEADER_RE = re.compile(
    (
        rf"^([A-Z0-9._-]+)\s+flow"
        rf"(?:\s+net\s+window=\d+\s+\((\d{{4}}-\d{{2}}-\d{{2}})\s+{_FLOW_ARROW_PATTERN}\s+"
        rf"(\d{{4}}-\d{{2}}-\d{{2}})\)"
        rf"|\s+(\d{{4}}-\d{{2}}-\d{{2}})\s+{_FLOW_ARROW_PATTERN}\s+(\d{{4}}-\d{{2}}-\d{{2}}))"
    )
)
_FLOW_NO_DATA_RE = re.compile(r"^No flow data for\s+([A-Z0-9._-]+):")
_SAVED_SNAPSHOT_RE = re.compile(r"^([A-Z0-9._-]+)\s+\d{4}-\d{2}-\d{2}: saved\b")
_WARNING_SYMBOL_RE = re.compile(r"warning:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)
_ERROR_SYMBOL_RE = re.compile(r"error:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)

_PARTITION_START_DATE = (
    os.environ.get("OPTIONS_HELPER_DAGSTER_PARTITION_START", "2026-01-01").strip() or "2026-01-01"
)
DAILY_PARTITIONS = DailyPartitionsDefinition(start_date=_PARTITION_START_DATE)


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _coerce_iso_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


def _partition_date(context: AssetExecutionContext) -> date:
    partition_key = str(context.partition_key or "").strip()
    parsed = _coerce_iso_date(partition_key)
    if parsed is None:
        raise Failure("Daily partition key is required (YYYY-MM-DD).")
    return parsed


@contextmanager
def _runtime_defaults(paths: DagsterPaths, runtime_config: DagsterRuntimeConfig) -> Iterable[None]:
    provider_token = set_default_provider_name(runtime_config.provider)
    storage_token = set_default_storage_backend("duckdb")
    duckdb_token = set_default_duckdb_path(paths.duckdb_path)
    try:
        yield
    finally:
        reset_default_duckdb_path(duckdb_token)
        reset_default_storage_backend(storage_token)
        reset_default_provider_name(provider_token)


def _dagster_parent_run_id(context: AssetExecutionContext) -> str | None:
    raw = str(getattr(context, "run_id", "") or "").strip()
    return raw or None


@contextmanager
def _observed_asset_run(
    context: AssetExecutionContext,
    *,
    job_name: str,
    args: dict[str, Any],
) -> Iterable[Any]:
    paths: DagsterPaths = context.resources.paths
    runtime_config: DagsterRuntimeConfig = context.resources.runtime_config
    with _runtime_defaults(paths, runtime_config):
        run_logger = cli_deps.build_run_logger(
            job_name=job_name,
            triggered_by="dagster",
            parent_run_id=_dagster_parent_run_id(context),
            provider=runtime_config.provider,
            storage_backend="duckdb",
            args=args,
        )
        try:
            yield run_logger
        except Exception as exc:  # noqa: BLE001
            run_logger.finalize_failure(exc)
            raise
        else:
            run_logger.finalize_success()


def _flow_renderable_statuses(renderables: list[object]) -> tuple[dict[str, date | None], set[str]]:
    success_by_symbol: dict[str, date | None] = {}
    skipped_symbols: set[str] = set()
    for renderable in renderables:
        if not isinstance(renderable, str):
            continue
        plain = _strip_rich_markup(renderable)
        no_data_match = _FLOW_NO_DATA_RE.match(plain)
        if no_data_match:
            sym = no_data_match.group(1).upper()
            if sym not in success_by_symbol:
                skipped_symbols.add(sym)
            continue

        header_match = _FLOW_HEADER_RE.match(plain)
        if not header_match:
            continue
        sym = header_match.group(1).upper()
        end_date = _coerce_iso_date(header_match.group(3)) or _coerce_iso_date(header_match.group(5))
        success_by_symbol[sym] = end_date
        skipped_symbols.discard(sym)
    return success_by_symbol, skipped_symbols


def _snapshot_status_by_symbol(*, symbols: list[str], messages: list[str]) -> dict[str, str]:
    status_by_symbol = {sym.upper(): "skipped" for sym in symbols}
    for message in messages:
        plain = _strip_rich_markup(message)
        saved_match = _SAVED_SNAPSHOT_RE.match(plain)
        if saved_match:
            status_by_symbol[saved_match.group(1).upper()] = "success"
            continue

        error_match = _ERROR_SYMBOL_RE.search(plain)
        if error_match:
            status_by_symbol[error_match.group(1).upper()] = "failed"
            continue

        if "skipping snapshot" not in plain.lower():
            continue
        warning_match = _WARNING_SYMBOL_RE.search(plain)
        if warning_match and status_by_symbol.get(warning_match.group(1).upper()) != "success":
            status_by_symbol[warning_match.group(1).upper()] = "skipped"
    return status_by_symbol


def _resolve_default_symbols(paths: DagsterPaths) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    symbols: set[str] = set()

    try:
        selection = resolve_symbols(
            watchlists_path=paths.watchlists_path,
            watchlists=list(DEFAULT_WATCHLISTS),
            symbols=[],
            default_watchlists=DEFAULT_WATCHLISTS,
        )
        symbols.update(selection.symbols)
        warnings.extend(selection.warnings)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"watchlist symbol resolution failed: {exc}")

    try:
        portfolio = load_portfolio(paths.portfolio_path)
        symbols.update({p.symbol.upper() for p in portfolio.positions})
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"portfolio symbol resolution failed: {exc}")

    return sorted(symbols), warnings


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    required_resource_keys={"paths", "runtime_config"},
)
def candles_daily(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths

    with _observed_asset_run(
        context,
        job_name="dagster_candles_daily",
        args={
            "partition_key": partition_key,
            "watchlists_path": str(paths.watchlists_path),
            "watchlist": list(DEFAULT_WATCHLISTS),
            "candle_cache_dir": str(paths.data_dir / "candles"),
        },
    ) as run_logger:
        try:
            result = run_ingest_candles_job(
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

        ok_items = [item for item in result.results if item.status == "ok"]
        empty_items = [item for item in result.results if item.status == "empty"]
        failed_items = [item for item in result.results if item.status == "error"]
        latest_data_date = max(
            (item.last_date for item in ok_items if item.last_date is not None),
            default=None,
        )

        if result.no_symbols:
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

        if failed_items and not ok_items:
            run_logger.log_asset_failure(
                asset_key="candles_daily",
                asset_kind="table",
                partition_key=partition_key,
                extra={
                    "warnings": result.warnings,
                    "failed_symbols": [item.symbol for item in failed_items],
                },
            )
            raise Failure(f"candle ingest failed for all symbols in partition {partition_key}")

        if ok_items:
            run_logger.log_asset_success(
                asset_key="candles_daily",
                asset_kind="table",
                partition_key=partition_key,
                rows_inserted=len(ok_items),
                min_event_ts=latest_data_date,
                max_event_ts=latest_data_date,
                extra={
                    "warnings": result.warnings,
                    "empty_count": len(empty_items),
                    "failed_count": len(failed_items),
                },
            )
        else:
            run_logger.log_asset_skipped(
                asset_key="candles_daily",
                asset_kind="table",
                partition_key=partition_key,
                extra={
                    "reason": "no_success_rows",
                    "warnings": result.warnings,
                },
            )

        for item in ok_items:
            if item.last_date is None:
                continue
            run_logger.upsert_watermark(
                asset_key="candles_daily",
                scope_key=item.symbol,
                watermark_ts=item.last_date,
            )
        if latest_data_date is not None:
            run_logger.upsert_watermark(
                asset_key="candles_daily",
                scope_key="ALL",
                watermark_ts=latest_data_date,
            )

        return MaterializeResult(
            metadata={
                "partition": partition_key,
                "ok_count": len(ok_items),
                "empty_count": len(empty_items),
                "failed_count": len(failed_items),
            }
        )


def _options_bars_run_args(*, partition_key: str, paths: DagsterPaths) -> dict[str, str | list[str]]:
    return {
        "partition_key": partition_key,
        "watchlists_path": str(paths.watchlists_path),
        "watchlist": list(DEFAULT_WATCHLISTS),
        "contracts_store_dir": str(paths.data_dir / "option_contracts"),
        "bars_store_dir": str(paths.data_dir / "option_bars"),
    }


def _run_options_bars_job(
    *,
    paths: DagsterPaths,
    partition_day: date,
    partition_key: str,
    run_logger: Any,
):
    try:
        return run_ingest_options_bars_job(
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


def _options_bars_skip(
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


def _finalize_options_bars_summary(
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


def _options_bars_materialize_result(*, partition_key: str, summary: Any) -> MaterializeResult:
    return MaterializeResult(
        metadata={
            "partition": partition_key,
            "ok_contracts": summary.ok_contracts,
            "error_contracts": summary.error_contracts,
            "bars_rows": summary.bars_rows,
        }
    )


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    deps=["candles_daily"],
    required_resource_keys={"paths", "runtime_config"},
)
def options_bars(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths
    with _observed_asset_run(
        context,
        job_name="dagster_options_bars",
        args=_options_bars_run_args(partition_key=partition_key, paths=paths),
    ) as run_logger:
        result = _run_options_bars_job(
            paths=paths,
            partition_day=partition_day,
            partition_key=partition_key,
            run_logger=run_logger,
        )
        if result.no_symbols:
            return _options_bars_skip(run_logger=run_logger, partition_key=partition_key, reason="no_symbols")
        if result.no_contracts:
            return _options_bars_skip(run_logger=run_logger, partition_key=partition_key, reason="no_contracts")
        if result.no_eligible_contracts:
            return _options_bars_skip(
                run_logger=run_logger,
                partition_key=partition_key,
                reason="no_eligible_contracts",
            )

        summary = result.summary
        if summary is None:
            return _options_bars_skip(run_logger=run_logger, partition_key=partition_key, reason="missing_summary")

        _finalize_options_bars_summary(
            summary=summary,
            run_logger=run_logger,
            partition_key=partition_key,
            partition_day=partition_day,
        )
        return _options_bars_materialize_result(partition_key=partition_key, summary=summary)


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    deps=["options_bars"],
    required_resource_keys={"paths", "runtime_config"},
)
def options_snapshot_file(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths

    with _observed_asset_run(
        context,
        job_name="dagster_options_snapshot_file",
        args={
            "partition_key": partition_key,
            "portfolio_path": str(paths.portfolio_path),
            "watchlists_path": str(paths.watchlists_path),
            "cache_dir": str(paths.data_dir / "options_snapshots"),
            "candle_cache_dir": str(paths.data_dir / "candles"),
        },
    ) as run_logger:
        try:
            result = run_snapshot_options_job(
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
                portfolio_loader=load_portfolio,
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

        if result.no_symbols:
            run_logger.log_asset_skipped(
                asset_key="options_snapshot_file",
                asset_kind="file",
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

        status_by_symbol = _snapshot_status_by_symbol(symbols=result.symbols, messages=result.messages)
        success_symbols = sorted(sym for sym, status in status_by_symbol.items() if status == "success")
        failed_symbols = sorted(sym for sym, status in status_by_symbol.items() if status == "failed")
        max_data_date = max(result.dates_used) if result.dates_used else None

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

        for sym in success_symbols:
            if max_data_date is None:
                continue
            run_logger.upsert_watermark(
                asset_key="options_snapshot_file",
                scope_key=sym,
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


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    deps=["options_snapshot_file"],
    required_resource_keys={"paths", "runtime_config"},
)
def options_flow(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths

    with _observed_asset_run(
        context,
        job_name="dagster_options_flow",
        args={
            "partition_key": partition_key,
            "portfolio_path": str(paths.portfolio_path),
            "watchlists_path": str(paths.watchlists_path),
            "cache_dir": str(paths.data_dir / "options_snapshots"),
        },
    ) as run_logger:
        try:
            result = run_flow_report_job(
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
                portfolio_loader=load_portfolio,
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

        if result.no_symbols:
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

        success_by_symbol, skipped_symbols = _flow_renderable_statuses(result.renderables)
        success_count = len(success_by_symbol)
        latest_success = max((d for d in success_by_symbol.values() if d is not None), default=None)

        if success_count > 0:
            run_logger.log_asset_success(
                asset_key="options_flow",
                asset_kind="table",
                partition_key=partition_key,
                rows_inserted=success_count,
                min_event_ts=latest_success,
                max_event_ts=latest_success,
                extra={"skipped_symbols": sorted(skipped_symbols)},
            )
            for sym, flow_end_date in sorted(success_by_symbol.items()):
                if flow_end_date is None:
                    continue
                run_logger.upsert_watermark(
                    asset_key="options_flow",
                    scope_key=sym,
                    watermark_ts=flow_end_date,
                )
            if latest_success is not None:
                run_logger.upsert_watermark(
                    asset_key="options_flow",
                    scope_key="ALL",
                    watermark_ts=latest_success,
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


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    deps=["options_flow"],
    required_resource_keys={"paths", "runtime_config"},
)
def derived_metrics(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths
    symbols, symbol_warnings = _resolve_default_symbols(paths)

    with _observed_asset_run(
        context,
        job_name="dagster_derived_metrics",
        args={
            "partition_key": partition_key,
            "symbols": symbols,
            "cache_dir": str(paths.data_dir / "options_snapshots"),
            "derived_dir": str(paths.data_dir / "derived"),
            "candle_cache_dir": str(paths.data_dir / "candles"),
        },
    ) as run_logger:
        if not symbols:
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

        successful_symbols: list[str] = []
        skipped_symbols: list[str] = []
        failed_symbols: list[str] = []
        bytes_written = 0

        for sym in symbols:
            try:
                result = run_derived_update_job(
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

        if failed_symbols and not successful_symbols:
            run_logger.log_asset_failure(
                asset_key="derived_metrics",
                asset_kind="table",
                partition_key=partition_key,
                extra={
                    "failed_symbols": failed_symbols,
                    "skipped_symbols": skipped_symbols,
                    "warnings": symbol_warnings,
                },
            )
            raise Failure(f"derived stage failed for all symbols in partition {partition_key}")

        if successful_symbols:
            run_logger.log_asset_success(
                asset_key="derived_metrics",
                asset_kind="table",
                partition_key=partition_key,
                rows_inserted=len(successful_symbols),
                bytes_written=bytes_written,
                min_event_ts=partition_day,
                max_event_ts=partition_day,
                extra={
                    "successful_symbols": successful_symbols,
                    "skipped_symbols": skipped_symbols,
                    "failed_symbols": failed_symbols,
                    "warnings": symbol_warnings,
                },
            )
            for sym in successful_symbols:
                run_logger.upsert_watermark(
                    asset_key="derived_metrics",
                    scope_key=sym,
                    watermark_ts=partition_day,
                )
            run_logger.upsert_watermark(
                asset_key="derived_metrics",
                scope_key="ALL",
                watermark_ts=partition_day,
            )
        else:
            run_logger.log_asset_skipped(
                asset_key="derived_metrics",
                asset_kind="table",
                partition_key=partition_key,
                extra={
                    "reason": "no_symbols_succeeded",
                    "skipped_symbols": skipped_symbols,
                    "warnings": symbol_warnings,
                },
            )

        return MaterializeResult(
            metadata={
                "partition": partition_key,
                "successful_symbols": len(successful_symbols),
                "skipped_symbols": len(skipped_symbols),
                "failed_symbols": len(failed_symbols),
            }
        )


@asset(
    partitions_def=DAILY_PARTITIONS,
    group_name="daily_visibility",
    deps=["derived_metrics"],
    required_resource_keys={"paths", "runtime_config"},
)
def briefing_markdown(context) -> MaterializeResult:
    partition_day = _partition_date(context)
    partition_key = partition_day.isoformat()
    paths: DagsterPaths = context.resources.paths

    with _observed_asset_run(
        context,
        job_name="dagster_briefing_markdown",
        args={
            "partition_key": partition_key,
            "portfolio_path": str(paths.portfolio_path),
            "watchlists_path": str(paths.watchlists_path),
            "cache_dir": str(paths.data_dir / "options_snapshots"),
            "candle_cache_dir": str(paths.data_dir / "candles"),
            "derived_dir": str(paths.data_dir / "derived"),
            "reports_dir": str(paths.data_dir / "reports" / "daily"),
        },
    ) as run_logger:
        try:
            result = run_briefing_job(
                portfolio_path=paths.portfolio_path,
                watchlists_path=paths.watchlists_path,
                watchlist=[],
                symbol=None,
                as_of=partition_key,
                compare="-1",
                cache_dir=paths.data_dir / "options_snapshots",
                candle_cache_dir=paths.data_dir / "candles",
                technicals_config=paths.repo_root / "config" / "technical_backtesting.yaml",
                out=paths.data_dir / "reports" / "daily",
                print_to_console=False,
                write_json=True,
                strict=False,
                update_derived=False,
                derived_dir=paths.data_dir / "derived",
                top=3,
                snapshot_store_builder=cli_deps.build_snapshot_store,
                derived_store_builder=cli_deps.build_derived_store,
                candle_store_builder=cli_deps.build_candle_store,
                earnings_store_builder=cli_deps.build_earnings_store,
            )
        except (VisibilityJobExecutionError, VisibilityJobParameterError) as exc:
            run_logger.log_asset_failure(
                asset_key="briefing_markdown",
                asset_kind="file",
                partition_key=partition_key,
                extra={"error": str(exc)},
            )
            raise Failure(str(exc)) from exc

        report_day = _coerce_iso_date(result.report_date) or partition_day
        markdown_bytes = result.markdown_path.stat().st_size if result.markdown_path.exists() else None
        run_logger.log_asset_success(
            asset_key="briefing_markdown",
            asset_kind="file",
            partition_key=partition_key,
            bytes_written=markdown_bytes,
            min_event_ts=report_day,
            max_event_ts=report_day,
            extra={
                "markdown_path": str(result.markdown_path),
                "json_path": None if result.json_path is None else str(result.json_path),
            },
        )
        run_logger.upsert_watermark(
            asset_key="briefing_markdown",
            scope_key="ALL",
            watermark_ts=report_day,
        )

        return MaterializeResult(
            metadata={
                "partition": partition_key,
                "report_date": report_day.isoformat(),
                "markdown_path": str(result.markdown_path),
                "json_path": None if result.json_path is None else str(result.json_path),
            }
        )


ASSET_DEFINITIONS = [
    candles_daily,
    options_bars,
    options_snapshot_file,
    options_flow,
    derived_metrics,
    briefing_markdown,
]

ASSET_ORDER = (
    "candles_daily",
    "options_bars",
    "options_snapshot_file",
    "options_flow",
    "derived_metrics",
    "briefing_markdown",
)

__all__ = ["ASSET_DEFINITIONS", "ASSET_ORDER", "DAILY_PARTITIONS"]
