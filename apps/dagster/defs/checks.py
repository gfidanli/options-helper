from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Any, Iterable

from dagster import AssetCheckExecutionContext, AssetCheckResult, asset_check

import options_helper.cli_deps as cli_deps
from options_helper.data.quality_checks import (
    QualityCheckResult,
    persist_quality_checks,
    run_candle_quality_checks,
    run_derived_quality_checks,
    run_flow_quality_checks,
    run_options_bars_quality_checks,
    run_snapshot_quality_checks,
)

from .assets import (
    briefing_markdown,
    candles_daily,
    derived_metrics,
    options_bars,
    options_flow,
    options_snapshot_file,
    _resolve_default_symbols,
    _runtime_defaults,
)
from .resources import DagsterPaths, DagsterRuntimeConfig


def _partition_key_for_check(context: AssetCheckExecutionContext) -> str:
    partition_key = str(getattr(context, "partition_key", "") or "").strip()
    if partition_key:
        return partition_key
    asset_partition_key = str(getattr(context, "asset_partition_key", "") or "").strip()
    return asset_partition_key or "ALL"


def _dagster_parent_run_id(context: AssetCheckExecutionContext) -> str | None:
    raw = str(getattr(context, "run_id", "") or "").strip()
    return raw or None


def _coerce_iso_date(value: object) -> date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


def _quality_summary(checks: list[QualityCheckResult]) -> tuple[bool, dict[str, Any]]:
    counts = {"pass": 0, "fail": 0, "skip": 0}
    failing_checks: list[str] = []
    for check in checks:
        status = str(check.status).strip().lower()
        if status not in counts:
            continue
        counts[status] += 1
        if status == "fail":
            failing_checks.append(check.check_name)
    return (
        counts["fail"] == 0,
        {
            "total_checks": len(checks),
            "pass_checks": counts["pass"],
            "fail_checks": counts["fail"],
            "skip_checks": counts["skip"],
            "failing_check_names": sorted(set(failing_checks)),
        },
    )


@contextmanager
def _observed_check_run(
    context: AssetCheckExecutionContext,
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


@asset_check(
    asset=candles_daily,
    name="candles_daily_quality",
    description="Run candles quality checks and persist rows to meta.asset_checks.",
)
def candles_daily_quality(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    paths: DagsterPaths = context.resources.paths
    symbols, warnings = _resolve_default_symbols(paths)

    with _observed_check_run(
        context,
        job_name="dagster_check_candles_daily",
        args={"partition_key": partition_key, "symbols": symbols},
    ) as run_logger:
        candle_store = cli_deps.build_candle_store(paths.data_dir / "candles")
        checks = run_candle_quality_checks(
            candle_store=candle_store,
            symbols=symbols,
            skip_reason="no_symbols" if not symbols else None,
        )
        persist_quality_checks(run_logger=run_logger, checks=checks)
        passed, metadata = _quality_summary(checks)
        metadata["warnings"] = warnings
        return AssetCheckResult(passed=passed, metadata=metadata)


@asset_check(
    asset=options_bars,
    name="options_bars_quality",
    description="Run option bars quality checks and persist rows to meta.asset_checks.",
)
def options_bars_quality(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    paths: DagsterPaths = context.resources.paths

    with _observed_check_run(
        context,
        job_name="dagster_check_options_bars",
        args={"partition_key": partition_key},
    ) as run_logger:
        bars_store = cli_deps.build_option_bars_store(paths.data_dir / "option_bars")
        checks = run_options_bars_quality_checks(
            bars_store=bars_store,
            contract_symbols=None,
            dry_run=False,
        )
        persist_quality_checks(run_logger=run_logger, checks=checks)
        passed, metadata = _quality_summary(checks)
        return AssetCheckResult(passed=passed, metadata=metadata)


@asset_check(
    asset=options_snapshot_file,
    name="options_snapshot_file_quality",
    description="Validate snapshot contract-symbol parsing checks and persist rows.",
)
def options_snapshot_file_quality(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    partition_day = _coerce_iso_date(partition_key)
    paths: DagsterPaths = context.resources.paths
    symbols, warnings = _resolve_default_symbols(paths)

    with _observed_check_run(
        context,
        job_name="dagster_check_options_snapshot_file",
        args={"partition_key": partition_key, "symbols": symbols},
    ) as run_logger:
        snapshot_store = cli_deps.build_snapshot_store(paths.data_dir / "options_snapshots")
        snapshot_dates_by_symbol: dict[str, date] = {}
        for sym in symbols:
            try:
                snapshot_dates_by_symbol[sym] = snapshot_store.resolve_date(
                    sym,
                    partition_key if partition_day is not None else "latest",
                )
            except Exception:  # noqa: BLE001
                continue
        checks = run_snapshot_quality_checks(
            snapshot_store=snapshot_store,
            snapshot_dates_by_symbol=snapshot_dates_by_symbol,
            skip_reason="no_snapshot_dates" if not snapshot_dates_by_symbol else None,
        )
        persist_quality_checks(run_logger=run_logger, checks=checks)
        passed, metadata = _quality_summary(checks)
        metadata["warnings"] = warnings
        metadata["symbols_checked"] = len(snapshot_dates_by_symbol)
        return AssetCheckResult(passed=passed, metadata=metadata)


@asset_check(
    asset=options_flow,
    name="options_flow_quality",
    description="Validate options_flow primary key guards and persist check rows.",
)
def options_flow_quality(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    paths: DagsterPaths = context.resources.paths
    symbols, warnings = _resolve_default_symbols(paths)

    with _observed_check_run(
        context,
        job_name="dagster_check_options_flow",
        args={"partition_key": partition_key, "symbols": symbols},
    ) as run_logger:
        flow_store = cli_deps.build_flow_store(paths.data_dir / "options_snapshots")
        checks = run_flow_quality_checks(flow_store=flow_store, symbols=symbols)
        persist_quality_checks(run_logger=run_logger, checks=checks)
        passed, metadata = _quality_summary(checks)
        metadata["warnings"] = warnings
        return AssetCheckResult(passed=passed, metadata=metadata)


@asset_check(
    asset=derived_metrics,
    name="derived_metrics_quality",
    description="Validate derived duplicate-key guards and persist check rows.",
)
def derived_metrics_quality(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    paths: DagsterPaths = context.resources.paths
    symbols, warnings = _resolve_default_symbols(paths)

    with _observed_check_run(
        context,
        job_name="dagster_check_derived_metrics",
        args={"partition_key": partition_key, "symbols": symbols},
    ) as run_logger:
        derived_store = cli_deps.build_derived_store(paths.data_dir / "derived")
        checks: list[QualityCheckResult] = []
        if not symbols:
            checks.extend(
                run_derived_quality_checks(
                    derived_store=derived_store,
                    symbol="ALL",
                    skip_reason="no_symbols",
                )
            )
        else:
            for sym in symbols:
                checks.extend(run_derived_quality_checks(derived_store=derived_store, symbol=sym))
        persist_quality_checks(run_logger=run_logger, checks=checks)
        passed, metadata = _quality_summary(checks)
        metadata["warnings"] = warnings
        return AssetCheckResult(passed=passed, metadata=metadata)


@asset_check(
    asset=briefing_markdown,
    name="briefing_markdown_nonempty",
    description="Ensure daily briefing markdown exists and is not empty.",
)
def briefing_markdown_nonempty(context) -> AssetCheckResult:
    partition_key = _partition_key_for_check(context)
    paths: DagsterPaths = context.resources.paths
    markdown_path = paths.data_dir / "reports" / "daily" / f"{partition_key}.md"

    with _observed_check_run(
        context,
        job_name="dagster_check_briefing_markdown",
        args={"partition_key": partition_key, "path": str(markdown_path)},
    ) as run_logger:
        bytes_written = markdown_path.stat().st_size if markdown_path.exists() else 0
        passed = markdown_path.exists() and bytes_written > 0
        run_logger.log_check(
            asset_key="briefing_markdown",
            check_name="briefing_markdown_nonempty",
            severity="error",
            status="pass" if passed else "fail",
            partition_key=partition_key,
            metrics={"path": str(markdown_path), "bytes": int(bytes_written)},
            message=(
                "briefing markdown exists and is non-empty"
                if passed
                else "briefing markdown missing or empty"
            ),
        )
        return AssetCheckResult(
            passed=passed,
            metadata={"path": str(markdown_path), "bytes": int(bytes_written)},
        )


ASSET_CHECK_DEFINITIONS = [
    candles_daily_quality,
    options_bars_quality,
    options_snapshot_file_quality,
    options_flow_quality,
    derived_metrics_quality,
    briefing_markdown_nonempty,
]


def build_asset_checks() -> tuple[object, ...]:
    return tuple(ASSET_CHECK_DEFINITIONS)


__all__ = ["ASSET_CHECK_DEFINITIONS", "build_asset_checks"]
