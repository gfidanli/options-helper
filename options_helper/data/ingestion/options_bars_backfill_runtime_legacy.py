from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.options_bars_backfill_executor_legacy import (
    execute_backfill_plans,
)
from options_helper.data.ingestion.options_bars_backfill_planner_legacy import (
    build_backfill_plan,
)
from options_helper.data.ingestion.tuning import EndpointStats, build_endpoint_stats
from options_helper.data.option_bars import OptionBarsStore


@dataclass(frozen=True)
class BackfillRuntimeResult:
    total_contracts: int
    total_expiries: int
    planned_contracts: int
    skipped_contracts: int
    ok_contracts: int
    error_contracts: int
    bars_rows: int
    requests_attempted: int
    endpoint_stats: EndpointStats


def _empty_runtime_result() -> BackfillRuntimeResult:
    return BackfillRuntimeResult(
        total_contracts=0,
        total_expiries=0,
        planned_contracts=0,
        skipped_contracts=0,
        ok_contracts=0,
        error_contracts=0,
        bars_rows=0,
        requests_attempted=0,
        endpoint_stats=build_endpoint_stats(calls=0, error_count=0, latencies_ms=[]),
    )


def _dry_run_result(plan) -> BackfillRuntimeResult:  # noqa: ANN001
    return BackfillRuntimeResult(
        total_contracts=plan.total_contracts,
        total_expiries=plan.total_expiries,
        planned_contracts=plan.planned_contracts,
        skipped_contracts=plan.skipped_contracts,
        ok_contracts=0,
        error_contracts=0,
        bars_rows=0,
        requests_attempted=plan.requests_attempted,
        endpoint_stats=build_endpoint_stats(calls=plan.requests_attempted, error_count=0, latencies_ms=[]),
    )


def backfill_option_bars_runtime(
    client: AlpacaClient,
    store: OptionBarsStore,
    contracts: pd.DataFrame,
    *,
    provider: str,
    lookback_years: int,
    page_limit: int | None,
    bars_concurrency: int,
    bars_max_requests_per_second: float | None,
    bars_batch_mode: str,
    bars_batch_size: int,
    bars_write_batch_size: int,
    resume: bool,
    dry_run: bool,
    fail_fast: bool,
    today: date,
) -> BackfillRuntimeResult:
    if contracts is None or contracts.empty:
        return _empty_runtime_result()

    plan = build_backfill_plan(
        store=store,
        contracts=contracts,
        provider=provider,
        lookback_years=lookback_years,
        resume=resume,
        dry_run=dry_run,
        today=today,
    )
    if dry_run:
        return _dry_run_result(plan)

    execution = execute_backfill_plans(
        client=client,
        store=store,
        provider=provider,
        page_limit=page_limit,
        bars_concurrency=bars_concurrency,
        bars_max_requests_per_second=bars_max_requests_per_second,
        bars_batch_mode=bars_batch_mode,
        bars_batch_size=bars_batch_size,
        bars_write_batch_size=bars_write_batch_size,
        fail_fast=fail_fast,
        fetch_plans=plan.fetch_plans,
    )
    endpoint_stats = build_endpoint_stats(
        calls=execution.endpoint_calls,
        retries=0,
        rate_limit_429=execution.endpoint_rate_limit_429,
        timeout_count=execution.endpoint_timeout_count,
        error_count=execution.endpoint_errors,
        latencies_ms=execution.endpoint_latencies_ms,
        split_count=execution.endpoint_split_count,
        fallback_count=execution.endpoint_fallback_count,
    )
    return BackfillRuntimeResult(
        total_contracts=plan.total_contracts,
        total_expiries=plan.total_expiries,
        planned_contracts=plan.planned_contracts,
        skipped_contracts=plan.skipped_contracts,
        ok_contracts=execution.ok_contracts,
        error_contracts=execution.error_contracts,
        bars_rows=execution.bars_rows,
        requests_attempted=execution.endpoint_calls,
        endpoint_stats=endpoint_stats,
    )


__all__ = ["backfill_option_bars_runtime"]
