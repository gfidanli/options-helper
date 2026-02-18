from __future__ import annotations

from datetime import date
from typing import Any

import typer
from rich.console import Console


ASSET_OPTION_CONTRACTS = "option_contracts"
ASSET_OPTION_BARS = "option_bars"


def handle_options_no_symbols(*, console: Console, run_logger: Any) -> None:
    run_logger.log_asset_skipped(
        asset_key=ASSET_OPTION_CONTRACTS,
        asset_kind="table",
        partition_key="ALL",
        extra={"reason": "no_symbols"},
    )
    run_logger.log_asset_skipped(
        asset_key=ASSET_OPTION_BARS,
        asset_kind="table",
        partition_key="ALL",
        extra={"reason": "no_symbols"},
    )
    console.print("No symbols found (empty watchlists and no --symbol/--contracts-root-symbol override).")
    raise typer.Exit(0)


def print_underlying_limit_warning(*, console: Console, limited_underlyings: bool, underlyings: list[str]) -> None:
    if not limited_underlyings:
        return
    console.print(f"[yellow]Limiting to {len(underlyings)} underlyings (--max-underlyings).[/yellow]")


def record_contract_discovery(
    *,
    console: Console,
    run_logger: Any,
    discovery: Any,
    fetch_only: bool,
) -> None:
    for summary in discovery.summaries:
        if summary.status != "ok":
            run_logger.log_asset_failure(
                asset_key=ASSET_OPTION_CONTRACTS,
                asset_kind="table",
                partition_key=summary.underlying,
                extra={"error": summary.error or "contract discovery failed"},
            )
            console.print(
                f"[red]Error:[/red] {summary.underlying}: {summary.error or 'contract discovery failed'}"
            )
            continue

        if summary.contracts > 0:
            if fetch_only:
                run_logger.log_asset_skipped(
                    asset_key=ASSET_OPTION_CONTRACTS,
                    asset_kind="table",
                    partition_key=summary.underlying,
                    extra={
                        "reason": "fetch_only",
                        "discovered_contracts": summary.contracts,
                        "years_scanned": summary.years_scanned,
                        "empty_years": summary.empty_years,
                    },
                )
            else:
                run_logger.log_asset_success(
                    asset_key=ASSET_OPTION_CONTRACTS,
                    asset_kind="table",
                    partition_key=summary.underlying,
                    rows_inserted=summary.contracts,
                    extra={
                        "years_scanned": summary.years_scanned,
                        "empty_years": summary.empty_years,
                    },
                )
        else:
            run_logger.log_asset_skipped(
                asset_key=ASSET_OPTION_CONTRACTS,
                asset_kind="table",
                partition_key=summary.underlying,
                extra={"reason": "no_contracts_discovered"},
            )
        console.print(
            f"{summary.underlying}: {summary.contracts} contract(s) "
            f"({summary.years_scanned} year window(s), {summary.empty_years} empty)"
        )


def handle_options_no_contracts(*, console: Console, run_logger: Any) -> None:
    run_logger.log_asset_skipped(
        asset_key=ASSET_OPTION_BARS,
        asset_kind="table",
        partition_key="ALL",
        extra={"reason": "no_contracts"},
    )
    console.print("No contracts discovered; nothing to ingest.")
    raise typer.Exit(0)


def handle_options_contracts_only(
    *,
    console: Console,
    run_logger: Any,
    contracts_only: bool,
    discovery: Any,
) -> bool:
    if not contracts_only:
        return False
    run_logger.log_asset_skipped(
        asset_key=ASSET_OPTION_BARS,
        asset_kind="table",
        partition_key="ALL",
        extra={
            "reason": "contracts_only",
            "discovered_contracts": len(discovery.contracts),
        },
    )
    console.print("Contracts-only mode: persisted contract snapshots and skipped option-bars backfill.")
    return True


def print_options_fetch_mode_notice(*, console: Console, dry_run: bool, fetch_only: bool, discovered_contracts: int) -> None:
    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] skipping writes (would upsert {discovered_contracts} contracts)."
        )
    elif fetch_only:
        console.print(
            "[yellow]Fetch-only:[/yellow] benchmarking network fetch throughput; warehouse writes are skipped."
        )


def handle_options_no_eligible_contracts(*, console: Console, run_logger: Any) -> None:
    run_logger.log_asset_skipped(
        asset_key=ASSET_OPTION_BARS,
        asset_kind="table",
        partition_key="ALL",
        extra={"reason": "no_eligible_contracts"},
    )
    console.print("No contracts eligible for bars ingestion after filtering.")
    raise typer.Exit(0)


def _bars_summary_extra(summary: Any) -> dict[str, Any]:
    return {
        "ok_contracts": summary.ok_contracts,
        "error_contracts": summary.error_contracts,
        "skipped_contracts": summary.skipped_contracts,
        "requests_attempted": summary.requests_attempted,
        "total_expiries": summary.total_expiries,
    }


def record_options_bars_summary(
    *,
    console: Console,
    run_logger: Any,
    summary: Any,
    dry_run: bool,
    fetch_only: bool,
    run_day: date,
) -> None:
    if dry_run:
        run_logger.log_asset_skipped(
            asset_key=ASSET_OPTION_BARS,
            asset_kind="table",
            partition_key="ALL",
            extra={
                "reason": "dry_run",
                "planned_contracts": summary.planned_contracts,
                "skipped_contracts": summary.skipped_contracts,
                "requests_attempted": summary.requests_attempted,
            },
        )
        console.print(
            "Dry run summary: "
            f"{summary.planned_contracts} planned, {summary.skipped_contracts} skipped, "
            f"{summary.requests_attempted} request(s) across {summary.total_expiries} expiry group(s)."
        )
        return

    if fetch_only:
        run_logger.log_asset_skipped(
            asset_key=ASSET_OPTION_BARS,
            asset_kind="table",
            partition_key="ALL",
            extra={
                "reason": "fetch_only",
                **_bars_summary_extra(summary),
                "bars_rows_fetched": summary.bars_rows,
            },
        )
        console.print(
            "Fetch-only summary: "
            f"{summary.ok_contracts} ok, {summary.error_contracts} error(s), "
            f"{summary.skipped_contracts} skipped, {summary.bars_rows} bars fetched, "
            f"{summary.requests_attempted} request(s) across {summary.total_expiries} expiry group(s)."
        )
        return

    log_fn = run_logger.log_asset_failure if summary.error_contracts > 0 else run_logger.log_asset_success
    log_fn(
        asset_key=ASSET_OPTION_BARS,
        asset_kind="table",
        partition_key="ALL",
        rows_inserted=summary.bars_rows,
        extra=_bars_summary_extra(summary),
    )
    if summary.ok_contracts > 0:
        run_logger.upsert_watermark(
            asset_key=ASSET_OPTION_BARS,
            scope_key="ALL",
            watermark_ts=run_day,
        )
    console.print(
        "Bars backfill summary: "
        f"{summary.ok_contracts} ok, {summary.error_contracts} error(s), "
        f"{summary.skipped_contracts} skipped, {summary.bars_rows} bars, "
        f"{summary.requests_attempted} request(s) across {summary.total_expiries} expiry group(s)."
    )


def print_options_endpoint_stats(*, console: Console, result: Any) -> None:
    if result.contracts_endpoint_stats is not None:
        contracts_stats = result.contracts_endpoint_stats.endpoint_stats
        console.print(
            "Endpoint stats (/v2/options/contracts): "
            f"calls={contracts_stats.calls}, "
            f"429={contracts_stats.rate_limit_429}, "
            f"timeouts={contracts_stats.timeout_count}, "
            f"p50_ms={contracts_stats.latency_p50_ms}, "
            f"p95_ms={contracts_stats.latency_p95_ms}"
        )
    if result.bars_endpoint_stats is None:
        return
    bars_stats = result.bars_endpoint_stats.endpoint_stats
    console.print(
        "Endpoint stats (/v1beta1/options/bars): "
        f"calls={bars_stats.calls}, "
        f"429={bars_stats.rate_limit_429}, "
        f"timeouts={bars_stats.timeout_count}, "
        f"splits={bars_stats.split_count}, "
        f"fallbacks={bars_stats.fallback_count}, "
        f"p50_ms={bars_stats.latency_p50_ms}, "
        f"p95_ms={bars_stats.latency_p95_ms}"
    )
