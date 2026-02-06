from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobParameterError,
    run_ingest_candles_job,
    run_ingest_options_bars_job,
)


app = typer.Typer(help="Ingestion utilities (not financial advice).")

JOB_INGEST_CANDLES = "ingest_candles"
JOB_INGEST_OPTIONS_BARS = "ingest_options_bars"

ASSET_CANDLES_DAILY = "candles_daily"
ASSET_OPTION_CONTRACTS = "option_contracts"
ASSET_OPTION_BARS = "option_bars"

NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _coerce_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except Exception:  # noqa: BLE001
        return None


def _latest_contract_expiry(discovery: object | None) -> date | None:
    if discovery is None:
        return None
    contracts = getattr(discovery, "contracts", None)
    if contracts is None or getattr(contracts, "empty", True):
        return None
    columns = getattr(contracts, "columns", [])
    if "expiry" not in columns:
        return None
    latest: date | None = None
    for raw in contracts["expiry"].tolist():
        parsed = _coerce_date(raw)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


@app.command("candles")
def ingest_candles_command(
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: list[str] = typer.Option(
        list(DEFAULT_WATCHLISTS),
        "--watchlist",
        help="Watchlist name(s) to ingest (default: positions + monitor).",
    ),
    symbol: list[str] = typer.Option(
        [],
        "--symbol",
        "-s",
        help="Optional symbol override (repeatable or comma-separated).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
) -> None:
    """Backfill daily candles for watchlist symbols (period=max)."""
    console = Console(width=200)
    with _observed_run(
        console=console,
        job_name=JOB_INGEST_CANDLES,
        args={
            "watchlists_path": str(watchlists_path),
            "watchlist": watchlist,
            "symbol": symbol,
            "candle_cache_dir": str(candle_cache_dir),
        },
    ) as run_logger:
        result = run_ingest_candles_job(
            watchlists_path=watchlists_path,
            watchlist=watchlist,
            symbol=symbol,
            candle_cache_dir=candle_cache_dir,
            provider_builder=cli_deps.build_provider,
            candle_store_builder=cli_deps.build_candle_store,
        )

        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

        if result.no_symbols:
            run_logger.log_asset_skipped(
                asset_key=ASSET_CANDLES_DAILY,
                asset_kind="table",
                partition_key="ALL",
                extra={"reason": "no_symbols"},
            )
            console.print("No symbols found (empty watchlists and no --symbol override).")
            raise typer.Exit(0)

        ok = 0
        empty = 0
        error = 0
        for item in result.results:
            if item.status == "ok":
                ok += 1
                run_logger.log_asset_success(
                    asset_key=ASSET_CANDLES_DAILY,
                    asset_kind="table",
                    partition_key=item.symbol,
                    min_event_ts=item.last_date,
                    max_event_ts=item.last_date,
                )
                if item.last_date is not None:
                    run_logger.upsert_watermark(
                        asset_key=ASSET_CANDLES_DAILY,
                        scope_key=item.symbol,
                        watermark_ts=item.last_date,
                    )
                console.print(f"{item.symbol}: cached through {item.last_date.isoformat()}")
            elif item.status == "empty":
                empty += 1
                run_logger.log_asset_skipped(
                    asset_key=ASSET_CANDLES_DAILY,
                    asset_kind="table",
                    partition_key=item.symbol,
                    extra={"reason": "empty"},
                )
                console.print(f"[yellow]Warning:[/yellow] {item.symbol}: no candles returned.")
            else:
                error += 1
                run_logger.log_asset_failure(
                    asset_key=ASSET_CANDLES_DAILY,
                    asset_kind="table",
                    partition_key=item.symbol,
                    extra={"error": item.error},
                )
                console.print(f"[red]Error:[/red] {item.symbol}: {item.error}")

        console.print(
            f"Summary: {ok} ok, {empty} empty, {error} error(s) for {len(result.results)} symbol(s)."
        )


@app.command("options-bars")
def ingest_options_bars_command(
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: list[str] = typer.Option(
        list(DEFAULT_WATCHLISTS),
        "--watchlist",
        help="Watchlist name(s) to ingest (default: positions + monitor).",
    ),
    symbol: list[str] = typer.Option(
        [],
        "--symbol",
        "-s",
        help="Optional symbol override (repeatable or comma-separated).",
    ),
    contracts_exp_start: str = typer.Option(
        "2000-01-01",
        "--contracts-exp-start",
        help="Contracts expiration start date (YYYY-MM-DD).",
    ),
    contracts_exp_end: str | None = typer.Option(
        None,
        "--contracts-exp-end",
        help="Contracts expiration end date (YYYY-MM-DD). Defaults to today + 5y.",
    ),
    lookback_years: int = typer.Option(
        10,
        "--lookback-years",
        min=1,
        help="Years of daily bars to backfill per expiry.",
    ),
    page_limit: int = typer.Option(
        200,
        "--page-limit",
        min=1,
        help="Max pages to request from Alpaca per call.",
    ),
    max_underlyings: int | None = typer.Option(
        None,
        "--max-underlyings",
        min=1,
        help="Safety cap on number of underlyings to ingest.",
    ),
    max_contracts: int | None = typer.Option(
        None,
        "--max-contracts",
        min=1,
        help="Safety cap on total contracts to ingest.",
    ),
    max_expiries: int | None = typer.Option(
        None,
        "--max-expiries",
        min=1,
        help="Safety cap on expiries (most-recent first).",
    ),
    contracts_max_requests_per_second: float | None = typer.Option(
        2.5,
        "--contracts-max-rps",
        min=0.1,
        help="Soft throttle for options-contracts requests per second.",
    ),
    bars_concurrency: int = typer.Option(
        8,
        "--bars-concurrency",
        min=1,
        help="Concurrent option-contract bars fetches (reduced to 1 when --fail-fast).",
    ),
    bars_max_requests_per_second: float | None = typer.Option(
        30.0,
        "--bars-max-rps",
        min=0.1,
        help="Soft throttle for options-bars requests per second.",
    ),
    bars_write_batch_size: int = typer.Option(
        200,
        "--bars-write-batch-size",
        min=1,
        help="Contracts per DB/meta write batch for options-bars ingestion.",
    ),
    alpaca_http_pool_maxsize: int | None = typer.Option(
        None,
        "--alpaca-http-pool-maxsize",
        min=1,
        help="Requests connection pool max size for Alpaca clients used by this run.",
    ),
    alpaca_http_pool_connections: int | None = typer.Option(
        None,
        "--alpaca-http-pool-connections",
        min=1,
        help="Requests connection pool count for Alpaca clients used by this run.",
    ),
    log_rate_limits: bool = typer.Option(
        False,
        "--log-rate-limits",
        help="Enable per-request Alpaca rate-limit logging for this run.",
    ),
    no_log_rate_limits: bool = typer.Option(
        False,
        "--no-log-rate-limits",
        help="Disable per-request Alpaca rate-limit logging for this run.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Skip contracts already covered in option_bars_meta.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do not write data; only print planned fetch ranges.",
    ),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast/--best-effort",
        help="Stop on first error (default: best-effort).",
    ),
) -> None:
    """Discover Alpaca option contracts and backfill daily bars."""
    console = Console(width=200)
    run_day = date.today()
    if log_rate_limits and no_log_rate_limits:
        raise typer.BadParameter("Choose either --log-rate-limits or --no-log-rate-limits, not both.")

    log_rate_limits_override: bool | None = None
    if log_rate_limits:
        log_rate_limits_override = True
    elif no_log_rate_limits:
        log_rate_limits_override = False

    with _observed_run(
        console=console,
        job_name=JOB_INGEST_OPTIONS_BARS,
        args={
            "watchlists_path": str(watchlists_path),
            "watchlist": watchlist,
            "symbol": symbol,
            "contracts_exp_start": contracts_exp_start,
            "contracts_exp_end": contracts_exp_end,
            "lookback_years": lookback_years,
            "page_limit": page_limit,
            "max_underlyings": max_underlyings,
            "max_contracts": max_contracts,
            "max_expiries": max_expiries,
            "contracts_max_requests_per_second": contracts_max_requests_per_second,
            "bars_concurrency": bars_concurrency,
            "bars_max_requests_per_second": bars_max_requests_per_second,
            "bars_write_batch_size": bars_write_batch_size,
            "alpaca_http_pool_maxsize": alpaca_http_pool_maxsize,
            "alpaca_http_pool_connections": alpaca_http_pool_connections,
            "log_rate_limits": log_rate_limits_override,
            "resume": resume,
            "dry_run": dry_run,
            "fail_fast": fail_fast,
        },
    ) as run_logger:
        def _build_client() -> AlpacaClient:
            kwargs: dict[str, object] = {}
            if log_rate_limits_override is not None:
                kwargs["log_rate_limits"] = log_rate_limits_override
            if alpaca_http_pool_maxsize is not None:
                kwargs["http_pool_maxsize"] = alpaca_http_pool_maxsize
            if alpaca_http_pool_connections is not None:
                kwargs["http_pool_connections"] = alpaca_http_pool_connections

            if not kwargs:
                return AlpacaClient()
            try:
                return AlpacaClient(**kwargs)
            except TypeError:
                # Backward-compatible fallback for tests that monkeypatch AlpacaClient
                # with callables that do not accept keyword args.
                return AlpacaClient()

        try:
            result = run_ingest_options_bars_job(
                watchlists_path=watchlists_path,
                watchlist=watchlist,
                symbol=symbol,
                contracts_exp_start=contracts_exp_start,
                contracts_exp_end=contracts_exp_end,
                lookback_years=lookback_years,
                page_limit=page_limit,
                max_underlyings=max_underlyings,
                max_contracts=max_contracts,
                max_expiries=max_expiries,
                contracts_max_requests_per_second=contracts_max_requests_per_second,
                bars_concurrency=bars_concurrency,
                bars_max_requests_per_second=bars_max_requests_per_second,
                bars_write_batch_size=bars_write_batch_size,
                resume=resume,
                dry_run=dry_run,
                fail_fast=fail_fast,
                provider_builder=cli_deps.build_provider,
                contracts_store_builder=cli_deps.build_option_contracts_store,
                bars_store_builder=cli_deps.build_option_bars_store,
                client_factory=_build_client,
                contracts_store_dir=Path("data/option_contracts"),
                bars_store_dir=Path("data/option_bars"),
                today=run_day,
            )
        except VisibilityJobParameterError as exc:
            if exc.param_hint:
                raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
            raise typer.BadParameter(str(exc)) from exc
        except (OptionContractsStoreError, OptionBarsStoreError) as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

        if result.no_symbols:
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
            console.print("No symbols found (empty watchlists and no --symbol override).")
            raise typer.Exit(0)

        if result.limited_underlyings:
            console.print(
                f"[yellow]Limiting to {len(result.underlyings)} underlyings (--max-underlyings).[/yellow]"
            )

        discovery = result.discovery
        assert discovery is not None
        for summary in discovery.summaries:
            if summary.status == "ok":
                if summary.contracts > 0:
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
            else:
                run_logger.log_asset_failure(
                    asset_key=ASSET_OPTION_CONTRACTS,
                    asset_kind="table",
                    partition_key=summary.underlying,
                    extra={"error": summary.error or "contract discovery failed"},
                )
                console.print(
                    f"[red]Error:[/red] {summary.underlying}: {summary.error or 'contract discovery failed'}"
                )

        latest_expiry = _latest_contract_expiry(discovery)
        if latest_expiry is not None:
            run_logger.upsert_watermark(
                asset_key=ASSET_OPTION_CONTRACTS,
                scope_key="ALL",
                watermark_ts=latest_expiry,
            )

        if result.no_contracts:
            run_logger.log_asset_skipped(
                asset_key=ASSET_OPTION_BARS,
                asset_kind="table",
                partition_key="ALL",
                extra={"reason": "no_contracts"},
            )
            console.print("No contracts discovered; nothing to ingest.")
            raise typer.Exit(0)

        if dry_run:
            console.print(
                f"[yellow]Dry run:[/yellow] skipping writes (would upsert {len(discovery.contracts)} contracts)."
            )

        if result.no_eligible_contracts:
            run_logger.log_asset_skipped(
                asset_key=ASSET_OPTION_BARS,
                asset_kind="table",
                partition_key="ALL",
                extra={"reason": "no_eligible_contracts"},
            )
            console.print("No contracts eligible for bars ingestion after filtering.")
            raise typer.Exit(0)

        summary = result.summary
        assert summary is not None
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
        else:
            if summary.error_contracts > 0:
                run_logger.log_asset_failure(
                    asset_key=ASSET_OPTION_BARS,
                    asset_kind="table",
                    partition_key="ALL",
                    rows_inserted=summary.bars_rows,
                    extra={
                        "ok_contracts": summary.ok_contracts,
                        "error_contracts": summary.error_contracts,
                        "skipped_contracts": summary.skipped_contracts,
                        "requests_attempted": summary.requests_attempted,
                        "total_expiries": summary.total_expiries,
                    },
                )
            else:
                run_logger.log_asset_success(
                    asset_key=ASSET_OPTION_BARS,
                    asset_kind="table",
                    partition_key="ALL",
                    rows_inserted=summary.bars_rows,
                    extra={
                        "ok_contracts": summary.ok_contracts,
                        "error_contracts": summary.error_contracts,
                        "skipped_contracts": summary.skipped_contracts,
                        "requests_attempted": summary.requests_attempted,
                        "total_expiries": summary.total_expiries,
                    },
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
