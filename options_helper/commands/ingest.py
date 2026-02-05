from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobParameterError,
    run_ingest_candles_job,
    run_ingest_options_bars_job,
)


app = typer.Typer(help="Ingestion utilities (not financial advice).")


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
        console.print("No symbols found (empty watchlists and no --symbol override).")
        raise typer.Exit(0)

    ok = 0
    empty = 0
    error = 0
    for item in result.results:
        if item.status == "ok":
            ok += 1
            console.print(f"{item.symbol}: cached through {item.last_date.isoformat()}")
        elif item.status == "empty":
            empty += 1
            console.print(f"[yellow]Warning:[/yellow] {item.symbol}: no candles returned.")
        else:
            error += 1
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
            resume=resume,
            dry_run=dry_run,
            fail_fast=fail_fast,
            provider_builder=cli_deps.build_provider,
            contracts_store_builder=cli_deps.build_option_contracts_store,
            bars_store_builder=cli_deps.build_option_bars_store,
            client_factory=AlpacaClient,
            contracts_store_dir=Path("data/option_contracts"),
            bars_store_dir=Path("data/option_bars"),
            today=date.today(),
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
        console.print("No symbols found (empty watchlists and no --symbol override).")
        raise typer.Exit(0)

    if result.limited_underlyings:
        console.print(f"[yellow]Limiting to {len(result.underlyings)} underlyings (--max-underlyings).[/yellow]")

    discovery = result.discovery
    assert discovery is not None
    for summary in discovery.summaries:
        if summary.status == "ok":
            console.print(
                f"{summary.underlying}: {summary.contracts} contract(s) "
                f"({summary.years_scanned} year window(s), {summary.empty_years} empty)"
            )
        else:
            console.print(
                f"[red]Error:[/red] {summary.underlying}: {summary.error or 'contract discovery failed'}"
            )

    if result.no_contracts:
        console.print("No contracts discovered; nothing to ingest.")
        raise typer.Exit(0)

    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] skipping writes (would upsert {len(discovery.contracts)} contracts)."
        )

    if result.no_eligible_contracts:
        console.print("No contracts eligible for bars ingestion after filtering.")
        raise typer.Exit(0)

    summary = result.summary
    assert summary is not None
    if dry_run:
        console.print(
            "Dry run summary: "
            f"{summary.planned_contracts} planned, {summary.skipped_contracts} skipped, "
            f"{summary.requests_attempted} request(s) across {summary.total_expiries} expiry group(s)."
        )
    else:
        console.print(
            "Bars backfill summary: "
            f"{summary.ok_contracts} ok, {summary.error_contracts} error(s), "
            f"{summary.skipped_contracts} skipped, {summary.bars_rows} bars, "
            f"{summary.requests_attempted} request(s) across {summary.total_expiries} expiry group(s)."
        )
