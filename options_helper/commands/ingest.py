from __future__ import annotations

from datetime import date
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.candles import ingest_candles
from options_helper.data.ingestion.common import DEFAULT_WATCHLISTS, parse_date, resolve_symbols, shift_years
from options_helper.data.ingestion.options_bars import (
    backfill_option_bars,
    discover_option_contracts,
    prepare_contracts_for_bars,
)
from options_helper.data.option_bars import OptionBarsStoreError
from options_helper.data.option_contracts import OptionContractsStoreError


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
    selection = resolve_symbols(
        watchlists_path=watchlists_path,
        watchlists=watchlist,
        symbols=symbol,
        default_watchlists=DEFAULT_WATCHLISTS,
    )

    for warning in selection.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    if not selection.symbols:
        console.print("No symbols found (empty watchlists and no --symbol override).")
        raise typer.Exit(0)

    provider = cli_deps.build_provider()
    store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)

    results = ingest_candles(store, selection.symbols, period="max", best_effort=True)

    ok = 0
    empty = 0
    error = 0
    for result in results:
        if result.status == "ok":
            ok += 1
            console.print(f"{result.symbol}: cached through {result.last_date.isoformat()}")
        elif result.status == "empty":
            empty += 1
            console.print(f"[yellow]Warning:[/yellow] {result.symbol}: no candles returned.")
        else:
            error += 1
            console.print(f"[red]Error:[/red] {result.symbol}: {result.error}")

    console.print(
        f"Summary: {ok} ok, {empty} empty, {error} error(s) for {len(results)} symbol(s)."
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

    selection = resolve_symbols(
        watchlists_path=watchlists_path,
        watchlists=watchlist,
        symbols=symbol,
        default_watchlists=DEFAULT_WATCHLISTS,
    )
    for warning in selection.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    if not selection.symbols:
        console.print("No symbols found (empty watchlists and no --symbol override).")
        raise typer.Exit(0)

    underlyings = selection.symbols
    if max_underlyings is not None:
        underlyings = underlyings[:max_underlyings]
        console.print(f"[yellow]Limiting to {len(underlyings)} underlyings (--max-underlyings).[/yellow]")

    provider = cli_deps.build_provider()
    provider_name = getattr(provider, "name", None)
    if provider_name != "alpaca":
        raise typer.BadParameter("Options bars ingestion requires --provider alpaca.")

    today = date.today()
    try:
        exp_start = parse_date(contracts_exp_start, label="contracts-exp-start")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--contracts-exp-start") from exc

    if contracts_exp_end:
        try:
            exp_end = parse_date(contracts_exp_end, label="contracts-exp-end")
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--contracts-exp-end") from exc
    else:
        exp_end = shift_years(today, 5)

    if exp_end < exp_start:
        raise typer.BadParameter("contracts-exp-end must be >= contracts-exp-start")

    try:
        contracts_store = cli_deps.build_option_contracts_store(Path("data/option_contracts"))
        bars_store = cli_deps.build_option_bars_store(Path("data/option_bars"))
    except (OptionContractsStoreError, OptionBarsStoreError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    client = AlpacaClient()
    discovery = discover_option_contracts(
        client,
        underlyings=underlyings,
        exp_start=exp_start,
        exp_end=exp_end,
        page_limit=page_limit,
        max_contracts=max_contracts,
        fail_fast=fail_fast,
    )

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

    if discovery.contracts.empty:
        console.print("No contracts discovered; nothing to ingest.")
        raise typer.Exit(0)

    if dry_run:
        console.print(
            f"[yellow]Dry run:[/yellow] skipping writes (would upsert {len(discovery.contracts)} contracts)."
        )
    else:
        contracts_store.upsert_contracts(
            discovery.contracts,
            provider="alpaca",
            as_of_date=today,
            raw_by_contract_symbol=discovery.raw_by_symbol,
        )

    prepared = prepare_contracts_for_bars(
        discovery.contracts,
        max_expiries=max_expiries,
        max_contracts=max_contracts,
    )

    if prepared.contracts.empty:
        console.print("No contracts eligible for bars ingestion after filtering.")
        raise typer.Exit(0)

    summary = backfill_option_bars(
        client,
        bars_store,
        prepared.contracts,
        provider="alpaca",
        lookback_years=lookback_years,
        page_limit=page_limit,
        resume=resume,
        dry_run=dry_run,
        fail_fast=fail_fast,
        today=today,
    )

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
