from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.compat import sync_legacy_seams
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


@wraps(legacy.research)
def research(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.research(*args, **kwargs)


def refresh_candles(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (used to include position underlyings)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbols are included if present).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    period: str = typer.Option(
        "5y",
        "--period",
        help="Daily candle period to ensure cached (yfinance period format).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to include (default: all watchlists).",
    ),
) -> None:
    """Refresh cached daily candles for portfolio symbols and watchlists."""
    portfolio = load_portfolio(portfolio_path)
    symbols: set[str] = {position.symbol.upper() for position in portfolio.positions}

    wl = load_watchlists(watchlists_path)
    if watchlist:
        for name in watchlist:
            symbols.update(wl.get(name))
    else:
        for syms in (wl.watchlists or {}).values():
            symbols.update(syms or [])

    symbols = {symbol.strip().upper() for symbol in symbols if symbol and symbol.strip()}
    if not symbols:
        Console().print("No symbols found (no positions and no watchlists).")
        raise typer.Exit(0)

    provider = cli_deps.build_provider()
    store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    console = Console()
    console.print(f"Refreshing daily candles for {len(symbols)} symbol(s)...")

    for sym in sorted(symbols):
        try:
            history = store.get_daily_history(sym, period=period)
            if history.empty:
                console.print(f"[yellow]Warning:[/yellow] {sym}: no candles returned.")
            else:
                last_dt = history.index.max()
                console.print(f"{sym}: cached through {last_dt.date().isoformat()}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {sym}: {exc}")


@wraps(legacy.analyze)
def analyze(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.analyze(*args, **kwargs)


@wraps(legacy.watch)
def watch(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.watch(*args, **kwargs)


__all__ = ["research", "refresh_candles", "analyze", "watch"]
