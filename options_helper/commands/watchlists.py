from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from options_helper.storage import load_portfolio
from options_helper.watchlists import build_default_watchlists, load_watchlists, save_watchlists

app = typer.Typer(help="Manage symbol watchlists.")


@app.command("init")
def watchlists_init(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (used to build 'positions')."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing watchlists file."),
) -> None:
    """Create a starter watchlists store with 'positions' and 'watchlist'."""
    if watchlists_path.exists() and not force:
        raise typer.BadParameter(f"{watchlists_path} already exists (use --force to overwrite)")

    portfolio = load_portfolio(portfolio_path)
    wl = build_default_watchlists(portfolio=portfolio, extra_watchlist_symbols=["IREN"])
    save_watchlists(watchlists_path, wl)
    Console().print(f"Wrote watchlists to {watchlists_path}")


@app.command("sync-positions")
def watchlists_sync_positions(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (source of symbols)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
    name: str = typer.Option("positions", "--name", help="Watchlist name to sync."),
) -> None:
    """Update a watchlist from the unique symbols in portfolio positions."""
    portfolio = load_portfolio(portfolio_path)
    wl = load_watchlists(watchlists_path)
    wl.set(name, sorted({p.symbol.upper() for p in portfolio.positions}))
    save_watchlists(watchlists_path, wl)
    Console().print(f"Synced {name} in {watchlists_path}")


@app.command("list")
def watchlists_list(
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
) -> None:
    """List available watchlists and their symbols."""
    wl = load_watchlists(watchlists_path)
    console = Console()

    if not wl.watchlists:
        console.print(f"No watchlists in {watchlists_path}")
        return

    from rich.table import Table

    table = Table(title=f"Watchlists ({watchlists_path})")
    table.add_column("Name")
    table.add_column("Symbols")
    for name in sorted(wl.watchlists.keys()):
        table.add_row(name, ", ".join(wl.watchlists[name]))
    console.print(table)


@app.command("show")
def watchlists_show(
    name: str = typer.Argument(..., help="Watchlist name."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
) -> None:
    """Show a watchlist's symbols."""
    wl = load_watchlists(watchlists_path)
    symbols = wl.get(name)
    Console().print(f"{name}: {', '.join(symbols) if symbols else '(empty)'}")


@app.command("create")
def watchlists_create(
    name: str = typer.Argument(..., help="Watchlist name."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
) -> None:
    """Create an empty watchlist."""
    wl = load_watchlists(watchlists_path)
    if name in wl.watchlists:
        raise typer.BadParameter(f"Watchlist already exists: {name}")
    wl.set(name, [])
    save_watchlists(watchlists_path, wl)
    Console().print(f"Created {name} in {watchlists_path}")


@app.command("add")
def watchlists_add(
    name: str = typer.Argument(..., help="Watchlist name."),
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
) -> None:
    """Add a symbol to a watchlist."""
    wl = load_watchlists(watchlists_path)
    if name not in wl.watchlists:
        wl.set(name, [])
    wl.add(name, symbol)
    save_watchlists(watchlists_path, wl)
    Console().print(f"Added {symbol.upper()} to {name}")


@app.command("remove")
def watchlists_remove(
    name: str = typer.Argument(..., help="Watchlist name."),
    symbol: str = typer.Argument(..., help="Ticker symbol."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--path",
        help="Path to watchlists JSON store.",
    ),
) -> None:
    """Remove a symbol from a watchlist."""
    wl = load_watchlists(watchlists_path)
    if name not in wl.watchlists:
        raise typer.BadParameter(f"Unknown watchlist: {name}")
    wl.remove(name, symbol)
    save_watchlists(watchlists_path, wl)
    Console().print(f"Removed {symbol.upper()} from {name}")
