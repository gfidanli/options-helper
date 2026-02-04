from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import typer
import pandas as pd
from rich.console import Console

from options_helper.analysis.osi import normalize_underlying
from options_helper.cli_deps import build_provider
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.intraday_store import IntradayStore
from options_helper.data.market_types import DataFetchError
from options_helper.data.option_contracts import OptionContractsStore, OptionContractsStoreError

app = typer.Typer(help="Intraday data capture (not financial advice).")


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


def _parse_symbols(symbols: list[str] | None, symbols_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in symbols or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if symbols_csv:
        tokens.extend([part for part in symbols_csv.split(",") if part.strip()])
    normalized = [normalize_underlying(tok) for tok in tokens if tok]
    return sorted({tok for tok in normalized if tok})


def _parse_expiries(expiries: list[str] | None, expiries_csv: str | None) -> list[date]:
    tokens: list[str] = []
    for item in expiries or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if expiries_csv:
        tokens.extend([part for part in expiries_csv.split(",") if part.strip()])
    parsed = [_parse_date(tok) for tok in tokens]
    return sorted({val for val in parsed if val})


def _latest_contracts_date(contracts_dir: Path, symbol: str) -> date | None:
    sym_dir = contracts_dir / symbol.upper()
    if not sym_dir.exists():
        return None
    dates: list[date] = []
    for entry in sym_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            dates.append(date.fromisoformat(entry.name))
        except ValueError:
            continue
    return max(dates) if dates else None


def _resolve_contracts_as_of(contracts_dir: Path, symbol: str, spec: str) -> date | None:
    spec = spec.strip().lower()
    if spec == "latest":
        return _latest_contracts_date(contracts_dir, symbol)
    return _parse_date(spec)


def _require_alpaca_provider() -> None:
    provider = build_provider()
    name = getattr(provider, "name", None)
    if name != "alpaca":
        raise typer.BadParameter("Intraday capture currently requires --provider alpaca.")


@app.command("pull-stocks-bars")
def pull_stocks_bars(
    symbol: list[str] | None = typer.Option(
        None,
        "--symbol",
        "-s",
        help="Stock symbol to pull (repeatable or comma-separated).",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated stock symbols to pull.",
    ),
    day: str = typer.Option(
        date.today().isoformat(),
        "--day",
        help="Trading day to pull (YYYY-MM-DD).",
    ),
    timeframe: str = typer.Option(
        "1Min",
        "--timeframe",
        help="Intraday timeframe (1Min or 5Min).",
    ),
    feed: str | None = typer.Option(
        None,
        "--feed",
        help="Alpaca stock feed override (defaults to OH_ALPACA_STOCK_FEED).",
    ),
    out_dir: Path = typer.Option(
        Path("data/intraday"),
        "--out-dir",
        help="Base directory for intraday partitions.",
    ),
) -> None:
    """Pull intraday stock bars and persist CSV.gz partitions."""
    _require_alpaca_provider()
    console = Console()

    symbols_list = _parse_symbols(symbol, symbols)
    if not symbols_list:
        raise typer.BadParameter("Provide at least one --symbol or --symbols.")

    target_day = _parse_date(day)
    client = AlpacaClient()
    store = IntradayStore(out_dir)

    failures = 0
    for sym in symbols_list:
        try:
            df = client.get_stock_bars_intraday(
                sym,
                day=target_day,
                timeframe=timeframe,
                feed=feed,
            )
            meta = {
                "provider": "alpaca",
                "provider_version": client.provider_version,
                "request": {
                    "day": target_day.isoformat(),
                    "timeframe": timeframe,
                    "feed": feed or client.stock_feed,
                },
            }
            path = store.save_partition("stocks", "bars", timeframe, sym, target_day, df, meta)
            console.print(f"[green]Saved[/green] {sym} -> {path}")
        except DataFetchError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")

    if failures:
        console.print(f"[yellow]{failures} symbol(s) failed.[/yellow]")


@app.command("pull-options-bars")
def pull_options_bars(
    underlying: list[str] | None = typer.Option(
        None,
        "--underlying",
        "-u",
        help="Underlying symbol to pull (repeatable or comma-separated).",
    ),
    underlyings: str | None = typer.Option(
        None,
        "--underlyings",
        help="Comma-separated underlyings to pull.",
    ),
    expiry: list[str] | None = typer.Option(
        None,
        "--expiry",
        help="Filter to expiry date(s) (repeatable or comma-separated).",
    ),
    expiries: str | None = typer.Option(
        None,
        "--expiries",
        help="Comma-separated expiries (YYYY-MM-DD).",
    ),
    contracts_dir: Path = typer.Option(
        Path("data/option_contracts"),
        "--contracts-dir",
        help="Directory containing option contracts cache.",
    ),
    contracts_as_of: str = typer.Option(
        "latest",
        "--contracts-as-of",
        help="Contract cache date to use (YYYY-MM-DD or 'latest').",
    ),
    day: str = typer.Option(
        date.today().isoformat(),
        "--day",
        help="Trading day to pull (YYYY-MM-DD).",
    ),
    timeframe: str = typer.Option(
        "1Min",
        "--timeframe",
        help="Intraday timeframe (1Min or 5Min).",
    ),
    feed: str | None = typer.Option(
        None,
        "--feed",
        help="Alpaca options feed override (defaults to OH_ALPACA_OPTIONS_FEED).",
    ),
    out_dir: Path = typer.Option(
        Path("data/intraday"),
        "--out-dir",
        help="Base directory for intraday partitions.",
    ),
) -> None:
    """Pull intraday option bars for cached contracts and persist CSV.gz partitions."""
    _require_alpaca_provider()
    console = Console()

    underlyings_list = _parse_symbols(underlying, underlyings)
    if not underlyings_list:
        raise typer.BadParameter("Provide at least one --underlying or --underlyings.")

    expiry_dates = _parse_expiries(expiry, expiries)
    target_day = _parse_date(day)
    contracts_store = OptionContractsStore(contracts_dir)
    client = AlpacaClient()
    store = IntradayStore(out_dir)

    failures = 0
    for sym in underlyings_list:
        as_of = _resolve_contracts_as_of(contracts_dir, sym, contracts_as_of)
        if as_of is None:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no contracts cache found.")
            continue

        try:
            contracts_df = contracts_store.load(sym, as_of)
        except OptionContractsStoreError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")
            continue

        if contracts_df is None or contracts_df.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: contracts cache empty.")
            continue

        filtered = contracts_df
        if expiry_dates and "expiry" in filtered.columns:
            allowed = {exp.isoformat() for exp in expiry_dates}
            filtered = filtered[filtered["expiry"].astype(str).isin(allowed)]

        if filtered.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no contracts for requested expiry.")
            continue

        contract_symbols = [
            str(raw).strip()
            for raw in filtered.get("contractSymbol", pd.Series(dtype=str)).dropna().tolist()
            if str(raw).strip()
        ]
        if not contract_symbols:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: missing contract symbols in cache.")
            continue

        try:
            bars_df = client.get_option_bars_intraday(
                contract_symbols,
                day=target_day,
                timeframe=timeframe,
                feed=feed,
            )
        except DataFetchError as exc:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: {exc}")
            continue

        if bars_df.empty:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no intraday bars returned.")
            continue

        written = 0
        for contract in contract_symbols:
            sub = bars_df[bars_df["contractSymbol"] == contract]
            if sub.empty:
                continue
            meta = {
                "provider": "alpaca",
                "provider_version": client.provider_version,
                "underlying": sym,
                "contracts_as_of": as_of.isoformat(),
                "request": {
                    "day": target_day.isoformat(),
                    "timeframe": timeframe,
                    "feed": feed or client.options_feed,
                },
            }
            path = store.save_partition("options", "bars", timeframe, contract, target_day, sub, meta)
            written += 1
            console.print(f"[green]Saved[/green] {contract} -> {path}")

        if written == 0:
            failures += 1
            console.print(f"[yellow]Warning:[/yellow] {sym}: no matching intraday bars to save.")

    if failures:
        console.print(f"[yellow]{failures} warning(s) encountered.[/yellow]")
