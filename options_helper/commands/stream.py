from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path

import typer
import pandas as pd
from rich.console import Console

from options_helper.analysis.osi import normalize_underlying
import options_helper.cli_deps as cli_deps
from options_helper.data.market_types import DataFetchError
from options_helper.data.option_contracts import OptionContractsStore, OptionContractsStoreError
from options_helper.data.streaming.runner import StreamRunner

app = typer.Typer(help="Streaming capture (Alpaca, not financial advice).")


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


def _parse_contracts(contracts: list[str] | None, contracts_csv: str | None) -> list[str]:
    tokens: list[str] = []
    for item in contracts or []:
        if not item:
            continue
        tokens.extend([part for part in str(item).split(",") if part.strip()])
    if contracts_csv:
        tokens.extend([part for part in contracts_csv.split(",") if part.strip()])
    normalized = [str(tok).strip().upper() for tok in tokens if str(tok).strip()]
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
    provider = cli_deps.build_provider()
    name = getattr(provider, "name", None)
    if name != "alpaca":
        raise typer.BadParameter("Streaming capture currently requires --provider alpaca.")


def _expand_underlyings_to_contracts(
    store: OptionContractsStore,
    *,
    contracts_dir: Path,
    underlyings: list[str],
    as_of_spec: str,
    expiry_dates: list[date],
) -> tuple[list[str], int]:
    contracts: list[str] = []
    missing = 0
    for sym in underlyings:
        as_of = _resolve_contracts_as_of(contracts_dir, sym, as_of_spec)
        if as_of is None:
            missing += 1
            continue
        try:
            df = store.load(sym, as_of)
        except OptionContractsStoreError:
            missing += 1
            continue
        if df is None or df.empty:
            missing += 1
            continue
        filtered = df
        if expiry_dates and "expiry" in filtered.columns:
            allowed = {exp.isoformat() for exp in expiry_dates}
            filtered = filtered[filtered["expiry"].astype(str).isin(allowed)]
        raw = filtered.get("contractSymbol", pd.Series(dtype=str)).dropna().tolist()
        contracts.extend([str(item).strip().upper() for item in raw if str(item).strip()])
    unique = sorted({tok for tok in contracts if tok})
    return unique, missing


_STREAM_CAPTURE_STOCK_OPT = typer.Option(
    None,
    "--stock",
    "-s",
    help="Stock symbol to stream (repeatable or comma-separated).",
)
_STREAM_CAPTURE_STOCKS_OPT = typer.Option(
    None,
    "--stocks",
    help="Comma-separated stock symbols to stream.",
)
_STREAM_CAPTURE_OPTIONS_CONTRACT_OPT = typer.Option(
    None,
    "--options-contract",
    "--option-contract",
    help="Option contract symbol to stream (repeatable or comma-separated).",
)
_STREAM_CAPTURE_OPTIONS_CONTRACTS_OPT = typer.Option(
    None,
    "--options-contracts",
    "--option-contracts",
    help="Comma-separated option contract symbols to stream.",
)
_STREAM_CAPTURE_OPTIONS_UNDERLYING_OPT = typer.Option(
    None,
    "--options-underlying",
    "--option-underlying",
    "-u",
    help="Underlying to expand into option contracts using the contracts cache.",
)
_STREAM_CAPTURE_OPTIONS_UNDERLYINGS_OPT = typer.Option(
    None,
    "--options-underlyings",
    "--option-underlyings",
    help="Comma-separated underlyings to expand into option contracts using the contracts cache.",
)
_STREAM_CAPTURE_EXPIRY_OPT = typer.Option(
    None,
    "--expiry",
    help="Filter expanded contracts to expiry date(s) (repeatable or comma-separated).",
)
_STREAM_CAPTURE_EXPIRIES_OPT = typer.Option(
    None,
    "--expiries",
    help="Comma-separated expiries (YYYY-MM-DD).",
)
_STREAM_CAPTURE_CONTRACTS_DIR_OPT = typer.Option(
    Path("data/option_contracts"),
    "--contracts-dir",
    help="Directory containing option contracts cache.",
)
_STREAM_CAPTURE_CONTRACTS_AS_OF_OPT = typer.Option(
    "latest",
    "--contracts-as-of",
    help="Contract cache date to use (YYYY-MM-DD or 'latest').",
)
_STREAM_CAPTURE_MAX_CONTRACTS_OPT = typer.Option(
    250,
    "--max-contracts",
    min=1,
    max=5000,
    help="Max option contracts to stream after expansion (prevents accidental huge subscriptions).",
)
_STREAM_CAPTURE_BARS_OPT = typer.Option(
    True,
    "--bars/--no-bars",
    help="Capture stock bars (1Min).",
)
_STREAM_CAPTURE_QUOTES_OPT = typer.Option(
    False,
    "--quotes/--no-quotes",
    help="Capture stock/option quotes (tick data; can be large).",
)
_STREAM_CAPTURE_TRADES_OPT = typer.Option(
    False,
    "--trades/--no-trades",
    help="Capture stock/option trades (tick data; can be large).",
)
_STREAM_CAPTURE_DURATION_OPT = typer.Option(
    None,
    "--duration",
    min=0.0,
    help="Run capture for N seconds then exit (default: run until Ctrl+C).",
)
_STREAM_CAPTURE_FLUSH_SECONDS_OPT = typer.Option(
    10.0,
    "--flush-seconds",
    min=1.0,
    help="Flush buffered events to disk at least every N seconds.",
)
_STREAM_CAPTURE_FLUSH_EVERY_OPT = typer.Option(
    250,
    "--flush-every",
    min=1,
    max=500000,
    help="Flush buffered events every N events.",
)
_STREAM_CAPTURE_MAX_RECONNECTS_OPT = typer.Option(
    5,
    "--max-reconnects",
    min=0,
    max=50,
    help="Max reconnect attempts per stream before exiting.",
)
_STREAM_CAPTURE_STOCK_FEED_OPT = typer.Option(
    None,
    "--stock-feed",
    help="Alpaca stock feed override (defaults to OH_ALPACA_STOCK_FEED).",
)
_STREAM_CAPTURE_OPTIONS_FEED_OPT = typer.Option(
    None,
    "--options-feed",
    help="Alpaca options feed override (defaults to OH_ALPACA_OPTIONS_FEED).",
)
_STREAM_CAPTURE_OUT_DIR_OPT = typer.Option(
    Path("data/intraday"),
    "--out-dir",
    help="Base directory for intraday partitions.",
)


def _prepare_capture_targets(
    *,
    console: Console,
    stock: list[str] | None,
    stocks: str | None,
    option_contract: list[str] | None,
    option_contracts: str | None,
    option_underlying: list[str] | None,
    option_underlyings: str | None,
    expiry: list[str] | None,
    expiries: str | None,
    contracts_dir: Path,
    contracts_as_of: str,
    max_contracts: int,
) -> tuple[list[str], list[str]]:
    stock_symbols = _parse_symbols(stock, stocks)
    contracts_direct = _parse_contracts(option_contract, option_contracts)
    underlying_symbols = _parse_symbols(option_underlying, option_underlyings)
    expiry_dates = _parse_expiries(expiry, expiries)
    expanded_contracts: list[str] = []
    missing_contracts = 0
    if underlying_symbols:
        store = OptionContractsStore(contracts_dir)
        expanded_contracts, missing_contracts = _expand_underlyings_to_contracts(
            store,
            contracts_dir=contracts_dir,
            underlyings=underlying_symbols,
            as_of_spec=contracts_as_of,
            expiry_dates=expiry_dates,
        )

    all_contracts = sorted({*contracts_direct, *expanded_contracts})
    if len(all_contracts) > max_contracts:
        console.print(
            f"[yellow]Warning:[/yellow] Truncating option contracts from {len(all_contracts)} to {max_contracts}."
        )
        all_contracts = all_contracts[:max_contracts]
    if missing_contracts:
        console.print(f"[yellow]Warning:[/yellow] {missing_contracts} underlying(s) missing contracts cache.")
    if not stock_symbols and not all_contracts:
        raise typer.BadParameter("Provide at least one --stock/--stocks or option contract/underlying.")
    return stock_symbols, all_contracts


def _build_capture_runner(
    *,
    out_dir: Path,
    stock_symbols: list[str],
    option_contracts: list[str],
    bars: bool,
    quotes: bool,
    trades: bool,
    flush_seconds: float,
    flush_every: int,
    max_reconnects: int,
    stock_feed: str | None,
    options_feed: str | None,
) -> StreamRunner:
    return StreamRunner(
        out_dir=out_dir,
        stocks=stock_symbols,
        option_contracts=option_contracts,
        capture_bars=bool(bars),
        capture_quotes=bool(quotes),
        capture_trades=bool(trades),
        flush_interval_seconds=float(flush_seconds),
        flush_every_events=int(flush_every),
        max_reconnects=int(max_reconnects),
        stock_feed=stock_feed,
        options_feed=options_feed,
    )


def _execute_capture_run(
    *,
    console: Console,
    runner: StreamRunner,
    duration: float | None,
    stock_feed: str | None,
    options_feed: str | None,
) -> None:
    try:
        written = runner.run(duration_seconds=duration)
    except DataFetchError as exc:
        effective_stock_feed = stock_feed or os.getenv("OH_ALPACA_STOCK_FEED") or "unset"
        effective_options_feed = options_feed or os.getenv("OH_ALPACA_OPTIONS_FEED") or "unset"
        console.print(
            f"[red]Error:[/red] {exc} (stock_feed={effective_stock_feed}, options_feed={effective_options_feed})"
        )
        raise typer.Exit(code=1) from exc
    if written:
        console.print(f"[green]Wrote[/green] {len(written)} partition(s).")
        return
    console.print("[yellow]No partitions written.[/yellow]")


@app.command("capture")
def capture(
    stock: list[str] | None = _STREAM_CAPTURE_STOCK_OPT,
    stocks: str | None = _STREAM_CAPTURE_STOCKS_OPT,
    option_contract: list[str] | None = _STREAM_CAPTURE_OPTIONS_CONTRACT_OPT,
    option_contracts: str | None = _STREAM_CAPTURE_OPTIONS_CONTRACTS_OPT,
    option_underlying: list[str] | None = _STREAM_CAPTURE_OPTIONS_UNDERLYING_OPT,
    option_underlyings: str | None = _STREAM_CAPTURE_OPTIONS_UNDERLYINGS_OPT,
    expiry: list[str] | None = _STREAM_CAPTURE_EXPIRY_OPT,
    expiries: str | None = _STREAM_CAPTURE_EXPIRIES_OPT,
    contracts_dir: Path = _STREAM_CAPTURE_CONTRACTS_DIR_OPT,
    contracts_as_of: str = _STREAM_CAPTURE_CONTRACTS_AS_OF_OPT,
    max_contracts: int = _STREAM_CAPTURE_MAX_CONTRACTS_OPT,
    bars: bool = _STREAM_CAPTURE_BARS_OPT,
    quotes: bool = _STREAM_CAPTURE_QUOTES_OPT,
    trades: bool = _STREAM_CAPTURE_TRADES_OPT,
    duration: float | None = _STREAM_CAPTURE_DURATION_OPT,
    flush_seconds: float = _STREAM_CAPTURE_FLUSH_SECONDS_OPT,
    flush_every: int = _STREAM_CAPTURE_FLUSH_EVERY_OPT,
    max_reconnects: int = _STREAM_CAPTURE_MAX_RECONNECTS_OPT,
    stock_feed: str | None = _STREAM_CAPTURE_STOCK_FEED_OPT,
    options_feed: str | None = _STREAM_CAPTURE_OPTIONS_FEED_OPT,
    out_dir: Path = _STREAM_CAPTURE_OUT_DIR_OPT,
) -> None:
    """Run a streaming capture session and persist events under data/intraday/."""
    _require_alpaca_provider()
    console = Console()
    stock_symbols, all_contracts = _prepare_capture_targets(
        console=console,
        stock=stock,
        stocks=stocks,
        option_contract=option_contract,
        option_contracts=option_contracts,
        option_underlying=option_underlying,
        option_underlyings=option_underlyings,
        expiry=expiry,
        expiries=expiries,
        contracts_dir=contracts_dir,
        contracts_as_of=contracts_as_of,
        max_contracts=max_contracts,
    )
    runner = _build_capture_runner(
        out_dir=out_dir,
        stock_symbols=stock_symbols,
        option_contracts=all_contracts,
        bars=bars,
        quotes=quotes,
        trades=trades,
        flush_seconds=flush_seconds,
        flush_every=flush_every,
        max_reconnects=max_reconnects,
        stock_feed=stock_feed,
        options_feed=options_feed,
    )
    console.print(
        f"Streaming capture: stocks={len(stock_symbols)} options={len(all_contracts)} "
        f"bars={bars} quotes={quotes} trades={trades} out={out_dir}"
    )
    _execute_capture_run(
        console=console,
        runner=runner,
        duration=duration,
        stock_feed=stock_feed,
        options_feed=options_feed,
    )
