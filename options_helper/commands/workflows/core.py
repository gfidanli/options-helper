from __future__ import annotations

import json
from datetime import date, timezone
from pathlib import Path

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.commands.common import _parse_date
from options_helper.commands.position_metrics import _extract_float
from options_helper.commands import workflows_legacy as legacy
from options_helper.data.earnings import EarningsRecord
from options_helper.data.market_types import DataFetchError
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.reporting import render_summary
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


def _build_daily_performance_table():
    from rich.table import Table

    table = Table(title="Daily Performance (best-effort)")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Chg", justify="right")
    table.add_column("%Chg", justify="right")
    table.add_column("Daily PnL $", justify="right")
    return table


def _daily_pnl_style(daily_pnl: float | None) -> str | None:
    if daily_pnl is None:
        return None
    if daily_pnl > 0:
        return "green"
    if daily_pnl < 0:
        return "red"
    return None


def daily_performance(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """Estimate daily PnL for positions using best-effort quotes."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    provider = cli_deps.build_provider()
    table = _build_daily_performance_table()

    total_daily_pnl = 0.0
    total_prev_value = float(portfolio.cash)

    for position in portfolio.positions:
        try:
            chain = provider.get_options_chain(position.symbol, position.expiry)
            df = chain.calls if position.option_type == "call" else chain.puts
            row = contract_row_by_strike(df, position.strike)

            last = change = pct = None
            if row is not None:
                last = _extract_float(row, "lastPrice")
                change = _extract_float(row, "change")
                pct = _extract_float(row, "percentChange")

            quote = compute_daily_performance_quote(
                last_price=last,
                change=change,
                percent_change_raw=pct,
                contracts=position.contracts,
            )

            if quote.daily_pnl is not None:
                total_daily_pnl += quote.daily_pnl
            if quote.prev_close_price is not None:
                total_prev_value += quote.prev_close_price * 100.0 * position.contracts
            elif quote.last_price is not None:
                total_prev_value += quote.last_price * 100.0 * position.contracts

            table.add_row(
                position.id,
                position.symbol,
                position.expiry.isoformat(),
                f"{position.strike:g}",
                str(position.contracts),
                "-" if quote.last_price is None else f"${quote.last_price:.2f}",
                "-" if quote.change is None else f"{quote.change:+.2f}",
                "-" if quote.percent_change is None else f"{quote.percent_change:+.1f}%",
                "-" if quote.daily_pnl is None else f"{quote.daily_pnl:+.2f}",
                style=_daily_pnl_style(quote.daily_pnl),
            )
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")

    console.print(table)

    denom = total_prev_value if total_prev_value > 0 else None
    total_pct = (total_daily_pnl / denom) if denom else None
    total_str = f"{total_daily_pnl:+.2f}"
    pct_str = "-" if total_pct is None else f"{total_pct:+.2%}"
    console.print(f"\nTotal daily PnL: ${total_str} ({pct_str})")


def snapshot_options(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used to estimate spot).",
    ),
    window_pct: float = typer.Option(
        1.0,
        "--window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot in --windowed mode (e.g. 1.0 = +/-100%).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot price from daily candles.",
    ),
    require_data_date: str | None = typer.Option(
        None,
        "--require-data-date",
        help=(
            "Require each symbol's latest daily candle date to match this before snapshotting "
            "(YYYY-MM-DD or 'today'). If not met, the symbol is skipped to avoid mis-dated overwrites."
        ),
    ),
    require_data_tz: str = typer.Option(
        "America/Chicago",
        "--require-data-tz",
        help="Timezone used to interpret 'today' for --require-data-date (default: America/Chicago).",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist/--all-watchlists).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to snapshot (repeatable). When provided, snapshots those watchlists instead of portfolio positions.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Snapshot all watchlists in the watchlists store (instead of portfolio positions).",
    ),
    all_expiries: bool = typer.Option(
        True,
        "--all-expiries/--position-expiries",
        help=(
            "Snapshot all expiries per symbol (default). Use --position-expiries to restrict "
            "(portfolio: expiries in positions; watchlists: nearest expiries unless --max-expiries)."
        ),
    ),
    full_chain: bool = typer.Option(
        True,
        "--full-chain/--windowed",
        help=(
            "Snapshot the full Yahoo options payload per expiry (writes .raw.json + a full CSV). "
            "Disables strike-window filtering. Use --windowed for smaller flow-focused snapshots."
        ),
    ),
    max_expiries: int | None = typer.Option(
        None,
        "--max-expiries",
        min=1,
        help="Optional cap on expiries per symbol (nearest first). Useful for watchlists or --full-chain.",
    ),
    risk_free_rate: float = typer.Option(
        0.0,
        "--risk-free-rate",
        help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
    ),
) -> None:
    """Save a once-daily options chain snapshot (full-chain + all expiries by default)."""
    console = Console()

    import options_helper.commands.workflows as workflows_pkg

    with legacy._observed_run(
        console=console,
        job_name=legacy.JOB_SNAPSHOT_OPTIONS,
        args={
            "portfolio_path": str(portfolio_path),
            "cache_dir": str(cache_dir),
            "candle_cache_dir": str(candle_cache_dir),
            "window_pct": window_pct,
            "spot_period": spot_period,
            "require_data_date": require_data_date,
            "require_data_tz": require_data_tz,
            "watchlists_path": str(watchlists_path),
            "watchlist": watchlist,
            "all_watchlists": all_watchlists,
            "all_expiries": all_expiries,
            "full_chain": full_chain,
            "max_expiries": max_expiries,
            "risk_free_rate": risk_free_rate,
        },
    ) as run_logger:
        try:
            result = workflows_pkg.run_snapshot_options_job(
                portfolio_path=portfolio_path,
                cache_dir=cache_dir,
                candle_cache_dir=candle_cache_dir,
                window_pct=window_pct,
                spot_period=spot_period,
                require_data_date=require_data_date,
                require_data_tz=require_data_tz,
                watchlists_path=watchlists_path,
                watchlist=watchlist,
                all_watchlists=all_watchlists,
                all_expiries=all_expiries,
                full_chain=full_chain,
                max_expiries=max_expiries,
                risk_free_rate=risk_free_rate,
                provider_builder=cli_deps.build_provider,
                snapshot_store_builder=cli_deps.build_snapshot_store,
                candle_store_builder=cli_deps.build_candle_store,
                portfolio_loader=load_portfolio,
                watchlists_loader=load_watchlists,
            )
        except legacy.VisibilityJobParameterError as exc:
            if exc.param_hint:
                raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
            raise typer.BadParameter(str(exc)) from exc

        for message in result.messages:
            console.print(message)

        if result.no_symbols:
            run_logger.log_asset_skipped(
                asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                asset_kind="file",
                partition_key="ALL",
                extra={"reason": "no_symbols"},
            )
            raise typer.Exit(0)

        max_data_date = max(result.dates_used) if result.dates_used else None
        status_by_symbol = legacy._snapshot_status_by_symbol(symbols=result.symbols, messages=result.messages)

        for sym in result.symbols:
            status = status_by_symbol.get(sym.upper(), "skipped")
            if status == "success":
                run_logger.log_asset_success(
                    asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    min_event_ts=max_data_date,
                    max_event_ts=max_data_date,
                )
                if max_data_date is not None:
                    run_logger.upsert_watermark(
                        asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                        scope_key=sym.upper(),
                        watermark_ts=max_data_date,
                    )
            elif status == "failed":
                run_logger.log_asset_failure(
                    asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    extra={"reason": "snapshot_failed"},
                )
            else:
                run_logger.log_asset_skipped(
                    asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    extra={"reason": "snapshot_skipped"},
                )

        if max_data_date is not None:
            run_logger.upsert_watermark(
                asset_key=legacy.ASSET_OPTIONS_SNAPSHOTS,
                scope_key="ALL",
                watermark_ts=max_data_date,
            )


def earnings(
    symbol: str = typer.Argument(..., help="Ticker symbol (e.g. IREN)."),
    refresh: bool = typer.Option(False, "--refresh", help="Fetch from yfinance and update the local cache."),
    set_date: str | None = typer.Option(
        None,
        "--set",
        help="Manually set the next earnings date (YYYY-MM-DD). Overrides cached value.",
    ),
    clear: bool = typer.Option(False, "--clear", help="Delete cached earnings for the symbol."),
    cache_dir: Path = typer.Option(
        Path("data/earnings"),
        "--cache-dir",
        help="Directory for cached earnings dates.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the cached record as JSON."),
) -> None:
    """Show/cache the next earnings date (best-effort; Yahoo can be wrong/stale)."""
    console = Console(width=120)
    store = cli_deps.build_earnings_store(cache_dir)
    sym = symbol.upper().strip()

    if clear:
        deleted = store.delete(sym)
        console.print(f"Deleted: {sym}" if deleted else f"No cache found for: {sym}")
        return

    import options_helper.commands.workflows as workflows_pkg

    record: EarningsRecord | None = None

    if set_date is not None:
        parsed = _parse_date(set_date)
        record = EarningsRecord.manual(symbol=sym, next_earnings_date=parsed, note="Set via CLI --set.")
        out_path = store.save(record)
        console.print(f"Saved: {out_path}")
    else:
        record = store.load(sym)
        if refresh or record is None:
            # Earnings data is sourced from Yahoo via yfinance (best-effort).
            provider = cli_deps.build_provider("yahoo")
            try:
                event = provider.get_next_earnings_event(sym)
            except DataFetchError as exc:
                console.print(f"[red]Data error:[/red] {exc}")
                raise typer.Exit(1) from exc
            record = EarningsRecord(
                symbol=sym,
                fetched_at=workflows_pkg.datetime.now(tz=timezone.utc),
                source=event.source,
                next_earnings_date=event.next_date,
                window_start=event.window_start,
                window_end=event.window_end,
                raw=event.raw,
                notes=[],
            )
            out_path = store.save(record)
            console.print(f"Saved: {out_path}")

    if record is None:
        console.print(f"No earnings record for {sym}.")
        raise typer.Exit(1)

    if json_out:
        console.print_json(json.dumps(record.model_dump(mode="json"), sort_keys=True))
        return

    today = date.today()
    if record.next_earnings_date is None:
        console.print(f"{sym} next earnings date: [yellow]unknown[/yellow] (source={record.source})")
        return

    days = (record.next_earnings_date - today).days
    suffix = "today" if days == 0 else f"in {days} day(s)"
    console.print(f"{sym} next earnings: [bold]{record.next_earnings_date.isoformat()}[/bold] ({suffix})")
    if record.window_start or record.window_end:
        console.print(
            f"Earnings window: {record.window_start.isoformat() if record.window_start else '-'}"
            f" → {record.window_end.isoformat() if record.window_end else '-'}"
        )
    console.print(f"Source: {record.source} (fetched_at={record.fetched_at.isoformat()})")


def refresh_earnings(
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbols source).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to refresh (default: all watchlists).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/earnings"),
        "--cache-dir",
        help="Directory for cached earnings dates.",
    ),
) -> None:
    """Fetch and cache next earnings dates for symbols in watchlists (best-effort)."""
    console = Console(width=120)
    wl = load_watchlists(watchlists_path)

    symbols: set[str] = set()
    if watchlist:
        for name in watchlist:
            symbols.update(wl.get(name))
    else:
        for syms in (wl.watchlists or {}).values():
            symbols.update(syms or [])

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        console.print("No symbols found (no watchlists or empty watchlist selection).")
        raise typer.Exit(0)

    import options_helper.commands.workflows as workflows_pkg

    store = cli_deps.build_earnings_store(cache_dir)
    # Earnings data is sourced from Yahoo via yfinance (best-effort).
    provider = cli_deps.build_provider("yahoo")

    ok = 0
    err = 0
    unknown = 0

    console.print(f"Refreshing earnings for {len(symbols)} symbol(s)...")
    for sym in sorted(symbols):
        try:
            event = provider.get_next_earnings_event(sym)
            record = EarningsRecord(
                symbol=sym,
                fetched_at=workflows_pkg.datetime.now(tz=timezone.utc),
                source=event.source,
                next_earnings_date=event.next_date,
                window_start=event.window_start,
                window_end=event.window_end,
                raw=event.raw,
                notes=[],
            )
            path = store.save(record)
            ok += 1
            if record.next_earnings_date is None:
                unknown += 1
                console.print(f"[yellow]Warning:[/yellow] {sym}: next earnings unknown (saved {path})")
            else:
                console.print(f"{sym}: {record.next_earnings_date.isoformat()} (saved {path})")
        except DataFetchError as exc:
            err += 1
            console.print(f"[red]Error:[/red] {sym}: {exc}")
        except Exception as exc:  # noqa: BLE001
            err += 1
            console.print(f"[red]Error:[/red] {sym}: {exc}")

    console.print(f"Done. ok={ok} unknown={unknown} errors={err}")


__all__ = [
    "daily_performance",
    "snapshot_options",
    "earnings",
    "refresh_earnings",
]
