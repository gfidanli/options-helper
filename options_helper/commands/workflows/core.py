from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.commands.position_metrics import _extract_float
from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.compat import sync_legacy_seams
from options_helper.data.market_types import DataFetchError
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.reporting import render_summary
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


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
                style=(
                    "green"
                    if quote.daily_pnl and quote.daily_pnl > 0
                    else "red"
                    if quote.daily_pnl and quote.daily_pnl < 0
                    else None
                ),
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


@wraps(legacy.earnings)
def earnings(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.earnings(*args, **kwargs)


@wraps(legacy.refresh_earnings)
def refresh_earnings(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.refresh_earnings(*args, **kwargs)


__all__ = [
    "daily_performance",
    "snapshot_options",
    "earnings",
    "refresh_earnings",
]
