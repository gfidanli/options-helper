from __future__ import annotations

from functools import wraps
from pathlib import Path
import time
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


def analyze(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    period: str = typer.Option("2y", "--period", help="Underlying history period (yfinance)."),
    interval: str = typer.Option("1d", "--interval", help="Underlying history interval (yfinance)."),
    offline: bool = typer.Option(
        False,
        "--offline/--online",
        help="Run from local snapshots + candle cache for deterministic as-of outputs.",
    ),
    as_of: str = typer.Option(
        "latest",
        "--as-of",
        help="Snapshot date (YYYY-MM-DD) or 'latest' (used with --offline).",
    ),
    offline_strict: bool = typer.Option(
        False,
        "--offline-strict",
        help="Fail if any position is missing snapshot coverage (used with --offline).",
    ),
    snapshots_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshots-dir",
        help="Directory containing options snapshot folders (used with --offline).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Directory for locally cached daily candles.",
    ),
    stress_spot_pct: list[float] = typer.Option(
        [],
        "--stress-spot-pct",
        help="Spot shock percent (repeatable). Use 5 for 5%% or 0.05 for 5%%.",
    ),
    stress_vol_pp: float = typer.Option(
        5.0,
        "--stress-vol-pp",
        help="Volatility shock in IV points (e.g. 5 = 5pp). Set to 0 to disable.",
    ),
    stress_days: int = typer.Option(
        7,
        "--stress-days",
        help="Time decay stress in days. Set to 0 to disable.",
    ),
) -> None:
    """Fetch data and print metrics + rule-based advice."""
    legacy._ensure_pandas()
    pd = legacy.pd
    assert pd is not None

    if interval != "1d":
        raise typer.BadParameter("Only --interval 1d is supported for now (cache uses daily candles).")

    import options_helper.commands.workflows as workflows_pkg

    portfolio = load_portfolio(portfolio_path)
    console = Console()
    legacy.render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    snapshot_store = None
    if offline:
        snapshot_store = cli_deps.build_snapshot_store(snapshots_dir)

    provider = None if offline else cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(cache_dir, provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))

    history_by_symbol: dict[str, Any] = {}
    last_price_by_symbol: dict[str, float | None] = {}
    as_of_by_symbol: dict[str, Any] = {}
    next_earnings_by_symbol: dict[str, Any] = {}
    snapshot_day_by_symbol: dict[str, Any] = {}
    chain_cache: dict[tuple[str, Any], object] = {}
    for sym in sorted({position.symbol for position in portfolio.positions}):
        history = pd.DataFrame()
        snapshot_date = None

        if offline:
            try:
                history = candle_store.load(sym)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] candle cache read failed for {sym}: {exc}")
                history = pd.DataFrame()

            if snapshot_store is not None:
                try:
                    snapshot_date = snapshot_store.resolve_date(sym, as_of)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow]Warning:[/yellow] snapshot date resolve failed for {sym}: {exc}")
                    snapshot_date = None

            if snapshot_date is None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                last_ts = history.index.max()
                if last_ts is not None and not pd.isna(last_ts):
                    snapshot_date = last_ts.date()

            if snapshot_date is not None and not history.empty and isinstance(history.index, pd.DatetimeIndex):
                history = history.loc[history.index <= pd.Timestamp(snapshot_date)]

            snap_df = pd.DataFrame()
            if snapshot_store is not None and snapshot_date is not None:
                try:
                    snap_df = snapshot_store.load_day(sym, snapshot_date)
                except Exception as exc:  # noqa: BLE001
                    console.print(f"[yellow]Warning:[/yellow] snapshot load failed for {sym}: {exc}")
                    snap_df = pd.DataFrame()
            snapshot_day_by_symbol[sym] = snap_df
        else:
            try:
                history = candle_store.get_daily_history(sym, period=period)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Candle cache error:[/red] {sym}: {exc}")
                history = pd.DataFrame()

        history_by_symbol[sym] = history

        if offline:
            as_of_by_symbol[sym] = snapshot_date
            last_price_by_symbol[sym] = (
                legacy.close_asof(history, snapshot_date) if snapshot_date is not None else legacy.last_close(history)
            )
        else:
            last_price_by_symbol[sym] = legacy.last_close(history)
            as_of_date = None
            if not history.empty and isinstance(history.index, pd.DatetimeIndex):
                last_ts = history.index.max()
                if last_ts is not None and not pd.isna(last_ts):
                    as_of_date = last_ts.date()
            as_of_by_symbol[sym] = as_of_date

        next_earnings_by_symbol[sym] = workflows_pkg.safe_next_earnings_date(earnings_store, sym)

    single_metrics: list[Any] = []
    all_metrics: list[Any] = []
    multi_leg_summaries: list[Any] = []
    advice_by_id: dict[str, Any] = {}
    offline_missing: list[str] = []

    for position in portfolio.positions:
        try:
            if isinstance(position, legacy.MultiLegPosition):
                leg_metrics: list[Any] = []
                net_mark_total = 0.0
                net_mark_ready = True
                dte_vals: list[int] = []
                warnings: list[str] = []
                low_oi = False
                low_vol = False
                bad_spread = False
                quote_flags = False
                missing_as_of_warned = False
                missing_day_warned = False

                for idx, leg in enumerate(position.legs, start=1):
                    if offline:
                        snap_date = as_of_by_symbol.get(position.symbol)
                        df_snap = snapshot_day_by_symbol.get(position.symbol, pd.DataFrame())
                        row = None

                        if snap_date is None:
                            if not missing_as_of_warned:
                                offline_missing.append(f"{position.id}: missing offline as-of date for {position.symbol}")
                                missing_as_of_warned = True
                        elif df_snap.empty:
                            if not missing_day_warned:
                                offline_missing.append(
                                    f"{position.id}: missing snapshot day data for {position.symbol} (as-of {snap_date.isoformat()})"
                                )
                                missing_day_warned = True
                        else:
                            row = legacy.find_snapshot_row(
                                df_snap,
                                expiry=leg.expiry,
                                strike=leg.strike,
                                option_type=leg.option_type,
                            )
                            if row is None:
                                offline_missing.append(
                                    f"{position.id}: missing snapshot row for {position.symbol} {leg.expiry.isoformat()} "
                                    f"{leg.option_type} {leg.strike:g} (as-of {snap_date.isoformat()})"
                                )

                        snapshot_row = row if row is not None else {}
                    else:
                        row = None
                        if provider is not None:
                            key = (position.symbol, leg.expiry)
                            chain = chain_cache.get(key)
                            if chain is None:
                                chain = provider.get_options_chain(position.symbol, leg.expiry)
                                chain_cache[key] = chain
                            df_chain = chain.calls if leg.option_type == "call" else chain.puts
                            row = legacy.contract_row_by_strike(df_chain, leg.strike)
                        snapshot_row = row if row is not None else {}

                    leg_position = legacy.Position(
                        id=f"{position.id}:leg{idx}",
                        symbol=position.symbol,
                        option_type=leg.option_type,
                        expiry=leg.expiry,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        cost_basis=0.0,
                        opened_at=position.opened_at,
                    )

                    metrics = workflows_pkg._position_metrics(
                        provider,
                        leg_position,
                        risk_profile=portfolio.risk_profile,
                        underlying_history=history_by_symbol.get(position.symbol, pd.DataFrame()),
                        underlying_last_price=last_price_by_symbol.get(position.symbol),
                        as_of=as_of_by_symbol.get(position.symbol),
                        next_earnings_date=next_earnings_by_symbol.get(position.symbol),
                        snapshot_row=snapshot_row,
                        include_pnl=False,
                        contract_sign=1 if leg.side == "long" else -1,
                    )
                    leg_metrics.append(metrics)
                    all_metrics.append(metrics)

                    if metrics.mark is None:
                        net_mark_ready = False
                    else:
                        net_mark_total += metrics.mark * leg.signed_contracts * 100.0

                    if metrics.dte is not None:
                        dte_vals.append(metrics.dte)
                    if (
                        metrics.open_interest is not None
                        and metrics.open_interest < portfolio.risk_profile.min_open_interest
                    ):
                        low_oi = True
                    if metrics.volume is not None and metrics.volume < portfolio.risk_profile.min_volume:
                        low_vol = True
                    if metrics.execution_quality == "bad":
                        bad_spread = True
                    if metrics.quality_warnings:
                        quote_flags = True

                net_mark = net_mark_total if net_mark_ready else None
                if net_mark is None:
                    warnings.append("missing_leg_marks")
                if position.net_debit is None:
                    warnings.append("missing_net_debit")
                if low_oi:
                    warnings.append("low_open_interest_leg")
                if low_vol:
                    warnings.append("low_volume_leg")
                if bad_spread:
                    warnings.append("bad_spread_leg")
                if quote_flags:
                    warnings.append("quote_quality_leg")

                net_pnl_abs = net_pnl_pct = None
                if net_mark is not None and position.net_debit is not None:
                    net_pnl_abs = net_mark - position.net_debit
                    if position.net_debit > 0:
                        net_pnl_pct = net_pnl_abs / position.net_debit

                dte_min = min(dte_vals) if dte_vals else None
                dte_max = max(dte_vals) if dte_vals else None

                multi_leg_summaries.append(
                    legacy.MultiLegSummary(
                        position=position,
                        leg_metrics=leg_metrics,
                        net_mark=net_mark,
                        net_pnl_abs=net_pnl_abs,
                        net_pnl_pct=net_pnl_pct,
                        dte_min=dte_min,
                        dte_max=dte_max,
                        warnings=warnings,
                    )
                )
                continue

            if offline:
                snap_date = as_of_by_symbol.get(position.symbol)
                df_snap = snapshot_day_by_symbol.get(position.symbol, pd.DataFrame())
                row = None

                if snap_date is None:
                    offline_missing.append(f"{position.id}: missing offline as-of date for {position.symbol}")
                elif df_snap.empty:
                    offline_missing.append(
                        f"{position.id}: missing snapshot day data for {position.symbol} (as-of {snap_date.isoformat()})"
                    )
                else:
                    row = legacy.find_snapshot_row(
                        df_snap,
                        expiry=position.expiry,
                        strike=position.strike,
                        option_type=position.option_type,
                    )
                    if row is None:
                        offline_missing.append(
                            f"{position.id}: missing snapshot row for {position.symbol} {position.expiry.isoformat()} "
                            f"{position.option_type} {position.strike:g} (as-of {snap_date.isoformat()})"
                        )

                snapshot_row = row if row is not None else {}
            else:
                snapshot_row = None

            metrics = workflows_pkg._position_metrics(
                provider,
                position,
                risk_profile=portfolio.risk_profile,
                underlying_history=history_by_symbol.get(position.symbol, pd.DataFrame()),
                underlying_last_price=last_price_by_symbol.get(position.symbol),
                as_of=as_of_by_symbol.get(position.symbol),
                next_earnings_date=next_earnings_by_symbol.get(position.symbol),
                snapshot_row=snapshot_row,
            )
            single_metrics.append(metrics)
            all_metrics.append(metrics)
            advice_by_id[position.id] = legacy.advise(metrics, portfolio)
        except legacy.DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")

    if not single_metrics and not multi_leg_summaries:
        raise typer.Exit(1)

    if single_metrics:
        legacy.render_positions(console, portfolio, single_metrics, advice_by_id)
    if multi_leg_summaries:
        legacy.render_multi_leg_positions(console, multi_leg_summaries)

    exposure = legacy.compute_portfolio_exposure(all_metrics)
    legacy._render_portfolio_risk(
        console,
        exposure,
        stress_spot_pct=stress_spot_pct,
        stress_vol_pp=stress_vol_pp,
        stress_days=stress_days,
    )

    if offline_missing:
        for msg in offline_missing:
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        if offline_strict:
            raise typer.Exit(1)


def watch(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    minutes: int = typer.Option(15, "--minutes", help="Polling interval in minutes."),
) -> None:
    """Continuously re-run analysis at a fixed interval."""
    if minutes <= 0:
        raise typer.BadParameter("--minutes must be > 0")

    console = Console()
    console.print(f"Watching {portfolio_path} every {minutes} minute(s). Ctrl+C to stop.")
    while True:
        try:
            analyze(portfolio_path)
        except typer.Exit:
            pass
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Watch error:[/red] {exc}")
        time.sleep(minutes * 60)


__all__ = ["research", "refresh_candles", "analyze", "watch"]
