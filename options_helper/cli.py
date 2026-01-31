from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd
import typer
from rich.console import Console

from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.chain_metrics import compute_chain_report
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.research import (
    Direction,
    analyze_underlying,
    choose_expiry,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain, black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.data.candles import CandleStore, last_close
from options_helper.data.derived import DERIVED_COLUMNS_V1, DERIVED_SCHEMA_VERSION, DerivedStore
from options_helper.data.earnings import EarningsRecord, EarningsStore
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.technical_backtesting_artifacts import write_artifacts
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path
from options_helper.data.yf_client import DataFetchError, YFinanceClient, contract_row_by_strike
from options_helper.models import OptionType, Position, RiskProfile
from options_helper.reporting import render_positions, render_summary
from options_helper.reporting_chain import (
    render_chain_report_console,
    render_chain_report_markdown,
    render_compare_report_console,
)
from options_helper.reporting_briefing import BriefingSymbolSection, render_briefing_markdown
from options_helper.reporting_roll import render_roll_plan_console
from options_helper.storage import load_portfolio, save_portfolio, write_template
from options_helper.watchlists import build_default_watchlists, load_watchlists, save_watchlists
from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
from options_helper.technicals_backtesting.strategies.registry import get_strategy

app = typer.Typer(add_completion=False)
watchlists_app = typer.Typer(help="Manage symbol watchlists.")
app.add_typer(watchlists_app, name="watchlists")
derived_app = typer.Typer(help="Persist derived metrics from local snapshots.")
app.add_typer(derived_app, name="derived")
technicals_app = typer.Typer(help="Technical indicators + backtesting/optimization.")
app.add_typer(technicals_app, name="technicals")


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


def _spot_from_meta(meta: dict) -> float | None:
    if not meta:
        return None
    candidates = [
        meta.get("spot"),
        (meta.get("underlying") or {}).get("regularMarketPrice"),
        (meta.get("underlying") or {}).get("regularMarketPreviousClose"),
        (meta.get("underlying") or {}).get("regularMarketOpen"),
    ]
    for v in candidates:
        try:
            if v is None:
                continue
            spot = float(v)
            if spot > 0:
                return spot
        except Exception:  # noqa: BLE001
            continue
    return None


def _default_position_id(symbol: str, expiry: date, strike: float, option_type: OptionType) -> str:
    suffix = "c" if option_type == "call" else "p"
    strike_str = f"{strike:g}".replace(".", "p")
    return f"{symbol.lower()}-{expiry.isoformat()}-{strike_str}{suffix}"


def _extract_float(row, key: str) -> float | None:
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return float(val)
    except Exception:  # noqa: BLE001
        return None


def _extract_int(row, key: str) -> int | None:
    if key not in row:
        return None
    val = row[key]
    try:
        if val is None:
            return None
        return int(val)
    except Exception:  # noqa: BLE001
        return None


def _mark_price(*, bid: float | None, ask: float | None, last: float | None) -> float | None:
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    if last is not None and last > 0:
        return last
    if ask is not None and ask > 0:
        return ask
    if bid is not None and bid > 0:
        return bid
    return None


@derived_app.command("update")
def derived_update(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to update."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (writes {derived_dir}/{SYMBOL}.csv).",
    ),
) -> None:
    """Append or upsert a derived-metrics row for a symbol/day (offline)."""
    console = Console()
    store = OptionsSnapshotStore(cache_dir)
    derived = DerivedStore(derived_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode="near",
            top=10,
            best_effort=True,
        )

        row = DerivedRow.from_chain_report(report)
        out_path = derived.upsert(symbol, row)
        console.print(f"Derived schema v{DERIVED_SCHEMA_VERSION} updated: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@derived_app.command("show")
def derived_show(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to show."),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    last: int = typer.Option(30, "--last", min=1, max=3650, help="Show the last N rows."),
) -> None:
    """Print the last N rows of derived metrics for a symbol."""
    from rich.table import Table

    console = Console(width=200)
    derived = DerivedStore(derived_dir)

    try:
        df = derived.load(symbol)
        if df.empty:
            console.print(f"No derived rows found for {symbol.upper()} in {derived_dir}")
            raise typer.Exit(1)

        tail = df.tail(last)
        t = Table(title=f"{symbol.upper()} derived metrics (last {min(last, len(df))})")
        for col in tail.columns:
            t.add_column(col)
        for _, row in tail.iterrows():
            t.add_row(*["" if pd.isna(v) else str(v) for v in row.tolist()])
        console.print(t)
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@derived_app.command("stats")
def derived_stats(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to analyze."),
    as_of: str = typer.Option("latest", "--as-of", help="Derived date (YYYY-MM-DD) or 'latest'."),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (reads {derived_dir}/{SYMBOL}.csv).",
    ),
    window: int = typer.Option(60, "--window", min=1, max=3650, help="Lookback window for percentiles."),
    trend_window: int = typer.Option(5, "--trend-window", min=1, max=3650, help="Lookback window for trend flags."),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/derived/{SYMBOL}/).",
    ),
) -> None:
    """Percentile ranks and trend flags from the derived-metrics history (offline)."""
    from rich.table import Table

    console = Console(width=200)
    derived = DerivedStore(derived_dir)

    try:
        df = derived.load(symbol)
        if df.empty:
            console.print(f"No derived rows found for {symbol.upper()} in {derived_dir}")
            raise typer.Exit(1)

        fmt = format.strip().lower()
        if fmt not in {"console", "json"}:
            raise typer.BadParameter("Invalid --format (use console|json)", param_hint="--format")

        report = compute_derived_stats(
            df,
            symbol=symbol,
            as_of=as_of,
            window=window,
            trend_window=trend_window,
            metric_columns=[c for c in DERIVED_COLUMNS_V1 if c != "date"],
        )

        if fmt == "json":
            console.print(report.model_dump_json(indent=2))
        else:
            t = Table(
                title=f"{report.symbol} derived stats (as-of {report.as_of}; pct w={window}; trend w={trend_window})"
            )
            t.add_column("metric")
            t.add_column("value", justify="right")
            t.add_column(f"pct({window})", justify="right")
            t.add_column(f"trend({trend_window})", justify="right")
            t.add_column("Δ", justify="right")
            t.add_column("Δ%", justify="right")

            for m in report.metrics:
                value = "" if m.value is None else f"{m.value:.8g}"
                pct = "" if m.percentile is None else f"{m.percentile:.1f}"
                delta = "" if m.trend_delta is None else f"{m.trend_delta:.8g}"
                delta_pct = "" if m.trend_delta_pct is None else f"{m.trend_delta_pct:.2f}"
                trend = "" if m.trend_direction is None else m.trend_direction
                t.add_row(m.name, value, pct, trend, delta, delta_pct)

            console.print(t)
            if report.warnings:
                console.print(f"[yellow]Warnings:[/yellow] {', '.join(report.warnings)}")

        if out is not None:
            base = out / "derived" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{report.as_of}_w{window}_tw{trend_window}.json"
            out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except typer.Exit:
        raise
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


def _position_metrics(
    client: YFinanceClient,
    position: Position,
    *,
    risk_profile: RiskProfile,
    underlying_history: pd.DataFrame,
    underlying_last_price: float | None,
) -> PositionMetrics:
    today = date.today()
    chain = client.get_options_chain(position.symbol, position.expiry)
    df = chain.calls if position.option_type == "call" else chain.puts
    row = contract_row_by_strike(df, position.strike)

    bid = ask = last = iv = None
    oi = vol = None
    if row is not None:
        bid = _extract_float(row, "bid")
        ask = _extract_float(row, "ask")
        last = _extract_float(row, "lastPrice")
        iv = _extract_float(row, "impliedVolatility")
        oi = _extract_int(row, "openInterest")
        vol = _extract_int(row, "volume")

    mark = _mark_price(bid=bid, ask=ask, last=last)

    dte = (position.expiry - today).days
    dte_val = dte if dte >= 0 else 0

    underlying_price = underlying_last_price
    moneyness = None
    if underlying_price is not None:
        moneyness = underlying_price / position.strike

    pnl_abs = pnl_pct = None
    if mark is not None:
        pnl_abs = (mark - position.cost_basis) * 100.0 * position.contracts
        pnl_pct = None if position.cost_basis <= 0 else (mark - position.cost_basis) / position.cost_basis

    close_series: pd.Series | None = None
    volume_series: pd.Series | None = None
    if not underlying_history.empty and "Close" in underlying_history.columns:
        close_series = underlying_history["Close"].dropna()
    if not underlying_history.empty and "Volume" in underlying_history.columns:
        volume_series = underlying_history["Volume"].dropna()

    sma20 = sma(close_series, 20) if close_series is not None else None
    sma50 = sma(close_series, 50) if close_series is not None else None
    rsi14 = rsi(close_series, 14) if close_series is not None else None
    ema20 = ema(close_series, 20) if close_series is not None else None
    ema50 = ema(close_series, 50) if close_series is not None else None

    close_3d = rsi14_3d = ema20_3d = ema50_3d = None
    close_w = rsi14_w = ema20_w = ema50_w = None
    breakout_w = None
    near_support_w = None

    if close_series is not None and isinstance(close_series.index, pd.DatetimeIndex):
        close_3d_series = close_series.resample("3B").last().dropna()
        close_w_series = close_series.resample("W-FRI").last().dropna()

        close_3d = float(close_3d_series.iloc[-1]) if not close_3d_series.empty else None
        close_w = float(close_w_series.iloc[-1]) if not close_w_series.empty else None

        rsi14_3d = rsi(close_3d_series, 14) if not close_3d_series.empty else None
        ema20_3d = ema(close_3d_series, 20) if not close_3d_series.empty else None
        ema50_3d = ema(close_3d_series, 50) if not close_3d_series.empty else None

        rsi14_w = rsi(close_w_series, 14) if not close_w_series.empty else None
        ema20_w = ema(close_w_series, 20) if not close_w_series.empty else None
        ema50_w = ema(close_w_series, 50) if not close_w_series.empty else None

        if close_w is not None and ema50_w is not None and ema50_w != 0:
            near_support_w = abs(close_w - ema50_w) / abs(ema50_w) <= risk_profile.support_proximity_pct

        lookback = risk_profile.breakout_lookback_weeks
        breakout_price = (
            breakout_up(close_w_series, lookback)
            if position.option_type == "call"
            else breakout_down(close_w_series, lookback)
        )

        breakout_vol_ok = True
        if (
            breakout_price is True
            and volume_series is not None
            and isinstance(volume_series.index, pd.DatetimeIndex)
            and risk_profile.breakout_volume_mult > 0
        ):
            vol_w_series = volume_series.resample("W-FRI").sum().dropna()
            if len(vol_w_series) >= lookback + 1:
                last_vol = float(vol_w_series.iloc[-1])
                prev_avg = float(vol_w_series.iloc[-(lookback + 1) : -1].mean())
                if prev_avg > 0:
                    breakout_vol_ok = last_vol >= prev_avg * risk_profile.breakout_volume_mult

        if breakout_price is not None:
            breakout_w = bool(breakout_price and breakout_vol_ok)

    delta = theta_per_day = None
    if underlying_price is not None and iv is not None and dte_val > 0:
        greeks = black_scholes_greeks(
            option_type=position.option_type,
            s=underlying_price,
            k=position.strike,
            t_years=dte_val / 365.0,
            sigma=iv,
        )
        if greeks is not None:
            delta = greeks.delta
            theta_per_day = greeks.theta_per_day

    return PositionMetrics(
        position=position,
        underlying_price=underlying_price,
        mark=mark,
        bid=bid,
        ask=ask,
        last=last,
        implied_vol=iv,
        open_interest=oi,
        volume=vol,
        dte=dte_val,
        moneyness=moneyness,
        pnl_abs=pnl_abs,
        pnl_pct=pnl_pct,
        sma20=sma20,
        sma50=sma50,
        rsi14=rsi14,
        ema20=ema20,
        ema50=ema50,
        close_3d=close_3d,
        rsi14_3d=rsi14_3d,
        ema20_3d=ema20_3d,
        ema50_3d=ema50_3d,
        close_w=close_w,
        rsi14_w=rsi14_w,
        ema20_w=ema20_w,
        ema50_w=ema50_w,
        near_support_w=near_support_w,
        breakout_w=breakout_w,
        delta=delta,
        theta_per_day=theta_per_day,
    )


@app.command("daily")
def daily_performance(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """Show best-effort daily P&L for the portfolio (based on options chain change fields)."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    client = YFinanceClient()

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

    for p in portfolio.positions:
        try:
            chain = client.get_options_chain(p.symbol, p.expiry)
            df = chain.calls if p.option_type == "call" else chain.puts
            row = contract_row_by_strike(df, p.strike)

            last = change = pct = None
            if row is not None:
                last = _extract_float(row, "lastPrice")
                change = _extract_float(row, "change")
                pct = _extract_float(row, "percentChange")

            q = compute_daily_performance_quote(
                last_price=last,
                change=change,
                percent_change_raw=pct,
                contracts=p.contracts,
            )

            if q.daily_pnl is not None:
                total_daily_pnl += q.daily_pnl
            if q.prev_close_price is not None:
                total_prev_value += q.prev_close_price * 100.0 * p.contracts
            elif q.last_price is not None:
                total_prev_value += q.last_price * 100.0 * p.contracts

            table.add_row(
                p.id,
                p.symbol,
                p.expiry.isoformat(),
                f"{p.strike:g}",
                str(p.contracts),
                "-" if q.last_price is None else f"${q.last_price:.2f}",
                "-" if q.change is None else f"{q.change:+.2f}",
                "-" if q.percent_change is None else f"{q.percent_change:+.1f}%",
                "-" if q.daily_pnl is None else f"{q.daily_pnl:+.2f}",
                style="green" if q.daily_pnl and q.daily_pnl > 0 else "red" if q.daily_pnl and q.daily_pnl < 0 else None,
            )
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")

    console.print(table)

    denom = total_prev_value if total_prev_value > 0 else None
    total_pct = (total_daily_pnl / denom) if denom else None
    total_str = f"{total_daily_pnl:+.2f}"
    pct_str = "-" if total_pct is None else f"{total_pct:+.2%}"
    console.print(f"\nTotal daily PnL: ${total_str} ({pct_str})")


@app.command("snapshot-options")
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
        help="Strike window around spot (e.g. 1.0 = +/-100%).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot price from daily candles.",
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
        False,
        "--all-expiries",
        help="Snapshot all expiries per symbol (instead of just expiries in portfolio positions).",
    ),
    full_chain: bool = typer.Option(
        False,
        "--full-chain",
        help="Snapshot the full Yahoo options payload per expiry (writes .raw.json + a full CSV). Implies --all-expiries and disables strike-window filtering.",
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
    """Save a once-daily options chain snapshot (windowed around spot by default) for flow analysis."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = OptionsSnapshotStore(cache_dir)
    candle_store = CandleStore(candle_cache_dir)
    client = YFinanceClient()

    want_full_chain = full_chain
    want_all_expiries = all_expiries or want_full_chain

    use_watchlists = bool(watchlist) or all_watchlists

    watchlists_used: list[str] = []
    expiries_by_symbol: dict[str, set[date]] = {}
    symbols: list[str]

    if use_watchlists:
        wl = load_watchlists(watchlists_path)
        if all_watchlists:
            watchlists_used = sorted(wl.watchlists.keys())
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                console.print(f"No watchlists in {watchlists_path}")
                raise typer.Exit(0)
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise typer.BadParameter(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
            watchlists_used = sorted(set(watchlist))
    else:
        if not portfolio.positions:
            console.print("No positions.")
            raise typer.Exit(0)
        for p in portfolio.positions:
            expiries_by_symbol.setdefault(p.symbol, set()).add(p.expiry)
        symbols = sorted(expiries_by_symbol.keys())

    # Snapshot folder date should reflect the data period (latest available daily candle),
    # not the wall-clock run date. This matters for pre-market runs where the latest
    # daily candle is still yesterday's close.
    dates_used: set[date] = set()

    mode = "watchlists" if use_watchlists else "portfolio"
    console.print(
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full' if want_full_chain else 'windowed'})..."
    )

    # If the user snapshots watchlists without explicitly asking for all expiries/full-chain,
    # default to the nearest couple expiries to keep runtime and storage sane.
    effective_max_expiries = max_expiries
    if use_watchlists and not want_all_expiries and effective_max_expiries is None:
        effective_max_expiries = 2

    for symbol in symbols:
        history = candle_store.get_daily_history(symbol, period=spot_period)
        spot = last_close(history)
        data_date: date | None = history.index.max().date() if not history.empty else None
        if spot is None:
            try:
                underlying = client.get_underlying(symbol, period=spot_period, interval="1d")
                spot = underlying.last_price
                if data_date is None and underlying.history is not None and not underlying.history.empty:
                    try:
                        data_date = underlying.history.index.max().date()
                    except Exception:  # noqa: BLE001
                        pass
            except DataFetchError:
                spot = None

        if spot is None or spot <= 0:
            console.print(f"[yellow]Warning:[/yellow] {symbol}: missing spot price; skipping snapshot.")
            continue

        effective_snapshot_date = data_date or date.today()
        dates_used.add(effective_snapshot_date)

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        meta = {
            "spot": spot,
            "spot_period": spot_period,
            "full_chain": want_full_chain,
            "all_expiries": want_all_expiries,
            "risk_free_rate": risk_free_rate,
            "window_pct": None if want_full_chain else window_pct,
            "strike_min": None if want_full_chain else strike_min,
            "strike_max": None if want_full_chain else strike_max,
            "snapshot_date": effective_snapshot_date.isoformat(),
            "symbol_source": mode,
            "watchlists": watchlists_used,
        }

        expiries: list[date]
        if not use_watchlists and not want_all_expiries:
            expiries = sorted(expiries_by_symbol.get(symbol, set()))
        else:
            expiry_strs = list(client.ticker(symbol).options or [])
            if not expiry_strs:
                console.print(f"[yellow]Warning:[/yellow] {symbol}: no listed option expiries; skipping snapshot.")
                continue
            if effective_max_expiries is not None:
                expiry_strs = expiry_strs[:effective_max_expiries]
            expiries = [date.fromisoformat(s) for s in expiry_strs]

        for exp in expiries:
            if want_full_chain:
                try:
                    raw = client.get_options_chain_raw(symbol, exp)
                except DataFetchError as exc:
                    console.print(
                        f"[yellow]Warning:[/yellow] {symbol} {exp.isoformat()}: {exc}; skipping snapshot."
                    )
                    continue

                # Capture the full payload (raw) + a denormalized CSV for convenience.
                meta_with_underlying = dict(meta)
                meta_with_underlying["underlying"] = raw.get("underlying", {})

                calls = pd.DataFrame(raw.get("calls", []))
                puts = pd.DataFrame(raw.get("puts", []))
                calls["optionType"] = "call"
                puts["optionType"] = "put"
                calls["expiry"] = exp.isoformat()
                puts["expiry"] = exp.isoformat()

                df = pd.concat([calls, puts], ignore_index=True)
                df = add_black_scholes_greeks_to_chain(
                    df,
                    spot=spot,
                    expiry=exp,
                    as_of=effective_snapshot_date,
                    r=risk_free_rate,
                )

                store.save_expiry_snapshot(
                    symbol,
                    effective_snapshot_date,
                    expiry=exp,
                    snapshot=df,
                    meta=meta_with_underlying,
                )
                store.save_expiry_snapshot_raw(
                    symbol,
                    effective_snapshot_date,
                    expiry=exp,
                    raw=raw,
                )
                console.print(f"{symbol} {exp.isoformat()}: saved {len(df)} contracts (full)")
                continue

            # Default: windowed flow snapshot (compact columns).
            try:
                chain = client.get_options_chain(symbol, exp)
            except DataFetchError as exc:
                console.print(
                    f"[yellow]Warning:[/yellow] {symbol} {exp.isoformat()}: {exc}; skipping snapshot."
                )
                continue

            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["optionType"] = "call"
            puts["optionType"] = "put"
            calls["expiry"] = exp.isoformat()
            puts["expiry"] = exp.isoformat()

            df = pd.concat([calls, puts], ignore_index=True)
            if "strike" in df.columns:
                df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

            df = add_black_scholes_greeks_to_chain(
                df,
                spot=spot,
                expiry=exp,
                as_of=effective_snapshot_date,
                r=risk_free_rate,
            )

            keep = [
                "contractSymbol",
                "optionType",
                "expiry",
                "strike",
                "lastPrice",
                "bid",
                "ask",
                "change",
                "percentChange",
                "volume",
                "openInterest",
                "impliedVolatility",
                "inTheMoney",
                "bs_price",
                "bs_delta",
                "bs_gamma",
                "bs_theta_per_day",
                "bs_vega",
            ]
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

            store.save_expiry_snapshot(symbol, effective_snapshot_date, expiry=exp, snapshot=df, meta=meta)
            console.print(f"{symbol} {exp.isoformat()}: saved {len(df)} contracts")

    if dates_used:
        days = ", ".join(sorted({d.isoformat() for d in dates_used}))
        console.print(f"Snapshot complete. Data date(s): {days}.")


@app.command("flow")
def flow_report(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist/--all-watchlists).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). When provided, reports flow for those watchlists instead of portfolio positions.",
    ),
    all_watchlists: bool = typer.Option(
        False,
        "--all-watchlists",
        help="Report flow for all watchlists in the watchlists store (instead of portfolio positions).",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only report a single symbol (overrides portfolio/watchlists selection).",
    ),
    window: int = typer.Option(
        1,
        "--window",
        min=1,
        max=30,
        help="Number of snapshot-to-snapshot deltas to net (requires N+1 snapshots).",
    ),
    group_by: str = typer.Option(
        "contract",
        "--group-by",
        help="Aggregation mode: contract|strike|expiry|expiry-strike",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/flow/{SYMBOL}/).",
    ),
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = OptionsSnapshotStore(cache_dir)
    use_watchlists = bool(watchlist) or all_watchlists
    if use_watchlists:
        wl = load_watchlists(watchlists_path)
        if all_watchlists:
            symbols = sorted({s for syms in wl.watchlists.values() for s in syms})
            if not symbols:
                console.print(f"No watchlists in {watchlists_path}")
                raise typer.Exit(0)
        else:
            symbols_set: set[str] = set()
            for name in watchlist:
                syms = wl.get(name)
                if not syms:
                    raise typer.BadParameter(
                        f"Watchlist '{name}' is empty or missing in {watchlists_path}",
                        param_hint="--watchlist",
                    )
                symbols_set.update(syms)
            symbols = sorted(symbols_set)
    else:
        symbols = sorted({p.symbol for p in portfolio.positions})
        if not symbols and symbol is None:
            console.print("No positions.")
            raise typer.Exit(0)

    if symbol is not None:
        symbols = [symbol.upper()]

    pos_keys = {(p.symbol, p.expiry.isoformat(), float(p.strike), p.option_type) for p in portfolio.positions}

    from rich.table import Table

    group_by_norm = group_by.strip().lower()
    valid_group_by = {"contract", "strike", "expiry", "expiry-strike"}
    if group_by_norm not in valid_group_by:
        raise typer.BadParameter(
            f"Invalid --group-by (use {', '.join(sorted(valid_group_by))})",
            param_hint="--group-by",
        )
    group_by_val = cast(FlowGroupBy, group_by_norm)

    for sym in symbols:
        need = window + 1
        dates = store.latest_dates(sym, n=need)
        if len(dates) < need:
            console.print(f"[yellow]No flow data for {sym}:[/yellow] need at least {need} snapshots.")
            continue

        pair_flows: list[pd.DataFrame] = []
        for prev_date, today_date in zip(dates[:-1], dates[1:], strict=False):
            today_df = store.load_day(sym, today_date)
            prev_df = store.load_day(sym, prev_date)
            if today_df.empty or prev_df.empty:
                console.print(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s) in window.")
                pair_flows = []
                break

            spot = _spot_from_meta(store.load_meta(sym, today_date))
            pair_flows.append(compute_flow(today_df, prev_df, spot=spot))

        if not pair_flows:
            continue

        start_date, end_date = dates[0], dates[-1]

        # Backward-compatible view: window=1 + per-contract list.
        if window == 1 and group_by_norm == "contract":
            prev_date, today_date = dates[-2], dates[-1]
            flow = pair_flows[-1]
            summary = summarize_flow(flow)

            console.print(
                f"\n[bold]{sym}[/bold] flow {prev_date.isoformat()} → {today_date.isoformat()} | "
                f"calls ΔOI$={summary['calls_delta_oi_notional']:,.0f} | puts ΔOI$={summary['puts_delta_oi_notional']:,.0f}"
            )

            if flow.empty:
                console.print("No flow rows.")
                continue

            if "deltaOI_notional" in flow.columns:
                flow = flow.assign(_abs=flow["deltaOI_notional"].abs())
                flow = flow.sort_values("_abs", ascending=False).drop(columns=["_abs"])

            table = Table(title=f"{sym} top {top} contracts by |ΔOI_notional|")
            table.add_column("*")
            table.add_column("Expiry")
            table.add_column("Type")
            table.add_column("Strike", justify="right")
            table.add_column("ΔOI", justify="right")
            table.add_column("OI", justify="right")
            table.add_column("Vol", justify="right")
            table.add_column("ΔOI$", justify="right")
            table.add_column("Class")

            for _, row in flow.head(top).iterrows():
                expiry = str(row.get("expiry", "-"))
                opt_type = str(row.get("optionType", "-"))
                strike = row.get("strike")
                strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
                key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
                in_port = key in pos_keys if strike_val is not None else False

                table.add_row(
                    "*" if in_port else "",
                    expiry,
                    opt_type,
                    "-" if strike_val is None else f"{strike_val:g}",
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("openInterest")) else f"{row.get('openInterest'):.0f}",
                    "-" if pd.isna(row.get("volume")) else f"{row.get('volume'):.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    str(row.get("flow_class", "-")),
                )

            console.print(table)

            if out is not None:
                net = aggregate_flow_window(pair_flows, group_by="contract")
                net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
                sort_cols = ["_abs"]
                ascending = [False]
                for c in ["expiry", "strike", "optionType", "contractSymbol"]:
                    if c in net.columns:
                        sort_cols.append(c)
                        ascending.append(True)
                net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

                base = out / "flow" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                out_path = base / f"{prev_date.isoformat()}_to_{today_date.isoformat()}_w1_contract.json"
                artifact_net = net.rename(
                    columns={
                        "contractSymbol": "contract_symbol",
                        "optionType": "option_type",
                        "deltaOI": "delta_oi",
                        "deltaOI_notional": "delta_oi_notional",
                        "size": "n_pairs",
                    }
                )
                payload = {
                    "schema_version": 1,
                    "symbol": sym.upper(),
                    "from_date": prev_date.isoformat(),
                    "to_date": today_date.isoformat(),
                    "window": 1,
                    "group_by": "contract",
                    "snapshot_dates": [prev_date.isoformat(), today_date.isoformat()],
                    "net": artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                }
                out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                console.print(f"\nSaved: {out_path}")
            continue

        # Windowed + aggregated views.
        net = aggregate_flow_window(pair_flows, group_by=group_by_val)
        if net.empty:
            console.print(f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()})")
            console.print("No net flow rows.")
            continue

        calls_premium = float(net[net["optionType"] == "call"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0
        puts_premium = float(net[net["optionType"] == "put"]["deltaOI_notional"].sum()) if "deltaOI_notional" in net.columns else 0.0

        console.print(
            f"\n[bold]{sym}[/bold] flow net window={window} ({start_date.isoformat()} → {end_date.isoformat()}) | "
            f"group-by={group_by_norm} | calls ΔOI$={calls_premium:,.0f} | puts ΔOI$={puts_premium:,.0f}"
        )

        net = net.assign(_abs=net["deltaOI_notional"].abs() if "deltaOI_notional" in net.columns else 0.0)
        sort_cols = ["_abs"]
        ascending = [False]
        for c in ["expiry", "strike", "optionType", "contractSymbol"]:
            if c in net.columns:
                sort_cols.append(c)
                ascending.append(True)

        net = net.sort_values(sort_cols, ascending=ascending, na_position="last").drop(columns=["_abs"])

        def _render_zone_table(title: str) -> Table:
            t = Table(title=title)
            if group_by_norm == "contract":
                t.add_column("*")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                t.add_column("Expiry")
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                t.add_column("Strike", justify="right")
            t.add_column("Type")
            t.add_column("Net ΔOI", justify="right")
            t.add_column("Net ΔOI$", justify="right")
            t.add_column("Net Δ$", justify="right")
            t.add_column("N", justify="right")
            return t

        def _add_zone_row(t: Table, row) -> None:
            expiry = str(row.get("expiry", "-"))
            opt_type = str(row.get("optionType", "-"))
            strike = row.get("strike")
            strike_val = float(strike) if strike is not None and not pd.isna(strike) else None
            key = (sym, expiry, strike_val if strike_val is not None else float("nan"), opt_type)
            in_port = key in pos_keys if strike_val is not None else False

            cells: list[str] = []
            if group_by_norm == "contract":
                cells.append("*" if in_port else "")
            if group_by_norm in {"expiry", "expiry-strike", "contract"}:
                cells.append(expiry)
            if group_by_norm in {"strike", "expiry-strike", "contract"}:
                cells.append("-" if strike_val is None else f"{strike_val:g}")
            cells.extend(
                [
                    opt_type,
                    "-" if pd.isna(row.get("deltaOI")) else f"{row.get('deltaOI'):+.0f}",
                    "-" if pd.isna(row.get("deltaOI_notional")) else f"{row.get('deltaOI_notional'):+.0f}",
                    "-" if pd.isna(row.get("delta_notional")) else f"{row.get('delta_notional'):+.0f}",
                    "-" if pd.isna(row.get("size")) else f"{int(row.get('size')):d}",
                ]
            )
            t.add_row(*cells)

        building = net[net["deltaOI_notional"] > 0].head(top)
        unwinding = net[net["deltaOI_notional"] < 0].head(top)

        t_build = _render_zone_table(f"{sym} building zones (top {top} by |net ΔOI$|)")
        for _, row in building.iterrows():
            _add_zone_row(t_build, row)
        console.print(t_build)

        t_unwind = _render_zone_table(f"{sym} unwinding zones (top {top} by |net ΔOI$|)")
        for _, row in unwinding.iterrows():
            _add_zone_row(t_unwind, row)
        console.print(t_unwind)

        if out is not None:
            base = out / "flow" / sym.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{start_date.isoformat()}_to_{end_date.isoformat()}_w{window}_{group_by_norm}.json"
            artifact_net = net.rename(
                columns={
                    "contractSymbol": "contract_symbol",
                    "optionType": "option_type",
                    "deltaOI": "delta_oi",
                    "deltaOI_notional": "delta_oi_notional",
                    "size": "n_pairs",
                }
            )
            payload = {
                "schema_version": 1,
                "symbol": sym.upper(),
                "from_date": start_date.isoformat(),
                "to_date": end_date.isoformat(),
                "window": window,
                "group_by": group_by_norm,
                "snapshot_dates": [d.isoformat() for d in dates],
                "net": artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")


@app.command("chain-report")
def chain_report(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to report on."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    format: str = typer.Option(
        "console",
        "--format",
        help="Output format: console|md|json",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/chains/{SYMBOL}/).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top strikes to show for walls/gamma."),
    include_expiry: list[str] = typer.Option(
        [],
        "--include-expiry",
        help="Include a specific expiry date (repeatable). When provided, overrides --expiries selection.",
    ),
    expiries: str = typer.Option(
        "near",
        "--expiries",
        help="Expiry selection mode: near|monthly|all (ignored when --include-expiry is used).",
    ),
    best_effort: bool = typer.Option(
        False,
        "--best-effort",
        help="Don't fail hard on missing fields; emit warnings and partial outputs.",
    ),
) -> None:
    """Offline options chain dashboard from local snapshot files."""
    console = Console()
    store = OptionsSnapshotStore(cache_dir)

    try:
        as_of_date = store.resolve_date(symbol, as_of)
        df = store.load_day(symbol, as_of_date)
        meta = store.load_meta(symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        fmt = format.strip().lower()
        if fmt not in {"console", "md", "json"}:
            raise typer.BadParameter("Invalid --format (use console|md|json)", param_hint="--format")

        expiries_mode = expiries.strip().lower()
        if expiries_mode not in {"near", "monthly", "all"}:
            raise typer.BadParameter("Invalid --expiries (use near|monthly|all)", param_hint="--expiries")

        include_dates = [_parse_date(x) for x in include_expiry] if include_expiry else None

        report = compute_chain_report(
            df,
            symbol=symbol,
            as_of=as_of_date,
            spot=spot,
            expiries_mode=expiries_mode,  # type: ignore[arg-type]
            include_expiries=include_dates,
            top=top,
            best_effort=best_effort,
        )

        if fmt == "console":
            render_chain_report_console(console, report)
        elif fmt == "md":
            console.print(render_chain_report_markdown(report))
        else:
            console.print(report.model_dump_json(indent=2))

        if out is not None:
            base = out / "chains" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            json_path = base / f"{as_of_date.isoformat()}.json"
            json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

            # Write Markdown alongside JSON (human-friendly artifact).
            md_path = base / f"{as_of_date.isoformat()}.md"
            md_path.write_text(render_chain_report_markdown(report), encoding="utf-8")

            console.print(f"\nSaved: {json_path}")
            console.print(f"Saved: {md_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("compare")
def compare_snapshots(
    symbol: str = typer.Option(..., "--symbol", help="Symbol to diff."),
    from_spec: str = typer.Option(
        "-1",
        "--from",
        help="From snapshot date (YYYY-MM-DD) or a negative offset relative to --to (e.g. -1).",
    ),
    to_spec: str = typer.Option("latest", "--to", help="To snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top rows to include per section."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output root for saved artifacts (writes under {out}/compare/{SYMBOL}/).",
    ),
) -> None:
    """Diff two snapshot dates for a symbol (offline)."""
    console = Console()
    store = OptionsSnapshotStore(cache_dir)

    try:
        to_date = store.resolve_date(symbol, to_spec)

        from_spec_norm = from_spec.strip().lower()
        if from_spec_norm.startswith("-") and from_spec_norm[1:].isdigit():
            from_date = store.resolve_relative_date(symbol, to_date=to_date, offset=int(from_spec_norm))
        else:
            from_date = store.resolve_date(symbol, from_spec_norm)

        df_from = store.load_day(symbol, from_date)
        df_to = store.load_day(symbol, to_date)
        meta_from = store.load_meta(symbol, from_date)
        meta_to = store.load_meta(symbol, to_date)

        spot_from = _spot_from_meta(meta_from)
        spot_to = _spot_from_meta(meta_to)
        if spot_from is None or spot_to is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        diff, report_from, report_to = compute_compare_report(
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            from_df=df_from,
            to_df=df_to,
            spot_from=spot_from,
            spot_to=spot_to,
            top=top,
        )

        render_compare_report_console(console, diff)

        if out is not None:
            base = out / "compare" / symbol.upper()
            base.mkdir(parents=True, exist_ok=True)
            out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}.json"
            payload = {
                "schema_version": 1,
                "symbol": symbol.upper(),
                "from": report_from.model_dump(),
                "to": report_to.model_dump(),
                "diff": diff.model_dump(),
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            console.print(f"\nSaved: {out_path}")
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("briefing")
def briefing(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (used with --watchlist).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name to include (repeatable). Adds to portfolio symbols.",
    ),
    symbol: str | None = typer.Option(
        None,
        "--symbol",
        help="Only include a single symbol (overrides portfolio/watchlists selection).",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    compare: str = typer.Option(
        "-1",
        "--compare",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path (Markdown) or directory. Default: data/reports/daily/{ASOF}.md",
    ),
    update_derived: bool = typer.Option(
        True,
        "--update-derived/--no-update-derived",
        help="Update derived metrics for included symbols (per-symbol CSV).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used when --update-derived).",
    ),
    top: int = typer.Option(3, "--top", min=1, max=10, help="Top rows to include in compare/flow sections."),
) -> None:
    """Generate a daily Markdown briefing for portfolio + optional watchlists (offline-first)."""
    console = Console(width=200)
    portfolio = load_portfolio(portfolio_path)

    portfolio_symbols = sorted({p.symbol.upper() for p in portfolio.positions})
    watch_symbols: list[str] = []
    if watchlist:
        try:
            wl = load_watchlists(watchlists_path)
            for name in watchlist:
                watch_symbols.extend(wl.get(name))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] failed to load watchlists: {exc}")

    symbols = sorted(set(portfolio_symbols).union({s.upper() for s in watch_symbols if s}))
    if symbol is not None:
        symbols = [symbol.upper().strip()]

    if not symbols:
        console.print("[red]Error:[/red] no symbols selected (empty portfolio and no watchlists)")
        raise typer.Exit(1)

    store = OptionsSnapshotStore(cache_dir)
    derived_store = DerivedStore(derived_dir)

    # Cache day snapshots for portfolio marks (best-effort).
    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}

    sections: list[BriefingSymbolSection] = []
    resolved_to_dates: list[date] = []
    compare_norm = compare.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    for sym in symbols:
        errors: list[str] = []
        warnings: list[str] = []
        chain = None
        compare_report = None
        flow_net = None
        derived_updated = False

        try:
            to_date = store.resolve_date(sym, as_of)
            resolved_to_dates.append(to_date)

            df_to = store.load_day(sym, to_date)
            meta_to = store.load_meta(sym, to_date)
            spot_to = _spot_from_meta(meta_to)
            if spot_to is None:
                raise ValueError("missing spot price in meta.json (run snapshot-options first)")

            day_cache[sym] = (to_date, df_to)

            chain = compute_chain_report(
                df_to,
                symbol=sym,
                as_of=to_date,
                spot=spot_to,
                expiries_mode="near",
                top=10,
                best_effort=True,
            )

            if update_derived:
                row = DerivedRow.from_chain_report(chain)
                try:
                    derived_store.upsert(sym, row)
                    derived_updated = True
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"derived update failed: {exc}")

            if compare_enabled:
                from_date: date | None = None
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    try:
                        from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"compare unavailable: {exc}")
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                if from_date is not None and from_date != to_date:
                    df_from = store.load_day(sym, from_date)
                    meta_from = store.load_meta(sym, from_date)
                    spot_from = _spot_from_meta(meta_from)
                    if spot_from is None:
                        warnings.append("compare unavailable: missing spot in from-date meta.json")
                    elif df_from.empty or df_to.empty:
                        warnings.append("compare unavailable: missing snapshot CSVs for from/to date")
                    else:
                        compare_report, _, _ = compute_compare_report(
                            symbol=sym,
                            from_date=from_date,
                            to_date=to_date,
                            from_df=df_from,
                            to_df=df_to,
                            spot_from=spot_from,
                            spot_to=spot_to,
                            top=top,
                        )

                        try:
                            flow = compute_flow(df_to, df_from, spot=spot_to)
                            flow_net = aggregate_flow_window([flow], group_by="strike")
                        except Exception:  # noqa: BLE001
                            warnings.append("flow unavailable: compute failed")
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        as_of_label = "-" if sym not in day_cache else day_cache[sym][0].isoformat()
        sections.append(
            BriefingSymbolSection(
                symbol=sym,
                as_of=as_of_label,
                chain=chain,
                compare=compare_report,
                flow_net=flow_net,
                errors=errors,
                warnings=warnings,
                derived_updated=derived_updated,
            )
        )

    if not resolved_to_dates:
        console.print("[red]Error:[/red] no snapshots found for selected symbols")
        raise typer.Exit(1)
    report_date = max(resolved_to_dates).isoformat()
    portfolio_rows: list[dict[str, str]] = []
    for p in portfolio.positions:
        sym = p.symbol.upper()
        to_date, df_to = day_cache.get(sym, (None, pd.DataFrame()))

        mark = None
        if not df_to.empty:
            sub = df_to.copy()
            if "expiry" in sub.columns:
                sub = sub[sub["expiry"].astype(str) == p.expiry.isoformat()]
            if "optionType" in sub.columns:
                sub = sub[sub["optionType"].astype(str).str.lower() == p.option_type]
            if "strike" in sub.columns:
                strike = pd.to_numeric(sub["strike"], errors="coerce")
                sub = sub.assign(_strike=strike)
                sub = sub[(sub["_strike"] - float(p.strike)).abs() < 1e-9]
            if not sub.empty:
                row = sub.iloc[0]
                mark = _mark_price(
                    bid=_extract_float(row, "bid"),
                    ask=_extract_float(row, "ask"),
                    last=_extract_float(row, "lastPrice"),
                )

        pnl_abs = None
        pnl_pct = None
        if mark is not None:
            pnl_abs = (mark - p.cost_basis) * 100.0 * p.contracts
            pnl_pct = ((mark / p.cost_basis) - 1.0) if p.cost_basis > 0 else None

        portfolio_rows.append(
            {
                "id": p.id,
                "symbol": sym,
                "type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": f"{p.strike:g}",
                "ct": str(p.contracts),
                "cost": f"{p.cost_basis:.2f}",
                "mark": "-" if mark is None else f"{mark:.2f}",
                "pnl_$": "-" if pnl_abs is None else f"{pnl_abs:+.0f}",
                "pnl_%": "-" if pnl_pct is None else f"{pnl_pct * 100.0:+.1f}%",
                "as_of": "-" if to_date is None else to_date.isoformat(),
            }
        )

    portfolio_table_md = None
    if portfolio_rows:
        headers = ["ID", "Sym", "Type", "Exp", "Strike", "Ct", "Cost", "Mark", "PnL $", "PnL %", "As-of"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for r in portfolio_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        r["id"],
                        r["symbol"],
                        r["type"],
                        r["expiry"],
                        r["strike"],
                        r["ct"],
                        r["cost"],
                        r["mark"],
                        r["pnl_$"],
                        r["pnl_%"],
                        r["as_of"],
                    ]
                )
                + " |"
            )
        portfolio_table_md = "\n".join(lines)

    md = render_briefing_markdown(
        report_date=report_date,
        portfolio_path=str(portfolio_path),
        symbol_sections=sections,
        portfolio_table_md=portfolio_table_md,
    )

    if out is None:
        out_path = Path("data/reports/daily") / f"{report_date}.md"
    else:
        out_path = out
        if out_path.suffix.lower() != ".md":
            out_path = out_path / f"{report_date}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    console.print(f"Saved: {out_path}")


@app.command("roll-plan")
def roll_plan(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (positions + risk profile)."),
    position_id: str = typer.Option(..., "--id", help="Position id to plan a roll for."),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest'."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    intent: str = typer.Option(
        "max-upside",
        "--intent",
        help="Intent: max-upside|reduce-theta|increase-delta|de-risk",
    ),
    horizon_months: int = typer.Option(..., "--horizon-months", min=1, max=60),
    shape: str = typer.Option(
        "out-same-strike",
        "--shape",
        help="Roll shape: out-same-strike|out-up|out-down",
    ),
    top: int = typer.Option(10, "--top", min=1, max=50, help="Number of candidates to display."),
    max_debit: float | None = typer.Option(
        None,
        "--max-debit",
        help="Max roll debit in dollars (total for position size).",
    ),
    min_credit: float | None = typer.Option(
        None,
        "--min-credit",
        help="Min roll credit in dollars (total for position size).",
    ),
    min_open_interest: int | None = typer.Option(
        None,
        "--min-open-interest",
        help="Override minimum open interest liquidity gate (default from risk profile).",
    ),
    min_volume: int | None = typer.Option(
        None,
        "--min-volume",
        help="Override minimum volume liquidity gate (default from risk profile).",
    ),
) -> None:
    """Propose and rank roll candidates for a single position using offline snapshots."""
    console = Console(width=200)

    portfolio = load_portfolio(portfolio_path)
    position = next((p for p in portfolio.positions if p.id == position_id), None)
    if position is None:
        raise typer.BadParameter(f"No position found with id: {position_id}", param_hint="--id")

    intent_norm = intent.strip().lower()
    if intent_norm not in {"max-upside", "reduce-theta", "increase-delta", "de-risk"}:
        raise typer.BadParameter(
            "Invalid --intent (use max-upside|reduce-theta|increase-delta|de-risk)",
            param_hint="--intent",
        )

    shape_norm = shape.strip().lower()
    if shape_norm not in {"out-same-strike", "out-up", "out-down"}:
        raise typer.BadParameter("Invalid --shape (use out-same-strike|out-up|out-down)", param_hint="--shape")

    rp = portfolio.risk_profile
    min_oi = rp.min_open_interest if min_open_interest is None else int(min_open_interest)
    min_vol = rp.min_volume if min_volume is None else int(min_volume)

    store = OptionsSnapshotStore(cache_dir)

    try:
        as_of_date = store.resolve_date(position.symbol, as_of)
        df = store.load_day(position.symbol, as_of_date)
        meta = store.load_meta(position.symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        report = compute_roll_plan(
            df,
            symbol=position.symbol,
            as_of=as_of_date,
            spot=spot,
            position=position,
            intent=intent_norm,  # type: ignore[arg-type]
            horizon_months=horizon_months,
            shape=shape_norm,  # type: ignore[arg-type]
            min_open_interest=min_oi,
            min_volume=min_vol,
            max_debit=max_debit,
            min_credit=min_credit,
            top=top,
        )

        render_roll_plan_console(console, report)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


@app.command("earnings")
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
    store = EarningsStore(cache_dir)
    sym = symbol.upper().strip()

    if clear:
        deleted = store.delete(sym)
        console.print(f"Deleted: {sym}" if deleted else f"No cache found for: {sym}")
        return

    record: EarningsRecord | None = None

    if set_date is not None:
        d = _parse_date(set_date)
        record = EarningsRecord.manual(symbol=sym, next_earnings_date=d, note="Set via CLI --set.")
        out_path = store.save(record)
        console.print(f"Saved: {out_path}")
    else:
        record = store.load(sym)
        if refresh or record is None:
            client = YFinanceClient()
            try:
                ev = client.get_next_earnings_event(sym)
            except DataFetchError as exc:
                console.print(f"[red]Data error:[/red] {exc}")
                raise typer.Exit(1)
            record = EarningsRecord(
                symbol=sym,
                fetched_at=datetime.now(tz=timezone.utc),
                source=ev.source,
                next_earnings_date=ev.next_date,
                window_start=ev.window_start,
                window_end=ev.window_end,
                raw=ev.raw,
                notes=[],
            )
            out_path = store.save(record)
            console.print(f"Saved: {out_path}")

    if record is None:
        console.print(f"No earnings record for {sym}.")
        raise typer.Exit(1)

    if json_out:
        console.print_json(json.dumps(record.model_dump(mode='json'), sort_keys=True))
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


@app.command("refresh-earnings")
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

    store = EarningsStore(cache_dir)
    client = YFinanceClient()

    ok = 0
    err = 0
    unknown = 0

    console.print(f"Refreshing earnings for {len(symbols)} symbol(s)...")
    for sym in sorted(symbols):
        try:
            ev = client.get_next_earnings_event(sym)
            record = EarningsRecord(
                symbol=sym,
                fetched_at=datetime.now(tz=timezone.utc),
                source=ev.source,
                next_earnings_date=ev.next_date,
                window_start=ev.window_start,
                window_end=ev.window_end,
                raw=ev.raw,
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


@app.command()
def research(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (risk profile + candle cache config)."),
    symbol: str | None = typer.Option(None, "--symbol", help="Run research for a single symbol."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    watchlist: str = typer.Option(
        "watchlist",
        "--watchlist",
        help="Watchlist name to research (ignored when --symbol is provided).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    period: str = typer.Option(
        "5y",
        "--period",
        help="Daily candle period to ensure cached for research (yfinance period format).",
    ),
    window_pct: float = typer.Option(
        0.30,
        "--window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot for option selection (e.g. 0.30 = +/-30%).",
    ),
    short_min_dte: int = typer.Option(30, "--short-min-dte"),
    short_max_dte: int = typer.Option(90, "--short-max-dte"),
    long_min_dte: int = typer.Option(365, "--long-min-dte"),
    long_max_dte: int = typer.Option(1500, "--long-max-dte"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save the research output to a .txt report."),
    output_dir: Path = typer.Option(
        Path("data/research"),
        "--output-dir",
        help="Directory for saved research reports (ignored when --no-save).",
    ),
) -> None:
    """Recommend short-dated (30-90d) and long-dated (LEAPS) contracts based on technicals."""
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    if symbol:
        symbols = [symbol.strip().upper()]
    else:
        wl = load_watchlists(watchlists_path)
        symbols = wl.get(watchlist)
        if not symbols:
            raise typer.BadParameter(f"Watchlist '{watchlist}' is empty or missing in {watchlists_path}")

    candle_store = CandleStore(candle_cache_dir)
    client = YFinanceClient()
    console = Console()

    from rich.table import Table

    report_buffer = None
    report_console = None
    symbol_outputs: dict[str, str] = {}
    symbol_candle_dates: dict[str, date] = {}
    symbol_candle_datetimes: dict[str, datetime] = {}
    if save:
        import io

        report_buffer = io.StringIO()
        report_console = Console(file=report_buffer, width=200, force_terminal=False)

    symbol_console: Console | None = None

    def emit(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        console.print(*args, **kwargs)
        if report_console is not None:
            report_console.print(*args, **kwargs)
        if symbol_console is not None:
            symbol_console.print(*args, **kwargs)

    for sym in symbols:
        symbol_buffer = None
        symbol_console = None
        if save:
            import io

            symbol_buffer = io.StringIO()
            symbol_console = Console(file=symbol_buffer, width=200, force_terminal=False)

        history = candle_store.get_daily_history(sym, period=period)
        if not history.empty:
            last_ts = history.index.max()
            # Candle store normalizes to tz-naive DatetimeIndex.
            symbol_candle_dates[sym] = last_ts.date()
            symbol_candle_datetimes[sym] = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
        setup = analyze_underlying(sym, history=history, risk_profile=rp)

        emit(f"\n[bold]{sym}[/bold] — setup: {setup.direction.value}")
        for r in setup.reasons:
            emit(f"  - {r}")

        if setup.spot is None:
            emit("  - No spot price; skipping option selection.")
            continue

        levels = suggest_trade_levels(setup, history=history, risk_profile=rp)
        if levels.entry is not None:
            # Percent change relative to the latest close (setup.spot).
            pct_from_close = None
            try:
                if setup.spot:
                    pct_from_close = (float(levels.entry) / float(setup.spot) - 1.0) * 100.0
            except Exception:  # noqa: BLE001
                pct_from_close = None

            if pct_from_close is None:
                emit(f"  - Suggested entry (underlying): ${levels.entry:.2f}")
            else:
                emit(f"  - Suggested entry (underlying): ${levels.entry:.2f} ({pct_from_close:+.2f}%)")
        if levels.pullback_entry is not None:
            emit(f"  - Pullback entry (underlying): ${levels.pullback_entry:.2f}")
        if levels.stop is not None:
            emit(f"  - Suggested stop (underlying): ${levels.stop:.2f}")
        for note in levels.notes:
            emit(f"    - {note}")

        expiry_strs = list(client.ticker(sym).options or [])
        if not expiry_strs:
            emit("  - No listed option expirations found.")
            continue

        short_exp = choose_expiry(
            expiry_strs, min_dte=short_min_dte, max_dte=short_max_dte, target_dte=60
        )
        long_exp = choose_expiry(
            expiry_strs, min_dte=long_min_dte, max_dte=long_max_dte, target_dte=540
        )
        if long_exp is None:
            # Fallback: pick the farthest expiry that still qualifies as "long".
            parsed = []
            for s in expiry_strs:
                try:
                    exp = date.fromisoformat(s)
                except ValueError:
                    continue
                dte = (exp - date.today()).days
                parsed.append((dte, exp))
            parsed = [t for t in parsed if t[0] >= long_min_dte]
            if parsed:
                _, long_exp = max(parsed, key=lambda t: t[0])

        if setup.direction == Direction.NEUTRAL:
            emit("  - No strong directional setup; skipping contract recommendations.")
            continue

        opt_type: OptionType = "call" if setup.direction == Direction.BULLISH else "put"
        min_oi = rp.min_open_interest
        min_vol = rp.min_volume

        table = Table(title=f"{sym} option ideas (best-effort)")
        table.add_column("Horizon")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Type")
        table.add_column("Strike", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("OI", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Why")

        if short_exp is not None:
            chain = client.get_options_chain(sym, short_exp)
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.40 if opt_type == "call" else -0.40
            short_pick = select_option_candidate(
                df,
                symbol=sym,
                option_type=opt_type,
                expiry=short_exp,
                spot=setup.spot,
                target_delta=target_delta,
                window_pct=window_pct,
                min_open_interest=min_oi,
                min_volume=min_vol,
            )
            if short_pick is not None:
                why = "; ".join(short_pick.rationale[:2])
                table.add_row(
                    "30–90d",
                    short_pick.expiry.isoformat(),
                    str(short_pick.dte),
                    short_pick.option_type,
                    f"{short_pick.strike:g}",
                    "-" if short_pick.mark is None else f"${short_pick.mark:.2f}",
                    "-" if short_pick.delta is None else f"{short_pick.delta:+.2f}",
                    "-" if short_pick.iv is None else f"{short_pick.iv:.1%}",
                    "-" if short_pick.open_interest is None else str(short_pick.open_interest),
                    "-" if short_pick.volume is None else str(short_pick.volume),
                    why,
                )
        else:
            emit(f"  - No expiries found in {short_min_dte}-{short_max_dte} DTE range.")

        if long_exp is not None:
            chain = client.get_options_chain(sym, long_exp)
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.70 if opt_type == "call" else -0.70
            long_pick = select_option_candidate(
                df,
                symbol=sym,
                option_type=opt_type,
                expiry=long_exp,
                spot=setup.spot,
                target_delta=target_delta,
                window_pct=window_pct,
                min_open_interest=min_oi,
                min_volume=min_vol,
            )
            if long_pick is not None:
                why = "; ".join(long_pick.rationale[:2] + ["Longer DTE reduces theta pressure."])
                table.add_row(
                    "LEAPS",
                    long_pick.expiry.isoformat(),
                    str(long_pick.dte),
                    long_pick.option_type,
                    f"{long_pick.strike:g}",
                    "-" if long_pick.mark is None else f"${long_pick.mark:.2f}",
                    "-" if long_pick.delta is None else f"{long_pick.delta:+.2f}",
                    "-" if long_pick.iv is None else f"{long_pick.iv:.1%}",
                    "-" if long_pick.open_interest is None else str(long_pick.open_interest),
                    "-" if long_pick.volume is None else str(long_pick.volume),
                    why,
                )
        else:
            emit(f"  - No expiries found in {long_min_dte}-{long_max_dte} DTE range.")

        emit(table)

        if symbol_buffer is not None:
            symbol_outputs[sym] = symbol_buffer.getvalue().lstrip()

    def _render_ticker_entry(*, sym: str, candle_day: date, run_dt: datetime, body: str) -> str:
        run_ts = run_dt.strftime("%Y-%m-%d %H:%M:%S")
        header = f"=== {candle_day.isoformat()} ===\nrun_at: {run_ts}\ncandles_through: {candle_day.isoformat()}\n"
        return f"{header}\n{body.strip()}\n"

    def _parse_ticker_entries(text: str) -> dict[str, str]:
        import re

        pattern = re.compile(r"^=== (\\d{4}-\\d{2}-\\d{2}) ===$", re.M)
        matches = list(pattern.finditer(text))
        if not matches:
            return {}

        entries: dict[str, str] = {}
        for idx, match in enumerate(matches):
            day = match.group(1)
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            entries[day] = text[start:end].strip()
        return entries

    def _upsert_ticker_entry(*, path: Path, candle_day: date, new_entry: str) -> None:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        entries = _parse_ticker_entries(existing)
        entries[candle_day.isoformat()] = new_entry.strip()
        ordered_days = sorted(entries.keys(), reverse=True)
        out = "\n\n".join(entries[d] for d in ordered_days).rstrip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")

    if save and report_buffer is not None:
        run_dt = datetime.now()
        candle_dt = max(symbol_candle_datetimes.values()) if symbol_candle_datetimes else run_dt
        candle_day = candle_dt.date()
        candle_stamp = candle_dt.strftime("%Y-%m-%d_%H%M%S")
        run_stamp = run_dt.strftime("%Y-%m-%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)

        out_path = output_dir / f"research-{candle_stamp}-{run_stamp}.txt"
        header = (
            f"run_at: {run_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"candles_through: {candle_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"symbols: {', '.join(symbols)}\n\n"
        )
        out_path.write_text(header + report_buffer.getvalue().lstrip(), encoding="utf-8")

        tickers_dir = output_dir / "tickers"
        for sym, body in symbol_outputs.items():
            sym_day = symbol_candle_dates.get(sym) or candle_day
            entry = _render_ticker_entry(sym=sym, candle_day=sym_day, run_dt=run_dt, body=body)
            _upsert_ticker_entry(path=tickers_dir / f"{sym}.txt", candle_day=sym_day, new_entry=entry)

        console.print(f"\nSaved research report to {out_path}", soft_wrap=True)


@app.command("refresh-candles")
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
    symbols: set[str] = {p.symbol.upper() for p in portfolio.positions}

    wl = load_watchlists(watchlists_path)
    if watchlist:
        for name in watchlist:
            symbols.update(wl.get(name))
    else:
        for syms in (wl.watchlists or {}).values():
            symbols.update(syms or [])

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        Console().print("No symbols found (no positions and no watchlists).")
        raise typer.Exit(0)

    store = CandleStore(candle_cache_dir)
    console = Console()
    console.print(f"Refreshing daily candles for {len(symbols)} symbol(s)...")

    for sym in sorted(symbols):
        try:
            hist = store.get_daily_history(sym, period=period)
            if hist.empty:
                console.print(f"[yellow]Warning:[/yellow] {sym}: no candles returned.")
            else:
                last_dt = hist.index.max()
                console.print(f"{sym}: cached through {last_dt.date().isoformat()}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Error:[/red] {sym}: {exc}")


@app.command()
def init(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    force: bool = typer.Option(False, "--force", help="Overwrite if file exists."),
) -> None:
    """Create a starter portfolio JSON file."""
    write_template(portfolio_path, force=force)
    Console().print(f"Wrote template portfolio to {portfolio_path}")


@app.command("list")
def list_positions(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
) -> None:
    """List positions in the portfolio file."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        return

    # Minimal list (no fetch)
    from rich.table import Table

    table = Table(title="Portfolio Positions")
    table.add_column("ID")
    table.add_column("Symbol")
    table.add_column("Type")
    table.add_column("Expiry")
    table.add_column("Strike", justify="right")
    table.add_column("Ct", justify="right")
    table.add_column("Cost", justify="right")
    for p in portfolio.positions:
        table.add_row(
            p.id,
            p.symbol,
            p.option_type,
            p.expiry.isoformat(),
            f"{p.strike:g}",
            str(p.contracts),
            f"${p.cost_basis:.2f}",
        )
    console.print(table)


@watchlists_app.command("init")
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


@watchlists_app.command("sync-positions")
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


@watchlists_app.command("list")
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


@watchlists_app.command("show")
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


@watchlists_app.command("create")
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


@watchlists_app.command("add")
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


@watchlists_app.command("remove")
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


@app.command("add-position")
def add_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    symbol: str = typer.Option(..., "--symbol"),
    expiry: str = typer.Option(..., "--expiry", help="Expiry date, e.g. 2026-04-17."),
    strike: float = typer.Option(..., "--strike"),
    option_type: OptionType = typer.Option(..., "--type", case_sensitive=False),
    contracts: int = typer.Option(1, "--contracts"),
    cost_basis: float = typer.Option(..., "--cost-basis", help="Premium per share (e.g. 0.45)."),
    position_id: str | None = typer.Option(None, "--id", help="Optional position id."),
    opened_at: str | None = typer.Option(None, "--opened-at", help="Optional open date (YYYY-MM-DD)."),
) -> None:
    """Add a position to the portfolio JSON."""
    portfolio = load_portfolio(portfolio_path)

    expiry_date = _parse_date(expiry)
    opened_at_date = _parse_date(opened_at) if opened_at else None

    symbol = symbol.upper()
    option_type = option_type.lower()  # type: ignore[assignment]

    pid = position_id or _default_position_id(symbol, expiry_date, strike, option_type)
    if any(p.id == pid for p in portfolio.positions):
        raise typer.BadParameter(f"Position id already exists: {pid}")

    position = Position(
        id=pid,
        symbol=symbol,
        option_type=option_type,
        expiry=expiry_date,
        strike=float(strike),
        contracts=int(contracts),
        cost_basis=float(cost_basis),
        opened_at=opened_at_date,
    )

    portfolio.positions.append(position)
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Added {pid}")


@app.command("remove-position")
def remove_position(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    position_id: str = typer.Argument(..., help="Position id to remove."),
) -> None:
    """Remove a position by id."""
    portfolio = load_portfolio(portfolio_path)
    before = len(portfolio.positions)
    portfolio.positions = [p for p in portfolio.positions if p.id != position_id]
    after = len(portfolio.positions)
    if before == after:
        raise typer.BadParameter(f"No position found with id: {position_id}")
    save_portfolio(portfolio_path, portfolio)
    Console().print(f"Removed {position_id}")


@app.command()
def analyze(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    period: str = typer.Option("2y", "--period", help="Underlying history period (yfinance)."),
    interval: str = typer.Option("1d", "--interval", help="Underlying history interval (yfinance)."),
    cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--cache-dir",
        help="Directory for locally cached daily candles.",
    ),
) -> None:
    """Fetch data and print metrics + rule-based advice."""
    if interval != "1d":
        raise typer.BadParameter("Only --interval 1d is supported for now (cache uses daily candles).")

    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    client = YFinanceClient()
    candle_store = CandleStore(cache_dir)

    history_by_symbol: dict[str, pd.DataFrame] = {}
    last_price_by_symbol: dict[str, float | None] = {}
    for sym in sorted({p.symbol for p in portfolio.positions}):
        try:
            history = candle_store.get_daily_history(sym, period=period)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Candle cache error:[/red] {sym}: {exc}")
            history = pd.DataFrame()
        history_by_symbol[sym] = history
        last_price_by_symbol[sym] = last_close(history)

    metrics_list: list[PositionMetrics] = []
    advice_by_id: dict[str, Advice] = {}

    for p in portfolio.positions:
        try:
            metrics = _position_metrics(
                client,
                p,
                risk_profile=portfolio.risk_profile,
                underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                underlying_last_price=last_price_by_symbol.get(p.symbol),
            )
            metrics_list.append(metrics)
            advice_by_id[p.id] = advise(metrics, portfolio)
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")

    if not metrics_list:
        raise typer.Exit(1)

    render_positions(console, portfolio, metrics_list, advice_by_id)


@app.command()
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


def _setup_technicals_logging(cfg: dict) -> None:
    level = cfg["logging"]["level"].upper()
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "technical_backtesting.log"
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path)]
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_ohlc_df(
    *,
    ohlc_path: Path | None,
    symbol: str | None,
    cache_dir: Path,
) -> pd.DataFrame:
    if ohlc_path:
        return load_ohlc_from_path(ohlc_path)
    if symbol:
        return load_ohlc_from_cache(symbol, cache_dir)
    raise typer.BadParameter("Provide --ohlc-path or --symbol/--cache-dir")


def _stats_to_dict(stats: object | None) -> dict | None:
    if stats is None:
        return None
    if isinstance(stats, pd.Series):
        return {k: v for k, v in stats.to_dict().items() if not str(k).startswith("_")}
    if isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": stats}


@technicals_app.command("compute-indicators")
def technicals_compute_indicators(
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    output: Path | None = typer.Option(None, "--output", help="Output CSV/parquet path."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Compute indicators from OHLC data and optionally persist to disk."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for indicator computation.")

    features = compute_features(df, cfg)
    console.print(f"Computed features: {len(features)} rows, {len(features.columns)} columns")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.suffix.lower() == ".parquet":
            features.to_parquet(output)
        else:
            features.to_csv(output)
        console.print(f"Wrote features to {output}")


@technicals_app.command("optimize")
def technicals_optimize(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Optimize strategy parameters for a single dataset."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for optimization.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]

    warmup = warmup_bars(cfg)
    best_params, best_stats, heatmap = optimize_params(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        warmup_bars=warmup,
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    console.print(f"Best params: {best_params}")
    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=best_params,
        train_stats=best_stats,
        walk_forward_result=None,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


@technicals_app.command("walk-forward")
def technicals_walk_forward(
    strategy: str = typer.Option(..., "--strategy", help="Strategy name."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run walk-forward optimization and write artifacts."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for walk-forward.")

    features = compute_features(df, cfg)
    StrategyClass = get_strategy(strategy)
    strat_cfg = cfg["strategies"][strategy]
    opt_cfg = cfg["optimization"]
    walk_cfg = cfg["walk_forward"]

    warmup = warmup_bars(cfg)
    result = walk_forward_optimize(
        features,
        StrategyClass,
        cfg["backtest"],
        strat_cfg["search_space"],
        strat_cfg["constraints"],
        opt_cfg["maximize"],
        opt_cfg["method"],
        opt_cfg.get("sambo", {}),
        opt_cfg.get("custom_score", {}),
        walk_cfg,
        strat_cfg["defaults"],
        warmup_bars=warmup,
        min_train_bars=opt_cfg.get("min_train_bars", 0),
        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
    )

    ticker = symbol or "UNKNOWN"
    data_meta = {
        "start": features.index.min(),
        "end": features.index.max(),
        "bars": len(features),
        "warmup_bars": warmup,
    }
    optimize_meta = {
        "method": opt_cfg["method"],
        "maximize": opt_cfg["maximize"],
        "constraints": strat_cfg["constraints"],
    }
    folds_out = []
    for fold in result.folds:
        folds_out.append(
            {
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "validate_start": fold["validate_start"],
                "validate_end": fold["validate_end"],
                "best_params": fold["best_params"],
                "train_stats": _stats_to_dict(fold["train_stats"]),
                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                "validate_score": fold["validate_score"],
            }
        )
    wf_dict = {
        "params": result.params,
        "folds": folds_out,
        "stability": result.stability,
        "used_defaults": result.used_defaults,
        "reason": result.reason,
    }
    heatmap = None
    if result.folds:
        best_fold = max(result.folds, key=lambda f: f.get("validate_score", float("-inf")))
        heatmap = best_fold.get("heatmap")

    paths = write_artifacts(
        cfg,
        ticker=ticker,
        strategy=strategy,
        params=result.params,
        train_stats=None,
        walk_forward_result=wf_dict,
        optimize_meta=optimize_meta,
        data_meta=data_meta,
        heatmap=heatmap,
    )
    console.print(f"Wrote params: {paths.params_path}")
    console.print(f"Wrote summary: {paths.report_path}")


@technicals_app.command("run-all")
def technicals_run_all(
    tickers: str = typer.Option(..., "--tickers", help="Comma-separated tickers."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
) -> None:
    """Run both strategies for a list of tickers."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    if not symbols:
        raise typer.BadParameter("Provide at least one ticker.")

    for symbol in symbols:
        try:
            df = load_ohlc_from_cache(symbol, cache_dir)
            if df.empty:
                console.print(f"[yellow]No data for {symbol} in cache.[/yellow]")
                continue
            features = compute_features(df, cfg)
            warmup = warmup_bars(cfg)
            for strategy, strat_cfg in cfg["strategies"].items():
                if not strat_cfg.get("enabled", False):
                    continue
                StrategyClass = get_strategy(strategy)
                opt_cfg = cfg["optimization"]
                if cfg["walk_forward"]["enabled"]:
                    result = walk_forward_optimize(
                        features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        cfg["walk_forward"],
                        strat_cfg["defaults"],
                        warmup_bars=warmup,
                        min_train_bars=opt_cfg.get("min_train_bars", 0),
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    folds_out = []
                    for fold in result.folds:
                        folds_out.append(
                            {
                                "train_start": fold["train_start"],
                                "train_end": fold["train_end"],
                                "validate_start": fold["validate_start"],
                                "validate_end": fold["validate_end"],
                                "best_params": fold["best_params"],
                                "train_stats": _stats_to_dict(fold["train_stats"]),
                                "validate_stats": _stats_to_dict(fold["validate_stats"]),
                                "validate_score": fold["validate_score"],
                            }
                        )
                    wf_dict = {
                        "params": result.params,
                        "folds": folds_out,
                        "stability": result.stability,
                        "used_defaults": result.used_defaults,
                        "reason": result.reason,
                    }
                    heatmap = None
                    if result.folds:
                        best_fold = max(
                            result.folds, key=lambda f: f.get("validate_score", float("-inf"))
                        )
                        heatmap = best_fold.get("heatmap")
                    train_stats = None
                    params = result.params
                else:
                    best_params, train_stats, heatmap = optimize_params(
                        features,
                        StrategyClass,
                        cfg["backtest"],
                        strat_cfg["search_space"],
                        strat_cfg["constraints"],
                        opt_cfg["maximize"],
                        opt_cfg["method"],
                        opt_cfg.get("sambo", {}),
                        opt_cfg.get("custom_score", {}),
                        warmup_bars=warmup,
                        return_heatmap=cfg["artifacts"].get("write_heatmap", False),
                    )
                    wf_dict = None
                    params = best_params

                data_meta = {
                    "start": features.index.min(),
                    "end": features.index.max(),
                    "bars": len(features),
                    "warmup_bars": warmup,
                }
                optimize_meta = {
                    "method": opt_cfg["method"],
                    "maximize": opt_cfg["maximize"],
                    "constraints": strat_cfg["constraints"],
                }
                write_artifacts(
                    cfg,
                    ticker=symbol,
                    strategy=strategy,
                    params=params,
                    train_stats=train_stats,
                    walk_forward_result=wf_dict,
                    optimize_meta=optimize_meta,
                    data_meta=data_meta,
                    heatmap=heatmap,
                )
            console.print(f"[green]Completed[/green] {symbol}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]{symbol} failed:[/red] {exc}")
