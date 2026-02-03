from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.chain_metrics import compute_chain_report, execution_quality
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.portfolio_risk import (
    PortfolioExposure,
    StressScenario,
    compute_portfolio_exposure,
    run_stress,
)
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.analysis.research import (
    Direction,
    analyze_underlying,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain, black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.data.candles import CandleCacheError, CandleStore, close_asof, last_close
from options_helper.data.derived import DERIVED_COLUMNS, DERIVED_SCHEMA_VERSION, DerivedStore
from options_helper.data.earnings import EarningsRecord, EarningsStore, safe_next_earnings_date
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.data.options_snapshotter import snapshot_full_chain_for_symbols
from options_helper.data.scanner import (
    evaluate_liquidity_for_symbols,
    prefilter_symbols,
    read_exclude_symbols,
    read_scanned_symbols,
    scan_symbols,
    write_liquidity_csv,
    write_scan_csv,
    write_exclude_symbols,
    write_scanned_symbols,
)
from options_helper.data.technical_backtesting_artifacts import write_artifacts
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.data.technical_backtesting_io import load_ohlc_from_cache, load_ohlc_from_path
from options_helper.data.universe import UniverseError, load_universe_symbols
from options_helper.data.yf_client import DataFetchError, YFinanceClient, contract_row_by_strike
from options_helper.models import OptionType, Position, RiskProfile
from options_helper.reporting import render_positions, render_summary
from options_helper.reporting_chain import (
    render_chain_report_console,
    render_chain_report_markdown,
    render_compare_report_console,
)
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
    render_portfolio_table_markdown,
)
from options_helper.reporting_roll import render_roll_plan_console
from options_helper.storage import load_portfolio, save_portfolio, write_template
from options_helper.watchlists import build_default_watchlists, load_watchlists, save_watchlists
from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
from options_helper.technicals_backtesting.backtest.walk_forward import walk_forward_optimize
from options_helper.technicals_backtesting.extension_percentiles import (
    build_weekly_extension_series,
    compute_extension_percentiles,
    rolling_percentile_rank,
)
from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
from options_helper.technicals_backtesting.max_forward_returns import (
    forward_max_down_move,
    forward_max_up_move,
)
from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
from options_helper.technicals_backtesting.rsi_divergence import compute_rsi_divergence_flags, rsi_regime_tag
from options_helper.technicals_backtesting.snapshot import compute_technical_snapshot
from options_helper.technicals_backtesting.strategies.registry import get_strategy

app = typer.Typer(add_completion=False)
watchlists_app = typer.Typer(help="Manage symbol watchlists.")
app.add_typer(watchlists_app, name="watchlists")
derived_app = typer.Typer(help="Persist derived metrics from local snapshots.")
app.add_typer(derived_app, name="derived")
technicals_app = typer.Typer(help="Technical indicators + backtesting/optimization.")
app.add_typer(technicals_app, name="technicals")
scanner_app = typer.Typer(help="Market opportunity scanner (not financial advice).")
app.add_typer(scanner_app, name="scanner")


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
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for realized volatility).",
    ),
) -> None:
    """Append or upsert a derived-metrics row for a symbol/day (offline)."""
    console = Console(width=200)
    store = OptionsSnapshotStore(cache_dir)
    derived = DerivedStore(derived_dir)
    candle_store = CandleStore(candle_cache_dir)

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

        candles = candle_store.load(symbol)
        history = derived.load(symbol)
        row = DerivedRow.from_chain_report(report, candles=candles, derived_history=history)
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
            metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
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
    client: YFinanceClient | None,
    position: Position,
    *,
    risk_profile: RiskProfile,
    underlying_history: pd.DataFrame,
    underlying_last_price: float | None,
    as_of: date | None = None,
    next_earnings_date: date | None = None,
    snapshot_row: pd.Series | dict | None = None,
) -> PositionMetrics:
    today = as_of or date.today()

    row = snapshot_row
    if row is None:
        if client is None:
            raise ValueError("client is required when snapshot_row is not provided")
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

    quality_label = None
    last_trade_age_days = None
    quality_warnings: list[str] = []
    if row is not None:
        quality_df = compute_quote_quality(
            pd.DataFrame([row]),
            min_volume=risk_profile.min_volume,
            min_open_interest=risk_profile.min_open_interest,
            as_of=today,
        )
        if not quality_df.empty:
            q = quality_df.iloc[0]
            label = q.get("quality_label")
            if label is not None and not pd.isna(label):
                quality_label = str(label)
            age = q.get("last_trade_age_days")
            if age is not None and not pd.isna(age):
                try:
                    last_trade_age_days = int(age)
                except Exception:  # noqa: BLE001
                    last_trade_age_days = None
            warnings_val = q.get("quality_warnings")
            if isinstance(warnings_val, list):
                quality_warnings = [str(w) for w in warnings_val if w]

    mark = _mark_price(bid=bid, ask=ask, last=last)
    spread = spread_pct = None
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        spread = ask - bid
        mid = (ask + bid) / 2.0
        if mid > 0:
            spread_pct = spread / mid
    exec_quality = execution_quality(spread_pct)

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
        spread=spread,
        spread_pct=spread_pct,
        execution_quality=exec_quality,
        last=last,
        implied_vol=iv,
        open_interest=oi,
        volume=vol,
        quality_label=quality_label,
        last_trade_age_days=last_trade_age_days,
        quality_warnings=quality_warnings,
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
        as_of=today,
        next_earnings_date=next_earnings_date,
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
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = OptionsSnapshotStore(cache_dir)
    candle_store = CandleStore(candle_cache_dir)
    client = YFinanceClient()

    want_full_chain = full_chain
    want_all_expiries = all_expiries

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

    required_date: date | None = None
    if require_data_date is not None:
        spec = require_data_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_data_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(
                f"Invalid --require-data-date/--require-data-tz: {exc}",
                param_hint="--require-data-date",
            ) from exc

    mode = "watchlists" if use_watchlists else "portfolio"
    console.print(
        f"Snapshotting options chains for {len(symbols)} symbol(s) "
        f"({mode}, {'full-chain' if want_full_chain else 'windowed'})..."
    )

    # If the user snapshots watchlists with --position-expiries and doesn't explicitly cap expiries,
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

        if required_date is not None and data_date != required_date:
            got = "-" if data_date is None else data_date.isoformat()
            console.print(
                f"[yellow]Warning:[/yellow] {symbol}: candle date {got} != required {required_date.isoformat()}; "
                "skipping snapshot to avoid mis-dated overwrite."
            )
            continue

        effective_snapshot_date = data_date or date.today()
        dates_used.add(effective_snapshot_date)

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        total_contracts = 0
        missing_bid_ask = 0
        stale_quotes = 0
        spread_pcts: list[float] = []

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

                quality = compute_quote_quality(
                    df,
                    min_volume=0,
                    min_open_interest=0,
                    as_of=effective_snapshot_date,
                )
                total_contracts += len(df)
                if not quality.empty:
                    q_warn = quality["quality_warnings"].tolist()
                    missing_bid_ask += sum("quote_missing_bid_ask" in w for w in q_warn if isinstance(w, list))
                    stale_quotes += sum("quote_stale" in w for w in q_warn if isinstance(w, list))
                    spread_series = pd.to_numeric(quality["spread_pct"], errors="coerce")
                    spread_series = spread_series.where(spread_series >= 0)
                    spread_pcts.extend(spread_series.dropna().tolist())

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

        if want_full_chain and total_contracts > 0:
            spread_median = float(np.nanmedian(spread_pcts)) if spread_pcts else None
            spread_worst = float(np.nanmax(spread_pcts)) if spread_pcts else None
            store._upsert_meta(
                store._day_dir(symbol, effective_snapshot_date),
                {
                    "quote_quality": {
                        "contracts": int(total_contracts),
                        "missing_bid_ask_count": int(missing_bid_ask),
                        "missing_bid_ask_pct": float(missing_bid_ask / total_contracts),
                        "spread_pct_median": spread_median,
                        "spread_pct_worst": spread_worst,
                        "stale_quotes": int(stale_quotes),
                        "stale_pct": float(stale_quotes / total_contracts),
                    }
                },
            )

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


@app.command("report-pack")
def report_pack(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON (used for default paths)."),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store (symbol source).",
    ),
    watchlist: list[str] = typer.Option(
        [],
        "--watchlist",
        help="Watchlist name(s) to include (repeatable). Default: positions, monitor, Scanner - Shortlist.",
    ),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technicals artifacts).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (writes {derived_dir}/{SYMBOL}.csv).",
    ),
    out: Path = typer.Option(
        Path("data/reports"),
        "--out",
        help="Output root for saved artifacts (writes under chains/compare/flow/derived/technicals).",
    ),
    as_of: str = typer.Option("latest", "--as-of", help="Snapshot date (YYYY-MM-DD) or 'latest' (per-symbol)."),
    compare_from: str = typer.Option(
        "-1",
        "--compare-from",
        help="Compare spec: -1|-5|YYYY-MM-DD|none (relative offsets are per-symbol).",
    ),
    require_snapshot_date: str | None = typer.Option(
        None,
        "--require-snapshot-date",
        help="Only include symbols whose resolved --as-of snapshot date matches this (YYYY-MM-DD or 'today').",
    ),
    require_snapshot_tz: str = typer.Option(
        "America/Chicago",
        "--require-snapshot-tz",
        help="Timezone used to interpret 'today' for --require-snapshot-date (default: America/Chicago).",
    ),
    chain: bool = typer.Option(True, "--chain/--no-chain", help="Generate chain-report artifacts."),
    compare: bool = typer.Option(True, "--compare/--no-compare", help="Generate compare artifacts."),
    flow: bool = typer.Option(True, "--flow/--no-flow", help="Generate flow artifacts (contract + expiry-strike)."),
    derived: bool = typer.Option(True, "--derived/--no-derived", help="Upsert derived rows + write derived stats artifacts."),
    technicals: bool = typer.Option(
        True,
        "--technicals/--no-technicals",
        help="Generate technicals extension-stats artifacts (offline, from candle cache).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (used for extension-stats artifacts).",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top rows/strikes to include in reports."),
    derived_window: int = typer.Option(60, "--derived-window", min=1, max=3650, help="Derived stats lookback window."),
    derived_trend_window: int = typer.Option(
        5, "--derived-trend-window", min=1, max=3650, help="Derived stats trend lookback window."
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Optional symmetric tail threshold for technicals extension-stats (e.g. 5 => low<=5, high>=95).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Optional rolling window (years) for extension percentiles in technicals extension-stats.",
    ),
) -> None:
    """
    Offline report pack from local snapshots/candles.

    Generates per-symbol artifacts under `--out`:
    - chains/{SYMBOL}/{YYYY-MM-DD}.json + .md
    - compare/{SYMBOL}/{FROM}_to_{TO}.json
    - flow/{SYMBOL}/{FROM}_to_{TO}_w1_{group_by}.json
    - derived/{SYMBOL}/{ASOF}_w{N}_tw{M}.json
    - technicals/extension/{SYMBOL}/{ASOF}.json + .md
    """
    console = Console(width=200)
    _ = load_portfolio(portfolio_path)  # validates the file exists/loads; used by cron scripts.

    wl = load_watchlists(watchlists_path)
    watchlists_used = watchlist[:] if watchlist else ["positions", "monitor", "Scanner - Shortlist"]
    symbols: set[str] = set()
    for name in watchlists_used:
        syms = wl.get(name)
        if not syms:
            console.print(f"[yellow]Warning:[/yellow] watchlist '{name}' missing/empty in {watchlists_path}")
            continue
        symbols.update(syms)

    symbols = {s.strip().upper() for s in symbols if s and s.strip()}
    if not symbols:
        console.print("[yellow]No symbols selected (empty watchlists).[/yellow]")
        raise typer.Exit(0)

    store = OptionsSnapshotStore(cache_dir)
    derived_store = DerivedStore(derived_dir)
    candle_store = CandleStore(candle_cache_dir)

    required_date: date | None = None
    if require_snapshot_date is not None:
        spec = require_snapshot_date.strip().lower()
        try:
            if spec in {"today", "now"}:
                required_date = datetime.now(ZoneInfo(require_snapshot_tz)).date()
            else:
                required_date = date.fromisoformat(spec)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(
                f"Invalid --require-snapshot-date/--require-snapshot-tz: {exc}",
                param_hint="--require-snapshot-date",
            ) from exc

    compare_norm = compare_from.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    out = out.expanduser()
    (out / "chains").mkdir(parents=True, exist_ok=True)
    (out / "compare").mkdir(parents=True, exist_ok=True)
    (out / "flow").mkdir(parents=True, exist_ok=True)
    (out / "derived").mkdir(parents=True, exist_ok=True)
    (out / "technicals" / "extension").mkdir(parents=True, exist_ok=True)

    console.print(
        "Running offline report pack for "
        f"{len(symbols)} symbol(s) from watchlists: {', '.join([repr(x) for x in watchlists_used])}"
    )

    counts = {
        "symbols_total": len(symbols),
        "symbols_ok": 0,
        "chain_ok": 0,
        "compare_ok": 0,
        "flow_ok": 0,
        "derived_ok": 0,
        "technicals_ok": 0,
        "skipped_required_date": 0,
    }

    for sym in sorted(symbols):
        try:
            to_date = store.resolve_date(sym, as_of)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] {sym}: no snapshots ({exc})")
            continue

        if required_date is not None and to_date != required_date:
            counts["skipped_required_date"] += 1
            continue

        df_to = store.load_day(sym, to_date)
        meta_to = store.load_meta(sym, to_date)
        spot_to = _spot_from_meta(meta_to)
        if spot_to is None:
            console.print(f"[yellow]Warning:[/yellow] {sym}: missing spot in meta.json for {to_date.isoformat()}")
            continue

        # 1) Chain report (and derived update/stats based on it)
        chain_report_model = None
        if chain or derived:
            try:
                chain_report_model = compute_chain_report(
                    df_to,
                    symbol=sym,
                    as_of=to_date,
                    spot=spot_to,
                    expiries_mode="near",
                    top=top,
                    best_effort=True,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: chain-report failed: {exc}")
                chain_report_model = None

        if chain and chain_report_model is not None:
            try:
                base = out / "chains" / sym.upper()
                base.mkdir(parents=True, exist_ok=True)
                json_path = base / f"{to_date.isoformat()}.json"
                md_path = base / f"{to_date.isoformat()}.md"
                json_path.write_text(chain_report_model.model_dump_json(indent=2), encoding="utf-8")
                md_path.write_text(render_chain_report_markdown(chain_report_model), encoding="utf-8")
                counts["chain_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: failed writing chain artifacts: {exc}")

        if derived and chain_report_model is not None:
            try:
                candles = candle_store.load(sym)
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain_report_model, candles=candles, derived_history=history)
                derived_store.upsert(sym, row)
                df_derived = derived_store.load(sym)
                if not df_derived.empty:
                    stats = compute_derived_stats(
                        df_derived,
                        symbol=sym,
                        as_of="latest",
                        window=derived_window,
                        trend_window=derived_trend_window,
                        metric_columns=[c for c in DERIVED_COLUMNS if c != "date"],
                    )
                    base = out / "derived" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    stats_path = base / f"{stats.as_of}_w{derived_window}_tw{derived_trend_window}.json"
                    stats_path.write_text(stats.model_dump_json(indent=2), encoding="utf-8")
                    counts["derived_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: derived update/stats failed: {exc}")

        # 2) Compare + flow (requires previous snapshot)
        if compare_enabled and (compare or flow):
            try:
                from_date: date
                if compare_norm.startswith("-") and compare_norm[1:].isdigit():
                    from_date = store.resolve_relative_date(sym, to_date=to_date, offset=int(compare_norm))
                else:
                    from_date = store.resolve_date(sym, compare_norm)

                df_from = store.load_day(sym, from_date)
                meta_from = store.load_meta(sym, from_date)
                spot_from = _spot_from_meta(meta_from)
                if spot_from is None:
                    raise ValueError("missing spot in from-date meta.json")

                if compare:
                    diff, report_from, report_to = compute_compare_report(
                        symbol=sym,
                        from_date=from_date,
                        to_date=to_date,
                        from_df=df_from,
                        to_df=df_to,
                        spot_from=spot_from,
                        spot_to=spot_to,
                        top=top,
                    )
                    base = out / "compare" / sym.upper()
                    base.mkdir(parents=True, exist_ok=True)
                    out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}.json"
                    payload = {
                        "schema_version": 1,
                        "symbol": sym.upper(),
                        "from": report_from.model_dump(),
                        "to": report_to.model_dump(),
                        "diff": diff.model_dump(),
                    }
                    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                    counts["compare_ok"] += 1

                if flow:
                    pair_flow = compute_flow(df_to, df_from, spot=spot_to)
                    if not pair_flow.empty:
                        for group_by in ("contract", "expiry-strike"):
                            net = aggregate_flow_window([pair_flow], group_by=cast(FlowGroupBy, group_by))
                            base = out / "flow" / sym.upper()
                            base.mkdir(parents=True, exist_ok=True)
                            out_path = base / f"{from_date.isoformat()}_to_{to_date.isoformat()}_w1_{group_by}.json"
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
                                "from_date": from_date.isoformat(),
                                "to_date": to_date.isoformat(),
                                "window": 1,
                                "group_by": group_by,
                                "snapshot_dates": [from_date.isoformat(), to_date.isoformat()],
                                "net": artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                            }
                            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                        counts["flow_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: compare/flow skipped: {exc}")

        # 3) Technicals extension stats artifacts (offline, candle cache)
        if technicals:
            try:
                technicals_extension_stats(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=technicals_config,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=out / "technicals" / "extension",
                    write_json=True,
                    write_md=True,
                    print_to_console=False,
                )
                counts["technicals_ok"] += 1
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: technicals extension-stats failed: {exc}")

        counts["symbols_ok"] += 1

    console.print(
        "Report pack complete: "
        f"symbols ok={counts['symbols_ok']}/{counts['symbols_total']} | "
        f"chain={counts['chain_ok']} compare={counts['compare_ok']} flow={counts['flow_ok']} "
        f"derived={counts['derived_ok']} technicals={counts['technicals_ok']} | "
        f"skipped(required_date)={counts['skipped_required_date']}"
    )


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
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles (used for technical context).",
    ),
    technicals_config: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--technicals-config",
        help="Technical backtesting config (canonical indicator definitions).",
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output path (Markdown) or directory. Default: data/reports/daily/{ASOF}.md",
    ),
    print_to_console: bool = typer.Option(
        False,
        "--print/--no-print",
        help="Print the briefing to the console (in addition to writing files).",
    ),
    write_json: bool = typer.Option(
        True,
        "--write-json/--no-write-json",
        help="Write a JSON version of the briefing alongside the Markdown (LLM-friendly).",
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
    rp = portfolio.risk_profile

    positions_by_symbol: dict[str, list[Position]] = {}
    for p in portfolio.positions:
        positions_by_symbol.setdefault(p.symbol.upper(), []).append(p)

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
    candle_store = CandleStore(candle_cache_dir)
    earnings_store = EarningsStore(Path("data/earnings"))

    technicals_cfg: dict | None = None
    technicals_cfg_error: str | None = None
    try:
        technicals_cfg = load_technical_backtesting_config(technicals_config)
    except Exception as exc:  # noqa: BLE001
        technicals_cfg_error = str(exc)

    # Cache day snapshots for portfolio marks (best-effort).
    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}

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
        technicals = None
        candles = None
        derived_updated = False
        derived_row = None
        quote_quality = None
        next_earnings_date = safe_next_earnings_date(earnings_store, sym)
        next_earnings_by_symbol[sym] = next_earnings_date

        try:
            to_date = store.resolve_date(sym, as_of)
            resolved_to_dates.append(to_date)

            event_warnings: set[str] = set()
            base_risk = earnings_event_risk(
                today=to_date,
                expiry=None,
                next_earnings_date=next_earnings_date,
                warn_days=rp.earnings_warn_days,
                avoid_days=rp.earnings_avoid_days,
            )
            event_warnings.update(base_risk["warnings"])
            for pos in positions_by_symbol.get(sym, []):
                pos_risk = earnings_event_risk(
                    today=to_date,
                    expiry=pos.expiry,
                    next_earnings_date=next_earnings_date,
                    warn_days=rp.earnings_warn_days,
                    avoid_days=rp.earnings_avoid_days,
                )
                event_warnings.update(pos_risk["warnings"])
            if event_warnings:
                warnings.extend(sorted(event_warnings))

            df_to = store.load_day(sym, to_date)
            meta_to = store.load_meta(sym, to_date)
            spot_to = _spot_from_meta(meta_to)
            quote_quality = meta_to.get("quote_quality") if isinstance(meta_to, dict) else None
            if spot_to is None:
                raise ValueError("missing spot price in meta.json (run snapshot-options first)")

            day_cache[sym] = (to_date, df_to)

            if technicals_cfg is None:
                if technicals_cfg_error is not None:
                    warnings.append(f"technicals unavailable: {technicals_cfg_error}")
            else:
                try:
                    candles = candle_store.load(sym)
                    if candles.empty:
                        warnings.append("technicals unavailable: missing candle cache (run refresh-candles)")
                    else:
                        technicals = compute_technical_snapshot(candles, technicals_cfg)
                        if technicals is None:
                            warnings.append("technicals unavailable: insufficient candle history / warmup")
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"technicals unavailable: {exc}")

            if candles is None:
                candles = pd.DataFrame()
            candles_by_symbol[sym] = candles

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
                if candles is None:
                    candles = candle_store.load(sym)
                history = derived_store.load(sym)
                row = DerivedRow.from_chain_report(chain, candles=candles, derived_history=history)
                try:
                    derived_store.upsert(sym, row)
                    derived_updated = True
                    derived_row = row
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
                technicals=technicals,
                errors=errors,
                warnings=warnings,
                quote_quality=quote_quality,
                derived_updated=derived_updated,
                derived=derived_row,
                next_earnings_date=next_earnings_date,
            )
        )

    if not resolved_to_dates:
        console.print("[red]Error:[/red] no snapshots found for selected symbols")
        raise typer.Exit(1)
    report_date = max(resolved_to_dates).isoformat()
    portfolio_rows: list[dict[str, str]] = []
    portfolio_rows_with_pnl: list[tuple[float, dict[str, str]]] = []
    portfolio_metrics: list[PositionMetrics] = []
    for p in portfolio.positions:
        sym = p.symbol.upper()
        to_date, df_to = day_cache.get(sym, (None, pd.DataFrame()))

        mark = None
        spr_pct = None
        snapshot_row = None
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
                snapshot_row = sub.iloc[0]
                bid = _extract_float(snapshot_row, "bid")
                ask = _extract_float(snapshot_row, "ask")
                mark = _mark_price(bid=bid, ask=ask, last=_extract_float(snapshot_row, "lastPrice"))
                if bid is not None and ask is not None and bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    if mid > 0:
                        spr_pct = (ask - bid) / mid

        history = candles_by_symbol.get(sym)
        if history is None or history.empty:
            try:
                history = candle_store.load(sym)
            except Exception:  # noqa: BLE001
                history = pd.DataFrame()
            candles_by_symbol[sym] = history

        try:
            last_price = close_asof(history, to_date) if to_date is not None else last_close(history)
            metrics = _position_metrics(
                None,
                p,
                risk_profile=rp,
                underlying_history=history,
                underlying_last_price=last_price,
                as_of=to_date,
                next_earnings_date=next_earnings_by_symbol.get(sym),
                snapshot_row=snapshot_row if snapshot_row is not None else {},
            )
            portfolio_metrics.append(metrics)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] portfolio exposure skipped for {p.id}: {exc}")

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
                "spr_%": "-" if spr_pct is None else f"{spr_pct * 100.0:.1f}%",
                "as_of": "-" if to_date is None else to_date.isoformat(),
            }
        )
        pnl_sort = float(pnl_pct) if pnl_pct is not None else float("-inf")
        portfolio_rows_with_pnl.append((pnl_sort, portfolio_rows[-1]))

    portfolio_table_md = None
    if portfolio_rows:
        # Sort by pnl% descending; rows without pnl% go last.
        portfolio_rows_sorted = [row for _, row in sorted(portfolio_rows_with_pnl, key=lambda r: r[0], reverse=True)]
        include_spread = any(r.get("spr_%") not in (None, "-") for r in portfolio_rows_sorted)
        portfolio_table_md = render_portfolio_table_markdown(portfolio_rows_sorted, include_spread=include_spread)

    portfolio_exposure = None
    portfolio_stress = None
    if portfolio_metrics:
        portfolio_exposure = compute_portfolio_exposure(portfolio_metrics)
        portfolio_stress = run_stress(
            portfolio_exposure,
            _build_stress_scenarios(stress_spot_pct=[], stress_vol_pp=5.0, stress_days=7),
        )

    md = render_briefing_markdown(
        report_date=report_date,
        portfolio_path=str(portfolio_path),
        symbol_sections=sections,
        portfolio_table_md=portfolio_table_md,
        top=top,
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

    if write_json:
        payload = build_briefing_payload(
            report_date=report_date,
            portfolio_path=str(portfolio_path),
            symbol_sections=sections,
            top=top,
            technicals_config=str(technicals_config),
            portfolio_exposure=portfolio_exposure,
            portfolio_stress=portfolio_stress,
        )
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
        )
        console.print(f"Saved: {json_path}")

    if print_to_console:
        try:
            from rich.markdown import Markdown

            console.print(Markdown(md))
        except Exception:  # noqa: BLE001
            console.print(md)


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
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
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
    earnings_store = EarningsStore(Path("data/earnings"))
    next_earnings_date = safe_next_earnings_date(earnings_store, position.symbol)

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
            next_earnings_date=next_earnings_date,
            earnings_warn_days=rp.earnings_warn_days,
            earnings_avoid_days=rp.earnings_avoid_days,
            include_bad_quotes=include_bad_quotes,
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
    include_bad_quotes: bool = typer.Option(
        False,
        "--include-bad-quotes",
        help="Include candidates with bad quote quality (best-effort).",
    ),
    derived_dir: Path = typer.Option(
        Path("data/derived"),
        "--derived-dir",
        help="Directory for derived metric files (used for IV percentile context).",
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
    earnings_store = EarningsStore(Path("data/earnings"))
    derived_store = DerivedStore(derived_dir)
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
        as_of_date = symbol_candle_dates.get(sym)
        next_earnings_date = safe_next_earnings_date(earnings_store, sym)
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

        expiry_as_of = as_of_date or date.today()
        short_exp = choose_expiry(
            expiry_strs, min_dte=short_min_dte, max_dte=short_max_dte, target_dte=60, today=expiry_as_of
        )
        long_exp = choose_expiry(
            expiry_strs, min_dte=long_min_dte, max_dte=long_max_dte, target_dte=540, today=expiry_as_of
        )
        if long_exp is None:
            # Fallback: pick the farthest expiry that still qualifies as "long".
            parsed = []
            for s in expiry_strs:
                try:
                    exp = date.fromisoformat(s)
                except ValueError:
                    continue
                dte = (exp - expiry_as_of).days
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
        derived_history = derived_store.load(sym)
        vol_context = None

        table = Table(title=f"{sym} option ideas (best-effort)")
        table.add_column("Horizon")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Type")
        table.add_column("Strike", justify="right")
        table.add_column("Entry", justify="right")
        table.add_column("Δ", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("IV/RV20", justify="right")
        table.add_column("IV pct", justify="right")
        table.add_column("OI", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("Spr%", justify="right")
        table.add_column("Exec", justify="right")
        table.add_column("Quality", justify="right")
        table.add_column("Stale", justify="right")
        table.add_column("Why")

        def _fmt_stale(age_days: int | None) -> str:
            if age_days is None:
                return "-"
            age = int(age_days)
            return f"{age}d" if age > 5 else "-"

        def _fmt_iv_rv(ctx) -> str:  # type: ignore[no-untyped-def]
            if ctx is None or ctx.iv_rv_20d is None:
                return "-"
            return f"{ctx.iv_rv_20d:.2f}x"

        def _fmt_iv_pct(ctx) -> str:  # type: ignore[no-untyped-def]
            if ctx is None or ctx.iv_percentile is None:
                return "-"
            return f"{ctx.iv_percentile:.0f}"

        if short_exp is not None:
            chain = client.get_options_chain(sym, short_exp)
            if vol_context is None:
                vol_context = compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
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
                as_of=expiry_as_of,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
                include_bad_quotes=include_bad_quotes,
            )
            if short_pick is not None:
                if short_pick.exclude:
                    warn = ", ".join(short_pick.warnings) if short_pick.warnings else "earnings_unknown"
                    emit(f"  - Excluded 30–90d candidate due to earnings_avoid_days ({warn}).")
                else:
                    why = "; ".join(short_pick.rationale[:2])
                    if short_pick.warnings:
                        why = f"{why}; Warnings: {', '.join(short_pick.warnings)}"
                    table.add_row(
                        "30–90d",
                        short_pick.expiry.isoformat(),
                        str(short_pick.dte),
                        short_pick.option_type,
                        f"{short_pick.strike:g}",
                        "-" if short_pick.mark is None else f"${short_pick.mark:.2f}",
                        "-" if short_pick.delta is None else f"{short_pick.delta:+.2f}",
                        "-" if short_pick.iv is None else f"{short_pick.iv:.1%}",
                        _fmt_iv_rv(vol_context),
                        _fmt_iv_pct(vol_context),
                        "-" if short_pick.open_interest is None else str(short_pick.open_interest),
                        "-" if short_pick.volume is None else str(short_pick.volume),
                        "-" if short_pick.spread_pct is None else f"{short_pick.spread_pct:.1%}",
                        "-" if short_pick.execution_quality is None else short_pick.execution_quality,
                        "-" if short_pick.quality_label is None else short_pick.quality_label,
                        _fmt_stale(short_pick.last_trade_age_days),
                        why,
                    )
                    if short_pick.warnings:
                        emit(f"  - Earnings warnings (30–90d): {', '.join(short_pick.warnings)}")
                    if short_pick.quality_warnings:
                        emit(f"  - Quote warnings (30–90d): {', '.join(short_pick.quality_warnings)}")
        else:
            emit(f"  - No expiries found in {short_min_dte}-{short_max_dte} DTE range.")

        if long_exp is not None:
            chain = client.get_options_chain(sym, long_exp)
            if vol_context is None:
                vol_context = compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
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
                as_of=expiry_as_of,
                next_earnings_date=next_earnings_date,
                earnings_warn_days=rp.earnings_warn_days,
                earnings_avoid_days=rp.earnings_avoid_days,
                include_bad_quotes=include_bad_quotes,
            )
            if long_pick is not None:
                if long_pick.exclude:
                    warn = ", ".join(long_pick.warnings) if long_pick.warnings else "earnings_unknown"
                    emit(f"  - Excluded LEAPS candidate due to earnings_avoid_days ({warn}).")
                else:
                    why = "; ".join(long_pick.rationale[:2] + ["Longer DTE reduces theta pressure."])
                    if long_pick.warnings:
                        why = f"{why}; Warnings: {', '.join(long_pick.warnings)}"
                    table.add_row(
                        "LEAPS",
                        long_pick.expiry.isoformat(),
                        str(long_pick.dte),
                        long_pick.option_type,
                        f"{long_pick.strike:g}",
                        "-" if long_pick.mark is None else f"${long_pick.mark:.2f}",
                        "-" if long_pick.delta is None else f"{long_pick.delta:+.2f}",
                        "-" if long_pick.iv is None else f"{long_pick.iv:.1%}",
                        _fmt_iv_rv(vol_context),
                        _fmt_iv_pct(vol_context),
                        "-" if long_pick.open_interest is None else str(long_pick.open_interest),
                        "-" if long_pick.volume is None else str(long_pick.volume),
                        "-" if long_pick.spread_pct is None else f"{long_pick.spread_pct:.1%}",
                        "-" if long_pick.execution_quality is None else long_pick.execution_quality,
                        "-" if long_pick.quality_label is None else long_pick.quality_label,
                        _fmt_stale(long_pick.last_trade_age_days),
                        why,
                    )
                    if long_pick.warnings:
                        emit(f"  - Earnings warnings (LEAPS): {', '.join(long_pick.warnings)}")
                    if long_pick.quality_warnings:
                        emit(f"  - Quote warnings (LEAPS): {', '.join(long_pick.quality_warnings)}")
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


@scanner_app.command("run")
def scanner_run(
    universe: str = typer.Option(
        "file:data/universe/sec_company_tickers.json",
        "--universe",
        help="Universe source: us-all/us-equities/us-etfs or file:/path/to/list.txt.",
    ),
    universe_cache_dir: Path = typer.Option(
        Path("data/universe"),
        "--universe-cache-dir",
        help="Directory for cached universe lists.",
    ),
    universe_refresh_days: int = typer.Option(
        1,
        "--universe-refresh-days",
        help="Refresh universe cache if older than this many days.",
    ),
    max_symbols: int | None = typer.Option(
        None,
        "--max-symbols",
        min=1,
        help="Optional cap on number of symbols scanned (for dev/testing).",
    ),
    prefilter_mode: str = typer.Option(
        "default",
        "--prefilter-mode",
        help="Prefilter mode: default, aggressive, or none.",
    ),
    exclude_path: Path = typer.Option(
        Path("data/universe/exclude_symbols.txt"),
        "--exclude-path",
        help="Path to exclude symbols file (one ticker per line).",
    ),
    scanned_path: Path = typer.Option(
        Path("data/scanner/scanned_symbols.txt"),
        "--scanned-path",
        help="Path to scanned symbols file (one ticker per line).",
    ),
    skip_scanned: bool = typer.Option(
        True,
        "--skip-scanned/--no-skip-scanned",
        help="Skip symbols already recorded in the scanned file.",
    ),
    write_scanned: bool = typer.Option(
        True,
        "--write-scanned/--no-write-scanned",
        help="Persist scanned symbols so future runs skip them.",
    ),
    write_error_excludes: bool = typer.Option(
        True,
        "--write-error-excludes/--no-write-error-excludes",
        help="Persist symbols that error to the exclude file.",
    ),
    exclude_statuses: str = typer.Option(
        "error,no_candles",
        "--exclude-statuses",
        help="Comma-separated scan statuses to add to the exclude file.",
    ),
    error_flush_every: int = typer.Option(
        50,
        "--error-flush-every",
        min=1,
        help="Flush exclude file after this many new error symbols.",
    ),
    scanned_flush_every: int = typer.Option(
        250,
        "--scanned-flush-every",
        min=1,
        help="Flush scanned file after this many new symbols.",
    ),
    scan_period: str = typer.Option(
        "max",
        "--scan-period",
        help="Candle period to pull for the scan (yfinance period format).",
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Symmetric tail threshold percentile (e.g. 2.5 => low<=2.5, high>=97.5).",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Rolling window (years) for extension percentiles (default: auto 1y/3y).",
    ),
    watchlists_path: Path = typer.Option(
        Path("data/watchlists.json"),
        "--watchlists-path",
        help="Path to watchlists JSON store.",
    ),
    all_watchlist_name: str = typer.Option(
        "Scanner - All",
        "--all-watchlist-name",
        help="Watchlist name for all tail symbols (replaced each run).",
    ),
    shortlist_watchlist_name: str = typer.Option(
        "Scanner - Shortlist",
        "--shortlist-watchlist-name",
        help="Watchlist name for liquid short list (replaced each run).",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Directory for cached daily candles.",
    ),
    options_cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--options-cache-dir",
        help="Directory for options chain snapshots.",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot from daily candles for snapshotting.",
    ),
    backfill: bool = typer.Option(
        True,
        "--backfill/--no-backfill",
        help="Backfill max candle history for tail symbols.",
    ),
    snapshot_options: bool = typer.Option(
        True,
        "--snapshot-options/--no-snapshot-options",
        help="Snapshot full options chain for all expiries on tail symbols.",
    ),
    risk_free_rate: float = typer.Option(
        0.0,
        "--risk-free-rate",
        help="Risk-free rate used for best-effort Black-Scholes Greeks (e.g. 0.05 = 5%).",
    ),
    liquidity_min_dte: int = typer.Option(
        60,
        "--liquidity-min-dte",
        help="Minimum DTE for liquidity screening.",
    ),
    liquidity_min_volume: int = typer.Option(
        10,
        "--liquidity-min-volume",
        help="Minimum volume for liquidity screening.",
    ),
    liquidity_min_oi: int = typer.Option(
        500,
        "--liquidity-min-oi",
        help="Minimum open interest for liquidity screening.",
    ),
    run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--run-dir",
        help="Output root for scanner runs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run id (default: timestamp).",
    ),
    workers: int | None = typer.Option(
        None,
        "--workers",
        min=1,
        help="Max concurrent workers for scan (default: auto).",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        min=1,
        help="Batch size for scan requests.",
    ),
    batch_sleep_seconds: float = typer.Option(
        0.25,
        "--batch-sleep-seconds",
        min=0.0,
        help="Sleep between batches (seconds) to be polite to data sources.",
    ),
    reports_out: Path = typer.Option(
        Path("data/reports/technicals/extension"),
        "--reports-out",
        help="Output root for Extension Percentile Stats reports.",
    ),
    run_reports: bool = typer.Option(
        True,
        "--run-reports/--no-run-reports",
        help="Generate Extension Percentile Stats reports for shortlist symbols.",
    ),
    write_scan: bool = typer.Option(
        True,
        "--write-scan/--no-write-scan",
        help="Write scan CSV under the run directory.",
    ),
    write_liquidity: bool = typer.Option(
        True,
        "--write-liquidity/--no-write-liquidity",
        help="Write liquidity CSV under the run directory.",
    ),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"),
        "--config",
        help="Config path.",
    ),
) -> None:
    """Scan the market for extension tails and build watchlists (not financial advice)."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    ext_cfg = cfg.get("extension_percentiles", {})
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_low_pct = tail_low_cfg
        tail_high_pct = tail_high_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp

    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")

    try:
        symbols = load_universe_symbols(
            universe,
            cache_dir=universe_cache_dir,
            refresh_days=universe_refresh_days,
        )
    except UniverseError as exc:
        console.print(f"[red]Universe error:[/red] {exc}")
        raise typer.Exit(1)

    symbols = sorted({s.strip().upper() for s in symbols if s and s.strip()})

    exclude_symbols = read_exclude_symbols(exclude_path) if exclude_path else set()
    if exclude_symbols:
        console.print(f"Loaded {len(exclude_symbols)} excluded symbol(s) from {exclude_path}")

    scanned_symbols: set[str] = set()
    if scanned_path and (skip_scanned or write_scanned):
        scanned_symbols = read_scanned_symbols(scanned_path)
        if scanned_symbols:
            console.print(f"Loaded {len(scanned_symbols)} scanned symbol(s) from {scanned_path}")

    filtered, dropped = prefilter_symbols(
        symbols,
        mode=prefilter_mode,
        exclude=exclude_symbols,
        scanned=scanned_symbols if skip_scanned else None,
    )
    dropped_n = sum(dropped.values())
    if dropped_n:
        console.print(f"Prefiltered symbols: dropped {dropped_n} ({dropped})")
    symbols = filtered

    if max_symbols is not None:
        symbols = symbols[: int(max_symbols)]

    if not symbols:
        console.print("[yellow]No symbols found in universe.[/yellow]")
        raise typer.Exit(0)

    run_stamp = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_root = run_dir / run_stamp
    run_root.mkdir(parents=True, exist_ok=True)

    new_error_symbols: set[str] = set()
    new_scanned_symbols: set[str] = set()

    status_set = {s.strip().lower() for s in exclude_statuses.split(",") if s.strip()}

    def _row_callback(row) -> None:  # noqa: ANN001
        if write_scanned and scanned_path is not None:
            sym = row.symbol
            if sym not in scanned_symbols:
                scanned_symbols.add(sym)
                new_scanned_symbols.add(sym)
                if len(new_scanned_symbols) >= int(scanned_flush_every):
                    write_scanned_symbols(scanned_path, scanned_symbols)
                    new_scanned_symbols.clear()
        if not write_error_excludes or exclude_path is None:
            return
        if str(row.status).strip().lower() not in status_set:
            return
        sym = row.symbol
        if sym not in exclude_symbols:
            exclude_symbols.add(sym)
            new_error_symbols.add(sym)
            if len(new_error_symbols) >= int(error_flush_every):
                write_exclude_symbols(exclude_path, exclude_symbols)
                new_error_symbols.clear()

    console.print(
        f"Scanning {len(symbols)} symbol(s) from `{universe}` (tail {tail_low_pct:.1f}/{tail_high_pct:.1f})..."
    )
    candle_store = CandleStore(candle_cache_dir)
    scan_rows, tail_symbols = scan_symbols(
        symbols,
        candle_store=candle_store,
        cfg=cfg,
        scan_period=scan_period,
        tail_low_pct=float(tail_low_pct),
        tail_high_pct=float(tail_high_pct),
        percentile_window_years=percentile_window_years,
        workers=workers,
        batch_size=batch_size,
        batch_sleep_seconds=batch_sleep_seconds,
        row_callback=_row_callback,
    )

    if write_error_excludes and new_error_symbols and exclude_path is not None:
        write_exclude_symbols(exclude_path, exclude_symbols)
        console.print(f"Wrote {len(new_error_symbols)} new excluded symbol(s) to {exclude_path}")

    if write_scanned and new_scanned_symbols and scanned_path is not None:
        write_scanned_symbols(scanned_path, scanned_symbols)
        console.print(f"Wrote {len(new_scanned_symbols)} new scanned symbol(s) to {scanned_path}")

    if write_scan:
        scan_path = run_root / "scan.csv"
        write_scan_csv(scan_rows, scan_path)
        console.print(f"Wrote scan CSV: {scan_path}")

    wl = load_watchlists(watchlists_path)
    wl.set(all_watchlist_name, tail_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{all_watchlist_name}` ({len(tail_symbols)} symbol(s))")

    if not tail_symbols:
        wl.set(shortlist_watchlist_name, [])
        save_watchlists(watchlists_path, wl)
        console.print("[yellow]No tail symbols found; shortlist cleared.[/yellow]")
        if write_liquidity:
            liquidity_path = run_root / "liquidity.csv"
            write_liquidity_csv([], liquidity_path)
            console.print(f"Wrote liquidity CSV: {liquidity_path}")
        shortlist_md = run_root / "shortlist.md"
        lines = [
            f"# Scanner Shortlist — {run_stamp}",
            "",
            f"- Universe: `{universe}`",
            f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
            f"- Tail watchlist: `{all_watchlist_name}`",
            f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
            "- Symbols: `0`",
            "",
            "Not financial advice.",
            "",
            "## Symbols",
            "- (empty)",
        ]
        shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        console.print(f"Wrote shortlist summary: {shortlist_md}")
        console.print("Not financial advice.")
        return

    if backfill:
        console.print(f"Backfilling candles for {len(tail_symbols)} tail symbol(s)...")
        for sym in tail_symbols:
            try:
                candle_store.get_daily_history(sym, period="max")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: candle backfill failed: {exc}")

    if snapshot_options:
        console.print(f"Snapshotting full options chains for {len(tail_symbols)} tail symbol(s)...")
        snapshot_results = snapshot_full_chain_for_symbols(
            tail_symbols,
            cache_dir=options_cache_dir,
            candle_cache_dir=candle_cache_dir,
            spot_period=spot_period,
            max_expiries=None,
            risk_free_rate=risk_free_rate,
            symbol_source="scanner",
            watchlists=[all_watchlist_name],
        )
        ok = sum(1 for r in snapshot_results if r.status == "ok")
        console.print(f"Options snapshots complete: {ok}/{len(snapshot_results)} ok")

    options_store = OptionsSnapshotStore(options_cache_dir)
    liquidity_rows, shortlist_symbols = evaluate_liquidity_for_symbols(
        tail_symbols,
        store=options_store,
        min_dte=liquidity_min_dte,
        min_volume=liquidity_min_volume,
        min_open_interest=liquidity_min_oi,
    )

    if write_liquidity:
        liquidity_path = run_root / "liquidity.csv"
        write_liquidity_csv(liquidity_rows, liquidity_path)
        console.print(f"Wrote liquidity CSV: {liquidity_path}")

    wl.set(shortlist_watchlist_name, shortlist_symbols)
    save_watchlists(watchlists_path, wl)
    console.print(f"Updated watchlist `{shortlist_watchlist_name}` ({len(shortlist_symbols)} symbol(s))")

    if run_reports and shortlist_symbols:
        console.print(f"Running Extension Percentile Stats for {len(shortlist_symbols)} symbol(s)...")
        for sym in shortlist_symbols:
            try:
                technicals_extension_stats(
                    symbol=sym,
                    ohlc_path=None,
                    cache_dir=candle_cache_dir,
                    config_path=config_path,
                    tail_pct=tail_pct,
                    percentile_window_years=percentile_window_years,
                    out=reports_out,
                    write_json=True,
                    write_md=True,
                    print_to_console=False,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Warning:[/yellow] {sym}: extension-stats failed: {exc}")

    shortlist_md = run_root / "shortlist.md"
    lines = [
        f"# Scanner Shortlist — {run_stamp}",
        "",
        f"- Universe: `{universe}`",
        f"- Tail threshold: `{tail_low_pct:.1f}` / `{tail_high_pct:.1f}`",
        f"- Tail watchlist: `{all_watchlist_name}`",
        f"- Shortlist watchlist: `{shortlist_watchlist_name}`",
        f"- Symbols: `{len(shortlist_symbols)}`",
        "",
        "Not financial advice.",
        "",
        "## Symbols",
    ]
    if shortlist_symbols:
        for sym in shortlist_symbols:
            lines.append(f"- `{sym}` → `{reports_out / sym}`")
    else:
        lines.append("- (empty)")
    shortlist_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    console.print(f"Wrote shortlist summary: {shortlist_md}")
    console.print("Not financial advice.")


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
    if interval != "1d":
        raise typer.BadParameter("Only --interval 1d is supported for now (cache uses daily candles).")

    portfolio = load_portfolio(portfolio_path)
    console = Console()
    render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    snapshot_store: OptionsSnapshotStore | None = None
    if offline:
        snapshot_store = OptionsSnapshotStore(snapshots_dir)

    client = None if offline else YFinanceClient()
    candle_store = CandleStore(cache_dir)
    earnings_store = EarningsStore(Path("data/earnings"))

    history_by_symbol: dict[str, pd.DataFrame] = {}
    last_price_by_symbol: dict[str, float | None] = {}
    as_of_by_symbol: dict[str, date | None] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}
    snapshot_day_by_symbol: dict[str, pd.DataFrame] = {}
    for sym in sorted({p.symbol for p in portfolio.positions}):
        history = pd.DataFrame()
        snapshot_date: date | None = None

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
                close_asof(history, snapshot_date) if snapshot_date is not None else last_close(history)
            )
        else:
            last_price_by_symbol[sym] = last_close(history)
            as_of_date = None
            if not history.empty and isinstance(history.index, pd.DatetimeIndex):
                last_ts = history.index.max()
                if last_ts is not None and not pd.isna(last_ts):
                    as_of_date = last_ts.date()
            as_of_by_symbol[sym] = as_of_date

        next_earnings_by_symbol[sym] = safe_next_earnings_date(earnings_store, sym)

    metrics_list: list[PositionMetrics] = []
    advice_by_id: dict[str, Advice] = {}
    offline_missing: list[str] = []

    for p in portfolio.positions:
        try:
            snapshot_row = None
            if offline:
                snap_date = as_of_by_symbol.get(p.symbol)
                df_snap = snapshot_day_by_symbol.get(p.symbol, pd.DataFrame())
                row = None

                if snap_date is None:
                    offline_missing.append(f"{p.id}: missing offline as-of date for {p.symbol}")
                elif df_snap.empty:
                    offline_missing.append(
                        f"{p.id}: missing snapshot day data for {p.symbol} (as-of {snap_date.isoformat()})"
                    )
                else:
                    row = find_snapshot_row(
                        df_snap,
                        expiry=p.expiry,
                        strike=p.strike,
                        option_type=p.option_type,
                    )
                    if row is None:
                        offline_missing.append(
                            f"{p.id}: missing snapshot row for {p.symbol} {p.expiry.isoformat()} "
                            f"{p.option_type} {p.strike:g} (as-of {snap_date.isoformat()})"
                        )

                snapshot_row = row if row is not None else {}

            metrics = _position_metrics(
                client,
                p,
                risk_profile=portfolio.risk_profile,
                underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                underlying_last_price=last_price_by_symbol.get(p.symbol),
                as_of=as_of_by_symbol.get(p.symbol),
                next_earnings_date=next_earnings_by_symbol.get(p.symbol),
                snapshot_row=snapshot_row,
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

    exposure = compute_portfolio_exposure(metrics_list)
    _render_portfolio_risk(
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
        try:
            return load_ohlc_from_cache(
                symbol,
                cache_dir,
                backfill_if_missing=True,
                period="max",
                raise_on_backfill_error=True,
            )
        except CandleCacheError as exc:
            raise typer.BadParameter(f"Failed to backfill OHLC for {symbol}: {exc}") from exc
    raise typer.BadParameter("Provide --ohlc-path or --symbol/--cache-dir")


def _stats_to_dict(stats: object | None) -> dict | None:
    if stats is None:
        return None
    if isinstance(stats, pd.Series):
        return {k: v for k, v in stats.to_dict().items() if not str(k).startswith("_")}
    if isinstance(stats, dict):
        return {k: v for k, v in stats.items() if not str(k).startswith("_")}
    return {"value": stats}


def _normalize_pct(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _normalize_pp(val: float) -> float:
    try:
        val_f = float(val)
    except Exception:  # noqa: BLE001
        return 0.0
    return val_f / 100.0 if abs(val_f) > 1.0 else val_f


def _build_stress_scenarios(
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> list[StressScenario]:
    scenarios: list[StressScenario] = []
    spot_values = stress_spot_pct or [0.05]
    seen_spot: set[float] = set()
    for raw in spot_values:
        pct = abs(_normalize_pct(raw))
        if pct <= 0 or pct in seen_spot:
            continue
        seen_spot.add(pct)
        scenarios.append(StressScenario(name=f"Spot {pct:+.0%}", spot_pct=pct))
        scenarios.append(StressScenario(name=f"Spot {-pct:+.0%}", spot_pct=-pct))

    vol_pp = _normalize_pp(stress_vol_pp)
    if vol_pp != 0:
        pp_label = vol_pp * 100.0
        scenarios.append(StressScenario(name=f"IV {pp_label:+.1f}pp", vol_pp=vol_pp))
        scenarios.append(StressScenario(name=f"IV {-pp_label:+.1f}pp", vol_pp=-vol_pp))

    if stress_days > 0:
        scenarios.append(StressScenario(name=f"Time +{stress_days}d", days=stress_days))

    return scenarios


def _render_portfolio_risk(
    console: Console,
    exposure: PortfolioExposure,
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> None:
    from rich.table import Table

    def _fmt_num(val: float | None, *, digits: int = 2) -> str:
        if val is None:
            return "-"
        return f"{val:,.{digits}f}"

    def _fmt_money(val: float | None) -> str:
        if val is None:
            return "-"
        return f"${val:,.2f}"

    def _fmt_pct(val: float | None) -> str:
        if val is None:
            return "-"
        return f"{val:.1%}"

    table = Table(title="Portfolio Greeks (best-effort)")
    table.add_column("As-of")
    table.add_column("Delta (shares)", justify="right")
    table.add_column("Theta/day ($)", justify="right")
    table.add_column("Vega ($/IV)", justify="right")
    table.add_row(
        "-" if exposure.as_of is None else exposure.as_of.isoformat(),
        _fmt_num(exposure.total_delta_shares),
        _fmt_money(exposure.total_theta_dollars_per_day),
        _fmt_money(exposure.total_vega_dollars_per_iv),
    )
    console.print(table)

    if exposure.assumptions:
        console.print("Assumptions: " + "; ".join(exposure.assumptions))
    if exposure.warnings:
        console.print("[yellow]Warnings:[/yellow] " + "; ".join(exposure.warnings))

    scenarios = _build_stress_scenarios(
        stress_spot_pct=stress_spot_pct,
        stress_vol_pp=stress_vol_pp,
        stress_days=stress_days,
    )
    if not scenarios:
        return

    stress_results = run_stress(exposure, scenarios)
    stress_table = Table(title="Portfolio Stress (best-effort)")
    stress_table.add_column("Scenario")
    stress_table.add_column("PnL $", justify="right")
    stress_table.add_column("PnL %", justify="right")
    stress_table.add_column("Notes")

    for result in stress_results:
        notes = ", ".join(result.warnings) if result.warnings else "-"
        stress_table.add_row(
            result.name,
            _fmt_money(result.pnl),
            _fmt_pct(result.pnl_pct),
            notes,
        )
    console.print(stress_table)


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


@technicals_app.command("extension-stats")
def technicals_extension_stats(
    symbol: str | None = typer.Option(None, "--symbol", help="Symbol to load from cache."),
    ohlc_path: Path | None = typer.Option(None, "--ohlc-path", help="CSV/parquet OHLC path."),
    cache_dir: Path = typer.Option(Path("data/candles"), "--cache-dir", help="Candle cache dir."),
    config_path: Path = typer.Option(
        Path("config/technical_backtesting.yaml"), "--config", help="Config path."
    ),
    tail_pct: float | None = typer.Option(
        None,
        "--tail-pct",
        help="Symmetric tail threshold percentile (e.g. 5 => low<=5, high>=95). Overrides config tail_high_pct/tail_low_pct.",
    ),
    percentile_window_years: int | None = typer.Option(
        None,
        "--percentile-window-years",
        help="Rolling window (years) for extension percentiles + tail events. Default: auto (1y if <5y history, else 3y).",
    ),
    out: Path | None = typer.Option(
        Path("data/reports/technicals/extension"),
        "--out",
        help="Output root for extension stats artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write JSON artifact."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write Markdown artifact."),
    print_to_console: bool = typer.Option(False, "--print/--no-print", help="Print Markdown to console."),
    divergence_window_days: int = typer.Option(
        14, "--divergence-window-days", help="Lookback window (trading bars) for RSI divergence detection."
    ),
    divergence_min_extension_days: int = typer.Option(
        5,
        "--divergence-min-extension-days",
        help="Minimum days in the window where extension percentile is elevated/depressed to qualify.",
    ),
    divergence_min_extension_percentile: float | None = typer.Option(
        None,
        "--divergence-min-extension-percentile",
        help="High extension percentile threshold for bearish divergence gating (default: tail_high_pct).",
    ),
    divergence_max_extension_percentile: float | None = typer.Option(
        None,
        "--divergence-max-extension-percentile",
        help="Low extension percentile threshold for bullish divergence gating (default: tail_low_pct).",
    ),
    divergence_min_price_delta_pct: float = typer.Option(
        0.0,
        "--divergence-min-price-delta-pct",
        help="Minimum % move between swing points (in the divergence direction).",
    ),
    divergence_min_rsi_delta: float = typer.Option(
        0.0,
        "--divergence-min-rsi-delta",
        help="Minimum RSI difference between swing points.",
    ),
    rsi_overbought: float = typer.Option(70.0, "--rsi-overbought", help="RSI threshold for overbought tagging."),
    rsi_oversold: float = typer.Option(30.0, "--rsi-oversold", help="RSI threshold for oversold tagging."),
    require_rsi_extreme: bool = typer.Option(
        False,
        "--require-rsi-extreme/--allow-rsi-neutral",
        help="If set, only keep bearish divergences at overbought RSI and bullish divergences at oversold RSI.",
    ),
) -> None:
    """Compute extension percentile stats (tail events + rolling windows) from cached candles."""
    console = Console(width=200)
    cfg = load_technical_backtesting_config(config_path)
    _setup_technicals_logging(cfg)

    df = _load_ohlc_df(ohlc_path=ohlc_path, symbol=symbol, cache_dir=cache_dir)
    if df.empty:
        raise typer.BadParameter("No OHLC data found for extension stats.")

    features = compute_features(df, cfg)
    w = warmup_bars(cfg)
    if w > 0 and len(features) > w:
        features = features.iloc[w:]
    elif w > 0 and len(features) <= w:
        console.print("[yellow]Warning:[/yellow] insufficient history for warmup; using full history.")
    if features.empty:
        raise typer.BadParameter("No features after warmup; check candle history.")

    atr_window = int(cfg["indicators"]["atr"]["window_default"])
    sma_window = int(cfg["indicators"]["sma"]["window_default"])
    ext_col = f"extension_atr_{sma_window}_{atr_window}"
    if ext_col not in features.columns:
        raise typer.BadParameter(f"Missing extension column: {ext_col}")

    ext_cfg = cfg.get("extension_percentiles", {})
    days_per_year = int(ext_cfg.get("days_per_year", 252))

    # Tail thresholds:
    # - Used to select tail events
    # - Used as default extension gating for RSI divergence (unless explicitly overridden)
    tail_high_cfg = float(ext_cfg.get("tail_high_pct", 97.5))
    tail_low_cfg = float(ext_cfg.get("tail_low_pct", 2.5))
    if tail_pct is None:
        tail_high_pct = tail_high_cfg
        tail_low_pct = tail_low_cfg
    else:
        tp = float(tail_pct)
        if tp < 0.0 or tp >= 50.0:
            raise typer.BadParameter("--tail-pct must be >= 0 and < 50")
        tail_low_pct = tp
        tail_high_pct = 100.0 - tp

    if tail_low_pct >= tail_high_pct:
        raise typer.BadParameter("Tail thresholds must satisfy low < high")

    # Rolling window selection for extension percentiles:
    # - If the ticker has <5 years of history, use a 1-year rolling window.
    # - Otherwise, use a 3-year rolling window.
    # Rationale: if window bars >= history bars, percentiles are only defined at the last bar (min_periods=window),
    # which yields very few tail events and weak divergence gating.
    available_bars = int(features[ext_col].dropna().shape[0])
    if percentile_window_years is None:
        history_years = (float(available_bars) / float(days_per_year)) if days_per_year > 0 else 0.0
        window_years = 1 if history_years < 5.0 else 3
    else:
        window_years = int(percentile_window_years)

    if window_years <= 0:
        raise typer.BadParameter("--percentile-window-years must be >= 1")

    windows_years = [window_years]

    forward_days_base = [int(d) for d in (ext_cfg.get("forward_days", [1, 3, 5, 10]) or [])]
    forward_days_daily = [
        int(d)
        for d in (
            ext_cfg.get("forward_days_daily", None)
            or sorted({*forward_days_base, 15})  # include +15D (2 trading weeks) by default
        )
    ]
    forward_days_weekly = [
        int(d) for d in (ext_cfg.get("forward_days_weekly", None) or forward_days_base or [1, 3, 5, 10])
    ]

    report_daily = compute_extension_percentiles(
        extension_series=features[ext_col],
        close_series=features["Close"],
        windows_years=windows_years,
        days_per_year=days_per_year,
        tail_high_pct=float(tail_high_pct),
        tail_low_pct=float(tail_low_pct),
        forward_days=forward_days_daily,
        include_tail_events=True,
    )
    weekly_rule = cfg["weekly_regime"].get("resample_rule", "W-FRI")
    weekly_ext, weekly_close = build_weekly_extension_series(
        df[["Open", "High", "Low", "Close"]],
        sma_window=sma_window,
        atr_window=atr_window,
        resample_rule=weekly_rule,
    )
    report_weekly = compute_extension_percentiles(
        extension_series=weekly_ext,
        close_series=weekly_close,
        windows_years=windows_years,
        days_per_year=int(days_per_year / 5),
        tail_high_pct=float(tail_high_pct),
        tail_low_pct=float(tail_low_pct),
        forward_days=forward_days_weekly,
        include_tail_events=True,
    )

    if report_daily.asof == "-":
        fallback_daily = None
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            try:
                fallback_daily = df.index.max().date().isoformat()
            except Exception:  # noqa: BLE001
                fallback_daily = None
        if fallback_daily:
            report_daily = replace(report_daily, asof=fallback_daily)

    if report_weekly.asof == "-":
        fallback_weekly = None
        if weekly_close is not None and not weekly_close.empty:
            try:
                fallback_weekly = weekly_close.index.max().date().isoformat()
            except Exception:  # noqa: BLE001
                fallback_weekly = None
        if fallback_weekly:
            report_weekly = replace(report_weekly, asof=fallback_weekly)

    sym_label = symbol.upper() if symbol else "UNKNOWN"

    def _none_if_nan(val: object) -> object | None:
        try:
            if val is None:
                return None
            if isinstance(val, (float, int)) and pd.isna(val):
                return None
            if pd.isna(val):
                return None
            return val
        except Exception:  # noqa: BLE001
            return None

    # RSI config (used for event tagging + divergence enrichment).
    rsi_cfg = (cfg.get("indicators", {}) or {}).get("rsi", {}) or {}
    rsi_enabled = bool(rsi_cfg.get("enabled", False))
    rsi_window = int(rsi_cfg.get("window_default", 14)) if rsi_enabled else None
    rsi_col = f"rsi_{rsi_window}" if rsi_window is not None else None

    # Optional enrichment: RSI divergence (daily + weekly, anchored on swing points).
    rsi_divergence_cfg: dict | None = None
    rsi_divergence_daily: dict | None = None
    rsi_divergence_weekly: dict | None = None

    # Pre-align daily series for deterministic iloc-based lookups.
    ext_series_daily = features[ext_col].dropna()
    close_series_daily = features["Close"].reindex(ext_series_daily.index)
    high_series_daily = features["High"].reindex(ext_series_daily.index) if "High" in features.columns else None
    low_series_daily = features["Low"].reindex(ext_series_daily.index) if "Low" in features.columns else None
    rsi_series_daily = (
        features[rsi_col].reindex(ext_series_daily.index) if rsi_col and rsi_col in features.columns else None
    )

    # Weekly candles (for RSI + context alignment).
    weekly_candles = (
        df[["Open", "High", "Low", "Close"]]
        .resample(weekly_rule)
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        .dropna()
    )

    weekly_close_series = weekly_candles["Close"]
    weekly_high_series = weekly_candles["High"]
    weekly_low_series = weekly_candles["Low"]

    weekly_rsi_series = None
    if rsi_window is not None and not weekly_close_series.empty:
        try:
            from ta.momentum import RSIIndicator

            weekly_rsi_series = RSIIndicator(close=weekly_close_series, window=int(rsi_window)).rsi()
        except Exception:  # noqa: BLE001
            weekly_rsi_series = None

    # Divergence config (shared between daily/weekly; window is interpreted as bars for each timeframe).
    min_ext_pct = (
        float(divergence_min_extension_percentile)
        if divergence_min_extension_percentile is not None
        else float(tail_high_pct)
    )
    max_ext_pct = (
        float(divergence_max_extension_percentile)
        if divergence_max_extension_percentile is not None
        else float(tail_low_pct)
    )

    if rsi_window is not None:
        rsi_divergence_cfg = {
            "window_bars": int(divergence_window_days),
            "min_extension_bars": int(divergence_min_extension_days),
            "min_extension_percentile": min_ext_pct,
            "max_extension_percentile": max_ext_pct,
            "min_price_delta_pct": float(divergence_min_price_delta_pct),
            "min_rsi_delta": float(divergence_min_rsi_delta),
            "rsi_overbought": float(rsi_overbought),
            "rsi_oversold": float(rsi_oversold),
            "require_rsi_extreme": bool(require_rsi_extreme),
            "rsi_window": int(rsi_window),
        }

    # Daily RSI divergence.
    try:
        tail_years = report_daily.tail_window_years or (
            max(report_daily.current_percentiles.keys()) if report_daily.current_percentiles else None
        )
        bars = int(tail_years * int(ext_cfg.get("days_per_year", 252))) if tail_years else None
        if rsi_series_daily is not None and bars and bars > 1:
            aligned = pd.concat(
                [
                    ext_series_daily.rename("ext"),
                    close_series_daily.rename("close"),
                    rsi_series_daily.rename("rsi"),
                ],
                axis=1,
            ).dropna()
            if not aligned.empty:
                ext_series = aligned["ext"]
                close_series = aligned["close"]
                rsi_series = aligned["rsi"]

                bars = bars if len(ext_series) >= bars else len(ext_series)
                ext_pct = rolling_percentile_rank(ext_series, bars)

                flags = compute_rsi_divergence_flags(
                    close_series=close_series,
                    rsi_series=rsi_series,
                    extension_percentile_series=ext_pct,
                    window_days=divergence_window_days,
                    min_extension_days=divergence_min_extension_days,
                    min_extension_percentile=min_ext_pct,
                    max_extension_percentile=max_ext_pct,
                    min_price_delta_pct=divergence_min_price_delta_pct,
                    min_rsi_delta=divergence_min_rsi_delta,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    require_rsi_extreme=require_rsi_extreme,
                )

                events = flags[(flags["bearish_divergence"]) | (flags["bullish_divergence"])]
                events_by_date: dict[str, dict] = {}
                for idx, row in events.iterrows():
                    d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                    events_by_date[d] = {
                        "date": d,
                        "divergence": _none_if_nan(row.get("divergence")),
                        "rsi_regime": _none_if_nan(row.get("rsi_regime")),
                        "swing1_date": _none_if_nan(row.get("swing1_date")),
                        "swing2_date": _none_if_nan(row.get("swing2_date")),
                        "close1": _none_if_nan(row.get("close1")),
                        "close2": _none_if_nan(row.get("close2")),
                        "rsi1": _none_if_nan(row.get("rsi1")),
                        "rsi2": _none_if_nan(row.get("rsi2")),
                        "price_delta_pct": _none_if_nan(row.get("price_delta_pct")),
                        "rsi_delta": _none_if_nan(row.get("rsi_delta")),
                    }

                recent = flags.tail(max(1, int(divergence_window_days) + 2))
                last_bearish = None
                last_bullish = None
                for idx, row in reversed(list(recent.iterrows())):
                    if last_bearish is None and bool(row.get("bearish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bearish = events_by_date.get(d)
                    if last_bullish is None and bool(row.get("bullish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bullish = events_by_date.get(d)
                    if last_bearish is not None and last_bullish is not None:
                        break

                rsi_divergence_daily = {
                    "asof": report_daily.asof,
                    "current": {
                        "bearish": last_bearish,
                        "bullish": last_bullish,
                    },
                    "events_by_date": events_by_date,
                }
    except Exception:  # noqa: BLE001
        rsi_divergence_daily = None

    # Weekly RSI divergence (weekly bars).
    try:
        if weekly_rsi_series is not None and not weekly_ext.empty:
            tail_years_w = report_weekly.tail_window_years or (
                max(report_weekly.current_percentiles.keys()) if report_weekly.current_percentiles else None
            )
            bars_w = int(tail_years_w * int(ext_cfg.get("days_per_year", 252) / 5)) if tail_years_w else None
            ext_w = weekly_ext.dropna()
            aligned_w = pd.concat(
                [
                    ext_w.rename("ext"),
                    weekly_close_series.reindex(ext_w.index).rename("close"),
                    weekly_rsi_series.reindex(ext_w.index).rename("rsi"),
                ],
                axis=1,
            ).dropna()
            if not aligned_w.empty and bars_w and bars_w > 1:
                ext_w = aligned_w["ext"]
                close_w = aligned_w["close"]
                rsi_w = aligned_w["rsi"]

                bars_w = bars_w if len(ext_w) >= bars_w else len(ext_w)
                ext_pct_w = rolling_percentile_rank(ext_w, bars_w)

                flags_w = compute_rsi_divergence_flags(
                    close_series=close_w,
                    rsi_series=rsi_w,
                    extension_percentile_series=ext_pct_w,
                    window_days=divergence_window_days,
                    min_extension_days=divergence_min_extension_days,
                    min_extension_percentile=min_ext_pct,
                    max_extension_percentile=max_ext_pct,
                    min_price_delta_pct=divergence_min_price_delta_pct,
                    min_rsi_delta=divergence_min_rsi_delta,
                    rsi_overbought=rsi_overbought,
                    rsi_oversold=rsi_oversold,
                    require_rsi_extreme=require_rsi_extreme,
                )

                events_w = flags_w[(flags_w["bearish_divergence"]) | (flags_w["bullish_divergence"])]
                events_by_date_w: dict[str, dict] = {}
                for idx, row in events_w.iterrows():
                    d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                    events_by_date_w[d] = {
                        "date": d,
                        "divergence": _none_if_nan(row.get("divergence")),
                        "rsi_regime": _none_if_nan(row.get("rsi_regime")),
                        "swing1_date": _none_if_nan(row.get("swing1_date")),
                        "swing2_date": _none_if_nan(row.get("swing2_date")),
                        "close1": _none_if_nan(row.get("close1")),
                        "close2": _none_if_nan(row.get("close2")),
                        "rsi1": _none_if_nan(row.get("rsi1")),
                        "rsi2": _none_if_nan(row.get("rsi2")),
                        "price_delta_pct": _none_if_nan(row.get("price_delta_pct")),
                        "rsi_delta": _none_if_nan(row.get("rsi_delta")),
                    }

                recent_w = flags_w.tail(max(1, int(divergence_window_days) + 2))
                last_bearish_w = None
                last_bullish_w = None
                for idx, row in reversed(list(recent_w.iterrows())):
                    if last_bearish_w is None and bool(row.get("bearish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bearish_w = events_by_date_w.get(d)
                    if last_bullish_w is None and bool(row.get("bullish_divergence")):
                        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
                        last_bullish_w = events_by_date_w.get(d)
                    if last_bearish_w is not None and last_bullish_w is not None:
                        break

                rsi_divergence_weekly = {
                    "asof": report_weekly.asof,
                    "current": {
                        "bearish": last_bearish_w,
                        "bullish": last_bullish_w,
                    },
                    "events_by_date": events_by_date_w,
                }
    except Exception:  # noqa: BLE001
        rsi_divergence_weekly = None

    ext_cfg_effective = dict(ext_cfg or {})
    ext_cfg_effective["windows_years"] = windows_years
    ext_cfg_effective["tail_high_pct"] = float(tail_high_pct)
    ext_cfg_effective["tail_low_pct"] = float(tail_low_pct)
    ext_cfg_effective["forward_days_daily"] = forward_days_daily
    ext_cfg_effective["forward_days_weekly"] = forward_days_weekly

    max_return_horizons_days = {"1w": 5, "4w": 20, "3m": 63, "6m": 126, "9m": 189, "1y": 252}

    payload = {
        "schema_version": 5,
        "symbol": sym_label,
        "asof": report_daily.asof,
        "config": {
            "extension_percentiles": ext_cfg_effective,
            "atr_window": atr_window,
            "sma_window": sma_window,
            "extension_column": ext_col,
            "rsi_divergence": rsi_divergence_cfg,
            "max_forward_returns": {
                "method": "directional_mfe",  # low-tail uses High (up move), high-tail uses Low (down move)
                "horizons_days": max_return_horizons_days,
            },
        },
        "report_daily": asdict(report_daily),
        "report_weekly": asdict(report_weekly),
        "rsi_divergence_daily": rsi_divergence_daily,
        "rsi_divergence_weekly": rsi_divergence_weekly,
    }

    # Enrich tail events (daily + weekly) with:
    # - RSI-at-event regime
    # - max-upside forward returns (High-based, MFE-style)
    # - divergence details (if available)
    # - weekly context attached to daily tail events

    daily = payload.get("report_daily", {}) or {}
    daily_tail_events = daily.get("tail_events", []) or []

    weekly = payload.get("report_weekly", {}) or {}
    weekly_tail_events = weekly.get("tail_events", []) or []

    # Deterministic iloc lookup: date string -> position in the aligned daily extension series.
    daily_date_to_iloc: dict[str, int] = {}
    for i, idx in enumerate(ext_series_daily.index):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        daily_date_to_iloc[d] = i

    # Weekly extension percentile series (for daily context) using the same window as the weekly report.
    weekly_ext_pct = None
    try:
        tail_years_w = report_weekly.tail_window_years or (
            max(report_weekly.current_percentiles.keys()) if report_weekly.current_percentiles else None
        )
        bars_w = int(tail_years_w * int(ext_cfg.get("days_per_year", 252) / 5)) if tail_years_w else None
        ext_w = weekly_ext.dropna()
        if bars_w and bars_w > 1 and not ext_w.empty:
            bars_w = bars_w if len(ext_w) >= bars_w else len(ext_w)
            if bars_w > 1:
                weekly_ext_pct = rolling_percentile_rank(ext_w, bars_w)
    except Exception:  # noqa: BLE001
        weekly_ext_pct = None

    weekly_pct_on_daily = (
        weekly_ext_pct.reindex(ext_series_daily.index, method="ffill") if weekly_ext_pct is not None else None
    )

    weekly_rsi_regime_on_daily = None
    if weekly_rsi_series is not None:
        try:
            weekly_rsi_regime = weekly_rsi_series.dropna().apply(
                lambda v: rsi_regime_tag(
                    rsi_value=float(v), rsi_overbought=float(rsi_overbought), rsi_oversold=float(rsi_oversold)
                )
            )
            weekly_rsi_regime_on_daily = weekly_rsi_regime.reindex(ext_series_daily.index, method="ffill")
        except Exception:  # noqa: BLE001
            weekly_rsi_regime_on_daily = None

    weekly_div_type_on_daily = None
    weekly_div_rsi_on_daily = None
    by_weekly_date = (
        (rsi_divergence_weekly or {}).get("events_by_date", {}) if isinstance(rsi_divergence_weekly, dict) else {}
    )
    if isinstance(by_weekly_date, dict) and by_weekly_date:
        try:
            s_div = pd.Series(index=weekly_close_series.index, dtype="object")
            s_tag = pd.Series(index=weekly_close_series.index, dtype="object")
            for d, ev in by_weekly_date.items():
                try:
                    ts = pd.Timestamp(d)
                except Exception:  # noqa: BLE001
                    continue
                if ts in s_div.index:
                    s_div.loc[ts] = (ev or {}).get("divergence")
                    s_tag.loc[ts] = (ev or {}).get("rsi_regime")
            weekly_div_type_on_daily = s_div.reindex(ext_series_daily.index, method="ffill")
            weekly_div_rsi_on_daily = s_tag.reindex(ext_series_daily.index, method="ffill")
        except Exception:  # noqa: BLE001
            weekly_div_type_on_daily = None
            weekly_div_rsi_on_daily = None

    by_daily_date = (
        (rsi_divergence_daily or {}).get("events_by_date", {}) if isinstance(rsi_divergence_daily, dict) else {}
    )

    # Daily tail events enrichment.
    for ev in daily_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_daily_date.get(d) if isinstance(by_daily_date, dict) else None

        i = daily_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = None
        if i is not None and rsi_series_daily is not None:
            rsi_val = _none_if_nan(rsi_series_daily.iloc[i])
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = None
        if rsi_val is not None:
            try:
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_val),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            except Exception:  # noqa: BLE001
                ev["rsi_regime"] = None

        # Max favorable move + drawdown (directional):
        # - low tail: favorable is bounce using High (>= 0), drawdown is pullback using Low (<= 0)
        # - high tail: favorable is pullback using Low (<= 0), drawdown is squeeze using High (<= 0)
        max_up_short: dict[int, float | None] = {int(h): None for h in forward_days_daily}
        max_down_short: dict[int, float | None] = {int(h): None for h in forward_days_daily}
        max_up_long: dict[str, float | None] = {k: None for k in max_return_horizons_days.keys()}
        max_down_long: dict[str, float | None] = {k: None for k in max_return_horizons_days.keys()}

        if i is not None:
            if high_series_daily is not None:
                for h in forward_days_daily:
                    r = forward_max_up_move(
                        close_series=close_series_daily,
                        high_series=high_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_up_short[int(h)] = None if r is None else float(r)
                for label, h in max_return_horizons_days.items():
                    r = forward_max_up_move(
                        close_series=close_series_daily,
                        high_series=high_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_up_long[str(label)] = None if r is None else float(r)

            if low_series_daily is not None:
                for h in forward_days_daily:
                    r = forward_max_down_move(
                        close_series=close_series_daily,
                        low_series=low_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_down_short[int(h)] = None if r is None else float(r)
                for label, h in max_return_horizons_days.items():
                    r = forward_max_down_move(
                        close_series=close_series_daily,
                        low_series=low_series_daily,
                        start_iloc=i,
                        horizon_bars=int(h),
                    )
                    max_down_long[str(label)] = None if r is None else float(r)

        def _neg0_to_0(val: float) -> float:
            return 0.0 if float(val) == 0.0 else float(val)

        direction = ev.get("direction")
        if direction == "low":
            max_fav_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            max_fav_long = {k: _none_if_nan(v) for k, v in max_up_long.items()}
            dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            dd_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
        elif direction == "high":
            max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            max_fav_long = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_long.items()}
            # Adverse move (drawdown) for high-tail mean reversion is a further squeeze up.
            dd_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            dd_long = {k: _none_if_nan(v) for k, v in max_up_long.items()}
        else:
            max_fav_short = {int(h): None for h in forward_days_daily}
            max_fav_long = {k: None for k in max_return_horizons_days.keys()}
            dd_short = {int(h): None for h in forward_days_daily}
            dd_long = {k: None for k in max_return_horizons_days.keys()}

        # Retain up/down maps for debugging, but prefer *favorable* maps in displays/summaries.
        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["max_up_returns"] = max_up_long
        ev["max_down_returns"] = max_down_long
        ev["max_fav_returns"] = max_fav_long
        ev["forward_drawdown_returns"] = dd_short
        ev["drawdown_returns"] = dd_long

        # Weekly context (ffill weekly values onto daily dates).
        wctx: dict[str, object] = {
            "extension_percentile": None,
            "rsi_regime": None,
            "divergence": None,
            "divergence_rsi_regime": None,
        }
        if i is not None:
            if weekly_pct_on_daily is not None:
                wctx["extension_percentile"] = _none_if_nan(weekly_pct_on_daily.iloc[i])
            if weekly_rsi_regime_on_daily is not None:
                wctx["rsi_regime"] = _none_if_nan(weekly_rsi_regime_on_daily.iloc[i])
            if weekly_div_type_on_daily is not None:
                wctx["divergence"] = _none_if_nan(weekly_div_type_on_daily.iloc[i])
            if weekly_div_rsi_on_daily is not None:
                wctx["divergence_rsi_regime"] = _none_if_nan(weekly_div_rsi_on_daily.iloc[i])
        ev["weekly_context"] = wctx

    # Weekly tail events enrichment (RSI-at-event + divergence + max-up returns).
    weekly_date_to_iloc: dict[str, int] = {}
    for i, idx in enumerate(weekly_close_series.index):
        d = idx.date().isoformat() if isinstance(idx, pd.Timestamp) else str(idx)
        weekly_date_to_iloc[d] = i

    by_weekly_date = (
        (rsi_divergence_weekly or {}).get("events_by_date", {}) if isinstance(rsi_divergence_weekly, dict) else {}
    )
    for ev in weekly_tail_events:
        d = ev.get("date")
        ev["rsi_divergence"] = by_weekly_date.get(d) if isinstance(by_weekly_date, dict) else None

        i = weekly_date_to_iloc.get(d) if isinstance(d, str) else None
        rsi_val = None
        if i is not None and weekly_rsi_series is not None:
            rsi_val = _none_if_nan(weekly_rsi_series.iloc[i])
        ev["rsi"] = rsi_val
        ev["rsi_regime"] = None
        if rsi_val is not None:
            try:
                ev["rsi_regime"] = rsi_regime_tag(
                    rsi_value=float(rsi_val),
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
            except Exception:  # noqa: BLE001
                ev["rsi_regime"] = None

        max_up_short: dict[int, float | None] = {int(h): None for h in forward_days_weekly}
        max_down_short: dict[int, float | None] = {int(h): None for h in forward_days_weekly}
        if i is not None:
            for h in forward_days_weekly:
                r_up = forward_max_up_move(
                    close_series=weekly_close_series,
                    high_series=weekly_high_series,
                    start_iloc=i,
                    horizon_bars=int(h),
                )
                r_dn = forward_max_down_move(
                    close_series=weekly_close_series,
                    low_series=weekly_low_series,
                    start_iloc=i,
                    horizon_bars=int(h),
                )
                max_up_short[int(h)] = None if r_up is None else float(r_up)
                max_down_short[int(h)] = None if r_dn is None else float(r_dn)

        def _neg0_to_0(val: float) -> float:
            return 0.0 if float(val) == 0.0 else float(val)

        direction = ev.get("direction")
        if direction == "low":
            max_fav_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
            dd_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
        elif direction == "high":
            max_fav_short = {k: (None if v is None else _neg0_to_0(-float(v))) for k, v in max_down_short.items()}
            dd_short = {k: _none_if_nan(v) for k, v in max_up_short.items()}
        else:
            max_fav_short = {int(h): None for h in forward_days_weekly}
            dd_short = {int(h): None for h in forward_days_weekly}

        ev["forward_max_up_returns"] = max_up_short
        ev["forward_max_down_returns"] = max_down_short
        ev["forward_max_fav_returns"] = max_fav_short
        ev["forward_drawdown_returns"] = dd_short

    # Store max favorable-move summaries (JSON) for both tails:
    # - low tail: bounce (max-up move using High)
    # - high tail: pullback (max-down move using Low)
    max_move_summary_daily: dict[str, object] = {"horizons_days": max_return_horizons_days, "buckets": []}
    try:
        def _quantile(values: list[float], q: float) -> float | None:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return None
            return float(np.percentile(vals, q * 100.0))

        buckets = [
            ("low_tail_all", "Low tail (all)", lambda ev: ev.get("direction") == "low"),
            (
                "low_tail_rsi_oversold",
                "Low tail + RSI oversold (event)",
                lambda ev: ev.get("direction") == "low" and ev.get("rsi_regime") == "oversold",
            ),
            (
                "low_tail_bull_div",
                "Low tail + bullish divergence",
                lambda ev: ev.get("direction") == "low"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish",
            ),
            (
                "low_tail_bull_div_rsi_oversold",
                "Low tail + bullish divergence + RSI oversold (event)",
                lambda ev: ev.get("direction") == "low"
                and ev.get("rsi_regime") == "oversold"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bullish",
            ),
            ("high_tail_all", "High tail (all)", lambda ev: ev.get("direction") == "high"),
            (
                "high_tail_rsi_overbought",
                "High tail + RSI overbought (event)",
                lambda ev: ev.get("direction") == "high" and ev.get("rsi_regime") == "overbought",
            ),
            (
                "high_tail_bear_div",
                "High tail + bearish divergence",
                lambda ev: ev.get("direction") == "high"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish",
            ),
            (
                "high_tail_bear_div_rsi_overbought",
                "High tail + bearish divergence + RSI overbought (event)",
                lambda ev: ev.get("direction") == "high"
                and ev.get("rsi_regime") == "overbought"
                and isinstance(ev.get("rsi_divergence"), dict)
                and (ev.get("rsi_divergence") or {}).get("divergence") == "bearish",
            ),
        ]
        for key, label, fn in buckets:
            rows = [ev for ev in daily_tail_events if fn(ev)]
            bucket_out: dict[str, object] = {"key": key, "label": label, "n": len(rows), "stats": {}}
            for h_label in max_return_horizons_days.keys():
                is_high_tail_bucket = str(key).startswith("high_tail")
                fav_sign = -1.0 if is_high_tail_bucket else 1.0
                dd_sign = 1.0 if is_high_tail_bucket else -1.0

                fav_mags = []
                dd_mags = []
                for ev in rows:
                    v = (ev.get("max_fav_returns") or {}).get(h_label)
                    if v is not None and not pd.isna(v):
                        fav_mags.append(abs(float(v)) * 100.0)
                    dd = (ev.get("drawdown_returns") or {}).get(h_label)
                    if dd is not None and not pd.isna(dd):
                        dd_mags.append(abs(float(dd)) * 100.0)

                fav_med = _quantile(fav_mags, 0.50)
                fav_p75 = _quantile(fav_mags, 0.75)
                dd_med = _quantile(dd_mags, 0.50)
                dd_p75 = _quantile(dd_mags, 0.75)

                bucket_out["stats"][h_label] = {
                    "fav_median": None if fav_med is None else float(fav_sign) * float(fav_med),
                    "fav_p75": None if fav_p75 is None else float(fav_sign) * float(fav_p75),
                    "dd_median": None if dd_med is None else float(dd_sign) * float(dd_med),
                    "dd_p75": None if dd_p75 is None else float(dd_sign) * float(dd_p75),
                }
            max_move_summary_daily["buckets"].append(bucket_out)
    except Exception:  # noqa: BLE001
        max_move_summary_daily = {"horizons_days": max_return_horizons_days, "buckets": []}

    payload["max_move_summary_daily"] = max_move_summary_daily
    # Backward-compatible alias (schema v3 called this "max_upside_summary_daily").
    payload["max_upside_summary_daily"] = max_move_summary_daily

    # Divergence-conditioned summaries (daily): tail events with divergence vs without, using max-up returns.
    if rsi_divergence_daily is not None:
        # Conditional summaries: tail events with divergence vs without.
        def _median(values: list[float]) -> float | None:
            vals = [float(v) for v in values if v is not None]
            if not vals:
                return None
            vals.sort()
            m = len(vals) // 2
            if len(vals) % 2:
                return float(vals[m])
            return float((vals[m - 1] + vals[m]) / 2.0)

        def _get_forward(dct: dict, day: int) -> object | None:
            if not dct:
                return None
            if day in dct:
                return dct.get(day)
            return dct.get(str(day))

        fwd_days_int = [int(d) for d in forward_days_daily]
        summary: dict[str, dict[str, dict]] = {"high": {}, "low": {}}
        for tail in ("high", "low"):
            want = "bearish" if tail == "high" else "bullish"

            def _bucket(has_div: bool) -> dict[str, object]:
                rows = []
                for ev in daily_tail_events:
                    if ev.get("direction") != tail:
                        continue
                    div = ev.get("rsi_divergence") or None
                    match = bool(div) and div.get("divergence") == want
                    if match != has_div:
                        continue
                    rows.append(ev)

                out: dict[str, object] = {"n": len(rows)}
                for d in fwd_days_int:
                    maxrets = []
                    pcts = []
                    for ev in rows:
                        r = _get_forward(ev.get("forward_max_fav_returns") or {}, d)
                        p = _get_forward(ev.get("forward_extension_percentiles") or {}, d)
                        if r is not None and not pd.isna(r):
                            maxrets.append(float(r) * 100.0)
                        if p is not None and not pd.isna(p):
                            pcts.append(float(p))
                    out[f"median_fwd_max_fav_move_pct_{d}d"] = _median(maxrets)
                    out[f"median_fwd_extension_percentile_{d}d"] = _median(pcts)
                return out

            summary[tail]["with_divergence"] = _bucket(True)
            summary[tail]["without_divergence"] = _bucket(False)

        rsi_divergence_daily["tail_event_summary"] = summary

    md_lines: list[str] = [
        f"# {sym_label} — Extension Percentile Stats",
        "",
        f"- As-of (daily): `{report_daily.asof}`",
        f"- Extension (daily, ATR units): `{'-' if report_daily.extension_atr is None else f'{report_daily.extension_atr:+.2f}'}`",
        "",
        "## Current Percentiles",
    ]
    if report_daily.current_percentiles:
        for years, pct in sorted(report_daily.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Daily p5 / p50 / p95)")
    if report_daily.quantiles_by_window:
        for years, q in sorted(report_daily.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Daily)")
    if rsi_divergence_daily is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_daily.get("current") or {}) if isinstance(rsi_divergence_daily, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
        if cur_bear is None and cur_bull is None:
            md_lines.append("- No divergences detected in the most recent window.")
        if cur_bear is not None:
            try:
                drsi = "-" if cur_bear.get("rsi_delta") is None else f"{float(cur_bear.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = "-" if cur_bear.get("price_delta_pct") is None else f"{float(cur_bear.get('price_delta_pct')):+.2f}%"
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
                f"(RSI tag `{cur_bear.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )
        if cur_bull is not None:
            try:
                drsi = "-" if cur_bull.get("rsi_delta") is None else f"{float(cur_bull.get('rsi_delta')):+.2f}"
            except Exception:  # noqa: BLE001
                drsi = "-"
            try:
                dpct = "-" if cur_bull.get("price_delta_pct") is None else f"{float(cur_bull.get('price_delta_pct')):+.2f}%"
            except Exception:  # noqa: BLE001
                dpct = "-"
            md_lines.append(
                f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
                f"(RSI tag `{cur_bull.get('rsi_regime')}`, ΔRSI `{drsi}`, ΔClose% `{dpct}`)"
            )

        # Compact conditional summary table (focus on the most commonly used horizons).
        summ = rsi_divergence_daily.get("tail_event_summary") if isinstance(rsi_divergence_daily, dict) else None
        if isinstance(summ, dict):
            md_lines.append("")
            md_lines.append("### Tail Outcomes With vs Without Divergence (Daily)")
            md_lines.append("| Tail | Divergence | N | Med max move (5d/15d) | Med fwd pctl (5d/15d) |")
            md_lines.append("|---|---|---:|---|---|")
            for tail, want in (("high", "bearish"), ("low", "bullish")):
                for bucket_name, label in (("with_divergence", f"with {want}"), ("without_divergence", f"without {want}")):
                    b = (summ.get(tail) or {}).get(bucket_name) if isinstance(summ.get(tail), dict) else None
                    if not isinstance(b, dict):
                        continue
                    n = b.get("n", 0)
                    r5 = b.get("median_fwd_max_fav_move_pct_5d")
                    r15 = b.get("median_fwd_max_fav_move_pct_15d")
                    p5 = b.get("median_fwd_extension_percentile_5d")
                    p15 = b.get("median_fwd_extension_percentile_15d")
                    def _fmt_ret(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):+.1f}%"
                        except Exception:  # noqa: BLE001
                            return "-"

                    def _fmt_pct(v: object) -> str:
                        try:
                            return "-" if v is None else f"{float(v):.1f}"
                        except Exception:  # noqa: BLE001
                            return "-"

                    r_str = f"{_fmt_ret(r5)} / {_fmt_ret(r15)}"
                    p_str = f"{_fmt_pct(p5)} / {_fmt_pct(p15)}"
                    md_lines.append(f"| {tail} | {label} | {n} | {r_str} | {p_str} |")

    md_lines.append("")
    md_lines.append("## Max Favorable Move (Daily)")
    md_lines.append(
        "Directional metrics: fav is in the mean-reversion direction; dd is max adverse move against it (both use High/Low vs entry Close)."
    )
    md_lines.append("Cells: fav (med/p75); dd (med/p75).")
    md_lines.append("Descriptive (not financial advice).")
    max_up = payload.get("max_move_summary_daily", {}) if isinstance(payload, dict) else {}
    buckets = max_up.get("buckets", []) if isinstance(max_up, dict) else []
    if buckets:
        md_lines.append("| Bucket | N | 1w | 4w | 3m | 6m | 9m | 1y |")
        md_lines.append("|---|---:|---|---|---|---|---|---|")

        def _fmt_pair(med: object, p75: object) -> str:
            try:
                if med is None or p75 is None:
                    return "-"
                return f"{float(med):+.1f}% / {float(p75):+.1f}%"
            except Exception:  # noqa: BLE001
                return "-"

        def _fmt_cell(fav_med: object, fav_p75: object, dd_med: object, dd_p75: object) -> str:
            fav = _fmt_pair(fav_med, fav_p75)
            dd = _fmt_pair(dd_med, dd_p75)
            if fav == "-" and dd == "-":
                return "-"
            if dd == "-":
                return fav
            if fav == "-":
                return dd
            return f"{fav}; {dd}"

        for b in buckets:
            if not isinstance(b, dict):
                continue
            n = int(b.get("n", 0) or 0)
            stats = b.get("stats", {}) if isinstance(b.get("stats"), dict) else {}
            def _get(label: str) -> str:
                s = stats.get(label, {}) if isinstance(stats.get(label), dict) else {}
                return _fmt_cell(s.get("fav_median"), s.get("fav_p75"), s.get("dd_median"), s.get("dd_p75"))

            md_lines.append(
                f"| {b.get('label', '-')} | {n} | {_get('1w')} | {_get('4w')} | {_get('3m')} | {_get('6m')} | {_get('9m')} | {_get('1y')} |"
            )
    else:
        md_lines.append("- Not available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Tail Events (Daily, all)")
    if daily_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_daily)
        max_horiz = "/".join(max_return_horizons_days.keys())
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | W pctl | W RSI | W div | Fwd pctl ({horiz}) | Max ret% ({max_horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---:|---|---|---|---|")
        for ev in daily_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_daily]
            maxrets = [(ev.get("max_fav_returns") or {}).get(str(lbl)) for lbl in max_return_horizons_days.keys()]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            wctx = ev.get("weekly_context") if isinstance(ev.get("weekly_context"), dict) else {}
            w_pct = wctx.get("extension_percentile")
            w_pct_str = "-" if w_pct is None else f"{float(w_pct):.1f}"
            w_rsi = wctx.get("rsi_regime") or "-"
            w_div = wctx.get("divergence") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {w_pct_str} | {w_rsi} | {w_div} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    md_lines.append("")
    md_lines.append("## Weekly Context")
    md_lines.append(f"- As-of (weekly): `{report_weekly.asof}`")
    md_lines.append(
        f"- Extension (weekly, ATR units): `{'-' if report_weekly.extension_atr is None else f'{report_weekly.extension_atr:+.2f}'}`"
    )
    weekly_rsi_val = None
    weekly_rsi_tag = None
    if weekly_rsi_series is not None:
        try:
            v = weekly_rsi_series.dropna()
            if not v.empty:
                weekly_rsi_val = float(v.iloc[-1])
                weekly_rsi_tag = rsi_regime_tag(
                    rsi_value=weekly_rsi_val,
                    rsi_overbought=float(rsi_overbought),
                    rsi_oversold=float(rsi_oversold),
                )
        except Exception:  # noqa: BLE001
            weekly_rsi_val = None
            weekly_rsi_tag = None
    if weekly_rsi_val is not None and weekly_rsi_tag is not None:
        md_lines.append(f"- RSI (weekly): `{weekly_rsi_val:.1f}` (tag `{weekly_rsi_tag}`)")

    md_lines.append("")
    md_lines.append("## Current Percentiles (Weekly)")
    if report_weekly.current_percentiles:
        for years, pct in sorted(report_weekly.current_percentiles.items()):
            md_lines.append(f"- {years}y: `{pct:.1f}`")
    else:
        md_lines.append("- No percentile windows available (insufficient history).")

    md_lines.append("")
    md_lines.append("## Rolling Quantiles (Weekly p5 / p50 / p95)")
    if report_weekly.quantiles_by_window:
        for years, q in sorted(report_weekly.quantiles_by_window.items()):
            if q.p5 is None or q.p50 is None or q.p95 is None:
                continue
            md_lines.append(f"- {years}y: `{q.p5:+.2f} / {q.p50:+.2f} / {q.p95:+.2f}`")
    else:
        md_lines.append("- Not available.")

    md_lines.append("")
    md_lines.append("## RSI Divergence (Weekly)")
    if rsi_divergence_weekly is None:
        md_lines.append("- Not available (RSI disabled/missing or insufficient history).")
    else:
        cfg_line = (
            f"- Window (bars): `{rsi_divergence_cfg.get('window_bars')}` "
            f"(min ext bars `{rsi_divergence_cfg.get('min_extension_bars')}`, "
            f"ext pct gates `{rsi_divergence_cfg.get('min_extension_percentile'):.1f}` / `{rsi_divergence_cfg.get('max_extension_percentile'):.1f}`, "
            f"RSI `{rsi_divergence_cfg.get('rsi_overbought'):.0f}`/`{rsi_divergence_cfg.get('rsi_oversold'):.0f}`)"
        )
        md_lines.append(cfg_line)
        cur = (rsi_divergence_weekly.get("current") or {}) if isinstance(rsi_divergence_weekly, dict) else {}
        cur_bear = cur.get("bearish")
        cur_bull = cur.get("bullish")
        if cur_bear is None and cur_bull is None:
            md_lines.append("- No divergences detected in the most recent window.")
        if cur_bear is not None:
            md_lines.append(
                f"- Current bearish divergence: `{cur_bear.get('swing1_date')} → {cur_bear.get('swing2_date')}` "
                f"(RSI tag `{cur_bear.get('rsi_regime')}`)"
            )
        if cur_bull is not None:
            md_lines.append(
                f"- Current bullish divergence: `{cur_bull.get('swing1_date')} → {cur_bull.get('swing2_date')}` "
                f"(RSI tag `{cur_bull.get('rsi_regime')}`)"
            )

    md_lines.append("")
    md_lines.append("## Tail Events (Weekly, all)")
    if weekly_tail_events:
        horiz = "/".join(str(int(d)) for d in forward_days_weekly)
        md_lines.append(
            f"| Date | Tail | Ext | Pctl | RSI | Div | Div RSI | Fwd pctl ({horiz}) | Max move% ({horiz}) |"
        )
        md_lines.append("|---|---|---:|---:|---|---|---|---|---|")
        for ev in weekly_tail_events:
            pcts = [(ev.get("forward_extension_percentiles") or {}).get(int(d)) for d in forward_days_weekly]
            maxrets = [(ev.get("forward_max_fav_returns") or {}).get(int(d)) for d in forward_days_weekly]
            pcts_str = ", ".join("-" if v is None else f"{float(v):.1f}" for v in pcts)
            maxrets_str = ", ".join("-" if v is None else f"{float(v)*100.0:+.1f}%" for v in maxrets)

            div = ev.get("rsi_divergence") if isinstance(ev.get("rsi_divergence"), dict) else None
            div_type = "-" if not div else (div.get("divergence") or "-")
            div_tag = "-" if not div else (div.get("rsi_regime") or "-")

            rsi_tag = ev.get("rsi_regime") or "-"

            md_lines.append(
                f"| {ev.get('date')} | {ev.get('direction')} | {float(ev.get('extension_atr')):+.2f} | {float(ev.get('percentile')):.1f} | {rsi_tag} | {div_type} | {div_tag} | {pcts_str} | {maxrets_str} |"
            )
    else:
        md_lines.append("- No tail events found.")

    md = "\n".join(md_lines).rstrip() + "\n"

    if out:
        base = out / sym_label
        base.mkdir(parents=True, exist_ok=True)
        json_path = base / f"{report_daily.asof}.json"
        md_path = base / f"{report_daily.asof}.md"
        if write_json:
            json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            console.print(f"Wrote JSON: {json_path}")
        if write_md:
            md_path.write_text(md, encoding="utf-8")
            console.print(f"Wrote Markdown: {md_path}")

    if print_to_console:
        try:
            from rich.markdown import Markdown

            console.print(Markdown(md))
        except Exception:  # noqa: BLE001
            console.print(md)


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
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [c for c in needed if c in features.columns]
    features = features.loc[:, [c for c in cols if c in features.columns]]

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
    needed = required_feature_columns_for_strategy(strategy, strat_cfg)
    cols = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        cols.append("Volume")
    cols += [c for c in needed if c in features.columns]
    features = features.loc[:, [c for c in cols if c in features.columns]]

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
                needed = required_feature_columns_for_strategy(strategy, strat_cfg)
                cols = ["Open", "High", "Low", "Close"]
                if "Volume" in features.columns:
                    cols.append("Volume")
                cols += [c for c in needed if c in features.columns]
                strat_features = features.loc[:, [c for c in cols if c in features.columns]]
                StrategyClass = get_strategy(strategy)
                opt_cfg = cfg["optimization"]
                if cfg["walk_forward"]["enabled"]:
                    result = walk_forward_optimize(
                        strat_features,
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
                        strat_features,
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
