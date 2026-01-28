from __future__ import annotations

import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.flow import compute_flow, summarize_flow
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.greeks import black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.data.candles import CandleStore, last_close
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.data.yf_client import DataFetchError, YFinanceClient, contract_row_by_strike
from options_helper.models import OptionType, Portfolio, Position, RiskProfile
from options_helper.reporting import render_positions, render_summary
from options_helper.storage import load_portfolio, save_portfolio, write_template

app = typer.Typer(add_completion=False)


def _parse_date(value: str) -> date:
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise typer.BadParameter("Invalid date format. Use YYYY-MM-DD (recommended).")


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
        0.30,
        "--window-pct",
        min=0.0,
        max=2.0,
        help="Strike window around spot (e.g. 0.30 = +/-30%).",
    ),
    spot_period: str = typer.Option(
        "10d",
        "--spot-period",
        help="Candle period used to estimate spot price from daily candles.",
    ),
) -> None:
    """Save a once-daily options chain snapshot (windowed around spot) for flow analysis."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    store = OptionsSnapshotStore(cache_dir)
    candle_store = CandleStore(candle_cache_dir)
    client = YFinanceClient()

    expiries_by_symbol: dict[str, set[date]] = {}
    for p in portfolio.positions:
        expiries_by_symbol.setdefault(p.symbol, set()).add(p.expiry)

    snapshot_date = date.today()

    console.print(
        f"Snapshotting options chains for {len(expiries_by_symbol)} symbol(s) on {snapshot_date.isoformat()}..."
    )

    for symbol, expiries in sorted(expiries_by_symbol.items()):
        history = candle_store.get_daily_history(symbol, period=spot_period)
        spot = last_close(history)
        if spot is None:
            try:
                spot = client.get_underlying(symbol, period=spot_period, interval="1d").last_price
            except DataFetchError:
                spot = None

        if spot is None or spot <= 0:
            console.print(f"[yellow]Warning:[/yellow] {symbol}: missing spot price; skipping snapshot.")
            continue

        strike_min = spot * (1.0 - window_pct)
        strike_max = spot * (1.0 + window_pct)

        meta = {
            "spot": spot,
            "window_pct": window_pct,
            "strike_min": strike_min,
            "strike_max": strike_max,
            "snapshot_date": snapshot_date.isoformat(),
        }

        for exp in sorted(expiries):
            chain = client.get_options_chain(symbol, exp)
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            calls["optionType"] = "call"
            puts["optionType"] = "put"
            calls["expiry"] = exp.isoformat()
            puts["expiry"] = exp.isoformat()

            df = pd.concat([calls, puts], ignore_index=True)
            if "strike" in df.columns:
                df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

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
            ]
            keep = [c for c in keep if c in df.columns]
            df = df[keep]

            store.save_expiry_snapshot(symbol, snapshot_date, expiry=exp, snapshot=df, meta=meta)
            console.print(f"{symbol} {exp.isoformat()}: saved {len(df)} contracts")


@app.command("flow")
def flow_report(
    portfolio_path: Path = typer.Argument(..., help="Path to portfolio JSON."),
    cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--cache-dir",
        help="Directory for options chain snapshots.",
    ),
    top: int = typer.Option(10, "--top", min=1, max=100, help="Top contracts per symbol to display."),
) -> None:
    """Report day-to-day OI/volume deltas from locally captured snapshots."""
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = OptionsSnapshotStore(cache_dir)
    symbols = sorted({p.symbol for p in portfolio.positions})
    if not symbols:
        console.print("No positions.")
        raise typer.Exit(0)

    pos_keys = {(p.symbol, p.expiry.isoformat(), float(p.strike), p.option_type) for p in portfolio.positions}

    from rich.table import Table

    for sym in symbols:
        dates = store.latest_dates(sym, n=2)
        if len(dates) < 2:
            console.print(f"[yellow]No flow data for {sym}:[/yellow] need at least 2 snapshots.")
            continue

        prev_date, today_date = dates[-2], dates[-1]
        today_df = store.load_day(sym, today_date)
        prev_df = store.load_day(sym, prev_date)
        if today_df.empty or prev_df.empty:
            console.print(f"[yellow]No flow data for {sym}:[/yellow] empty snapshot(s).")
            continue

        flow = compute_flow(today_df, prev_df)
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
