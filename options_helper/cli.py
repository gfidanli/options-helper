from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

if TYPE_CHECKING:
    import pandas as pd

from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.chain_metrics import compute_chain_report, execution_quality
from options_helper.analysis.compare_metrics import compute_compare_report
from options_helper.analysis.confluence import ConfluenceInputs, ConfluenceScore, score_confluence
from options_helper.analysis.derived_metrics import DerivedRow, compute_derived_stats
from options_helper.analysis.events import earnings_event_risk
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.flow import FlowGroupBy, aggregate_flow_window, compute_flow, summarize_flow
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.portfolio_risk import (
    PortfolioExposure,
    compute_portfolio_exposure,
    run_stress,
)
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.analysis.research import (
    Direction,
    OptionCandidate,
    TradeLevels,
    UnderlyingSetup,
    VolatilityContext,
    analyze_underlying,
    build_confluence_inputs,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.analysis.roll_plan import compute_roll_plan
from options_helper.analysis.roll_plan_multileg import compute_roll_plan_multileg
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain, black_scholes_greeks
from options_helper.analysis.indicators import breakout_down, breakout_up, ema, rsi, sma
from options_helper.data.candles import CandleCacheError, close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.derived import DERIVED_COLUMNS, DERIVED_SCHEMA_VERSION
from options_helper.data.earnings import EarningsRecord, safe_next_earnings_date
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.commands.backtest import app as backtest_app
from options_helper.commands.common import _build_stress_scenarios, _parse_date, _spot_from_meta
from options_helper.commands.derived import app as derived_app
from options_helper.commands.events import app as events_app
from options_helper.commands.intraday import app as intraday_app
from options_helper.commands.journal import app as journal_app
from options_helper.commands.position_metrics import _extract_float, _mark_price, _position_metrics
from options_helper.commands.portfolio import register as register_portfolio_commands
from options_helper.commands.scanner import app as scanner_app
from options_helper.commands.stream import app as stream_app
from options_helper.commands.technicals import app as technicals_app, technicals_extension_stats
from options_helper.commands.watchlists import app as watchlists_app
import options_helper.cli_deps as cli_deps
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.data.technical_backtesting_config import (
    ConfigError as TechnicalConfigError,
    load_technical_backtesting_config,
)
from options_helper.data.market_types import DataFetchError
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.models import OptionType, Position, RiskProfile
from options_helper.observability import finalize_run_logger, setup_run_logger
from options_helper.reporting import MultiLegSummary, render_multi_leg_positions, render_positions, render_summary
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
from options_helper.reporting_roll import render_roll_plan_console, render_roll_plan_multileg_console
from options_helper.schemas.briefing import BriefingArtifact
from options_helper.schemas.common import utc_now
from options_helper.schemas.chain_report import ChainReportArtifact
from options_helper.schemas.compare import CompareArtifact
from options_helper.schemas.flow import FlowArtifact
from options_helper.storage import load_portfolio
from options_helper.watchlists import build_default_watchlists, load_watchlists, save_watchlists
from options_helper.ui.dashboard import load_briefing_artifact, render_dashboard, resolve_briefing_paths
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot, compute_technical_snapshot

app = typer.Typer(add_completion=False)
app.add_typer(watchlists_app, name="watchlists")
app.add_typer(derived_app, name="derived")
app.add_typer(technicals_app, name="technicals")
app.add_typer(scanner_app, name="scanner")
app.add_typer(journal_app, name="journal")
app.add_typer(backtest_app, name="backtest")
app.add_typer(intraday_app, name="intraday")
app.add_typer(events_app, name="events")
app.add_typer(stream_app, name="stream")
register_portfolio_commands(app)


@app.callback()
def main(
    ctx: typer.Context,
    log_dir: Path = typer.Option(
        Path("data/logs"),
        "--log-dir",
        help="Directory to write per-command logs.",
    ),
    provider: str = typer.Option(
        "yahoo",
        "--provider",
        help="Market data provider (default: yahoo).",
    ),
) -> None:
    command_name = ctx.info_name or "options-helper"
    if ctx.invoked_subcommand:
        command_name = f"{command_name} {ctx.invoked_subcommand}"
    run_logger = setup_run_logger(log_dir, command_name)
    provider_token = set_default_provider_name(provider)

    def _on_close() -> None:
        reset_default_provider_name(provider_token)
        if run_logger is not None:
            finalize_run_logger(run_logger)

    ctx.call_on_close(_on_close)

    if run_logger is None:
        return


pd: object | None = None


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd




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

    for p in portfolio.positions:
        try:
            chain = provider.get_options_chain(p.symbol, p.expiry)
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
    _ensure_pandas()
    import numpy as np

    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = cli_deps.build_snapshot_store(cache_dir)
    provider = cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    provider_name = getattr(provider, "name", "unknown")
    provider_version = (
        getattr(provider, "version", None)
        or getattr(provider, "provider_version", None)
        or getattr(provider, "__version__", None)
    )

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
                underlying = provider.get_underlying(symbol, period=spot_period, interval="1d")
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
            "provider": provider_name,
        }
        if provider_version:
            meta["provider_version"] = provider_version

        expiries: list[date]
        if not use_watchlists and not want_all_expiries:
            expiries = sorted(expiries_by_symbol.get(symbol, set()))
        else:
            expiries = provider.list_option_expiries(symbol)
            if not expiries:
                console.print(f"[yellow]Warning:[/yellow] {symbol}: no listed option expiries; skipping snapshot.")
                continue
            if effective_max_expiries is not None:
                expiries = expiries[:effective_max_expiries]

        for exp in expiries:
            if want_full_chain:
                try:
                    raw = provider.get_options_chain_raw(symbol, exp)
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
                chain = provider.get_options_chain(symbol, exp)
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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Report OI/volume deltas from locally captured snapshots (single-day or windowed)."""
    _ensure_pandas()
    portfolio = load_portfolio(portfolio_path)
    console = Console()

    store = cli_deps.build_snapshot_store(cache_dir)
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
                payload = FlowArtifact(
                    schema_version=1,
                    generated_at=utc_now(),
                    as_of=today_date.isoformat(),
                    symbol=sym.upper(),
                    from_date=prev_date.isoformat(),
                    to_date=today_date.isoformat(),
                    window=1,
                    group_by="contract",
                    snapshot_dates=[prev_date.isoformat(), today_date.isoformat()],
                    net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                ).to_dict()
                if strict:
                    FlowArtifact.model_validate(payload)
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
            payload = FlowArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=end_date.isoformat(),
                symbol=sym.upper(),
                from_date=start_date.isoformat(),
                to_date=end_date.isoformat(),
                window=window,
                group_by=group_by_norm,
                snapshot_dates=[d.isoformat() for d in dates],
                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
            ).to_dict()
            if strict:
                FlowArtifact.model_validate(payload)
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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
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
    store = cli_deps.build_snapshot_store(cache_dir)

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
        report_artifact = ChainReportArtifact(
            generated_at=utc_now(),
            **report.model_dump(),
        )
        if strict:
            ChainReportArtifact.model_validate(report_artifact.to_dict())

        if fmt == "console":
            render_chain_report_console(console, report)
        elif fmt == "md":
            console.print(render_chain_report_markdown(report))
        else:
            console.print(report_artifact.model_dump_json(indent=2))

        if out is not None:
            base = out / "chains" / report.symbol
            base.mkdir(parents=True, exist_ok=True)
            json_path = base / f"{as_of_date.isoformat()}.json"
            json_path.write_text(report_artifact.model_dump_json(indent=2), encoding="utf-8")

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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
    ),
) -> None:
    """Diff two snapshot dates for a symbol (offline)."""
    console = Console()
    store = cli_deps.build_snapshot_store(cache_dir)

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
            payload = CompareArtifact(
                schema_version=1,
                generated_at=utc_now(),
                as_of=to_date.isoformat(),
                symbol=symbol.upper(),
                from_report=report_from.model_dump(),
                to_report=report_to.model_dump(),
                diff=diff.model_dump(),
            ).to_dict()
            if strict:
                CompareArtifact.model_validate(payload)
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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
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
    _ensure_pandas()
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

    store = cli_deps.build_snapshot_store(cache_dir)
    derived_store = cli_deps.build_derived_store(derived_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)

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
                chain_artifact = ChainReportArtifact(
                    generated_at=utc_now(),
                    **chain_report_model.model_dump(),
                )
                if strict:
                    ChainReportArtifact.model_validate(chain_artifact.to_dict())
                json_path.write_text(chain_artifact.model_dump_json(indent=2), encoding="utf-8")
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
                    payload = CompareArtifact(
                        schema_version=1,
                        generated_at=utc_now(),
                        as_of=to_date.isoformat(),
                        symbol=sym.upper(),
                        from_report=report_from.model_dump(),
                        to_report=report_to.model_dump(),
                        diff=diff.model_dump(),
                    ).to_dict()
                    if strict:
                        CompareArtifact.model_validate(payload)
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
                            payload = FlowArtifact(
                                schema_version=1,
                                generated_at=utc_now(),
                                as_of=to_date.isoformat(),
                                symbol=sym.upper(),
                                from_date=from_date.isoformat(),
                                to_date=to_date.isoformat(),
                                window=1,
                                group_by=group_by,
                                snapshot_dates=[from_date.isoformat(), to_date.isoformat()],
                                net=artifact_net.where(pd.notna(artifact_net), None).to_dict(orient="records"),
                            ).to_dict()
                            if strict:
                                FlowArtifact.model_validate(payload)
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
    strict: bool = typer.Option(
        False,
        "--strict",
        help="Validate JSON artifacts against schemas.",
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
    _ensure_pandas()
    console = Console(width=200)
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    positions_by_symbol: dict[str, list[Position]] = {}
    for p in portfolio.positions:
        positions_by_symbol.setdefault(p.symbol.upper(), []).append(p)

    portfolio_symbols = sorted({p.symbol.upper() for p in portfolio.positions})
    watch_symbols: list[str] = []
    watchlist_symbols_by_name: dict[str, list[str]] = {}
    if watchlist:
        try:
            wl = load_watchlists(watchlists_path)
            for name in watchlist:
                wl_symbols = wl.get(name)
                watch_symbols.extend(wl_symbols)
                watchlist_symbols_by_name[name] = wl_symbols
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Warning:[/yellow] failed to load watchlists: {exc}")

    symbols = sorted(set(portfolio_symbols).union({s.upper() for s in watch_symbols if s}))
    if symbol is not None:
        symbols = [symbol.upper().strip()]

    if symbol is not None:
        sym = symbols[0] if symbols else ""
        symbol_sources_map: dict[str, set[str]] = {}
        if sym:
            if sym in portfolio_symbols:
                symbol_sources_map.setdefault(sym, set()).add("portfolio")
            symbol_sources_map.setdefault(sym, set()).add("manual")
        symbol_sources_payload = [
            {"symbol": sym, "sources": sorted(symbol_sources_map.get(sym, set()))}
            for sym in symbols
        ]
        watchlists_payload: list[dict[str, object]] = []
    else:
        symbol_sources_map = {}
        for sym in portfolio_symbols:
            symbol_sources_map.setdefault(sym, set()).add("portfolio")
        for name, syms in watchlist_symbols_by_name.items():
            for sym in syms:
                symbol_sources_map.setdefault(sym, set()).add(f"watchlist:{name}")

        symbol_sources_payload = [
            {"symbol": sym, "sources": sorted(symbol_sources_map.get(sym, set()))} for sym in symbols
        ]
        watchlists_payload = [
            {"name": name, "symbols": watchlist_symbols_by_name.get(name, [])}
            for name in watchlist
            if name in watchlist_symbols_by_name
        ]

    if not symbols:
        console.print("[red]Error:[/red] no symbols selected (empty portfolio and no watchlists)")
        raise typer.Exit(1)

    store = cli_deps.build_snapshot_store(cache_dir)
    derived_store = cli_deps.build_derived_store(derived_dir)
    candle_store = cli_deps.build_candle_store(candle_cache_dir)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))

    technicals_cfg: dict | None = None
    technicals_cfg_error: str | None = None
    try:
        technicals_cfg = load_technical_backtesting_config(technicals_config)
    except Exception as exc:  # noqa: BLE001
        technicals_cfg_error = str(exc)
    confluence_cfg = None
    confluence_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    if confluence_cfg_error:
        console.print(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")

    # Cache day snapshots for portfolio marks (best-effort).
    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}

    sections: list[BriefingSymbolSection] = []
    resolved_to_dates: list[date] = []
    compare_norm = compare.strip().lower()
    compare_enabled = compare_norm not in {"none", "off", "false", "0"}

    def _trend_from_weekly_flag(flag: bool | None) -> str | None:
        if flag is True:
            return "up"
        if flag is False:
            return "down"
        return None

    def _extension_percentile_from_snapshot(snapshot: TechnicalSnapshot | None) -> float | None:
        if snapshot is None or snapshot.extension_percentiles is None:
            return None
        daily = snapshot.extension_percentiles.daily
        if daily is None or not daily.current_percentiles:
            return None
        items: list[tuple[float, float]] = []
        for key, value in daily.current_percentiles.items():
            try:
                items.append((float(key), float(value)))
            except Exception:  # noqa: BLE001
                continue
        if not items:
            return None
        return sorted(items, key=lambda t: t[0])[-1][1]

    def _net_flow_delta_oi_notional(flow_net: pd.DataFrame | None) -> float | None:
        if flow_net is None or flow_net.empty:
            return None
        if "deltaOI_notional" not in flow_net.columns or "optionType" not in flow_net.columns:
            return None
        df = flow_net.copy()
        df["deltaOI_notional"] = pd.to_numeric(df["deltaOI_notional"], errors="coerce")
        df["optionType"] = df["optionType"].astype(str).str.lower()
        calls = df[df["optionType"] == "call"]["deltaOI_notional"].dropna()
        puts = df[df["optionType"] == "put"]["deltaOI_notional"].dropna()
        if calls.empty and puts.empty:
            return None
        return float(calls.sum()) - float(puts.sum())

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
        confluence_score = None
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

            try:
                trend = _trend_from_weekly_flag(technicals.weekly_trend_up if technicals is not None else None)
                ext_pct = _extension_percentile_from_snapshot(technicals)
                flow_notional = _net_flow_delta_oi_notional(flow_net)
                iv_rv = derived_row.iv_rv_20d if derived_row is not None else None
                inputs = ConfluenceInputs(
                    weekly_trend=trend,
                    extension_percentile=ext_pct,
                    rsi_divergence=None,
                    flow_delta_oi_notional=flow_notional,
                    iv_rv_20d=iv_rv,
                )
                confluence_score = score_confluence(inputs, cfg=confluence_cfg)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"confluence unavailable: {exc}")
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
                confluence=confluence_score,
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
    portfolio_rows_payload: list[dict[str, object]] = []
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
        portfolio_rows_payload.append(
            {
                "id": p.id,
                "symbol": sym,
                "option_type": p.option_type,
                "expiry": p.expiry.isoformat(),
                "strike": float(p.strike),
                "contracts": int(p.contracts),
                "cost_basis": float(p.cost_basis),
                "mark": None if mark is None else float(mark),
                "pnl": None if pnl_abs is None else float(pnl_abs),
                "pnl_pct": None if pnl_pct is None else float(pnl_pct),
                "spr_pct": None if spr_pct is None else float(spr_pct),
                "as_of": None if to_date is None else to_date.isoformat(),
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
            as_of=report_date,
            portfolio_path=str(portfolio_path),
            symbol_sections=sections,
            top=top,
            technicals_config=str(technicals_config),
            portfolio_exposure=portfolio_exposure,
            portfolio_stress=portfolio_stress,
            portfolio_rows=portfolio_rows_payload,
            symbol_sources=symbol_sources_payload,
            watchlists=watchlists_payload,
        )
        if strict:
            BriefingArtifact.model_validate(payload)
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


@app.command("dashboard")
def dashboard(
    report_date: str = typer.Option(
        "latest",
        "--date",
        help="Briefing date (YYYY-MM-DD) or 'latest'.",
    ),
    reports_dir: Path = typer.Option(
        Path("data/reports"),
        "--reports-dir",
        help="Reports root (expects {reports_dir}/daily/{DATE}.json).",
    ),
    scanner_run_dir: Path = typer.Option(
        Path("data/scanner/runs"),
        "--scanner-run-dir",
        help="Scanner runs directory (for shortlist view).",
    ),
    scanner_run_id: str | None = typer.Option(
        None,
        "--scanner-run-id",
        help="Specific scanner run id to display (defaults to latest for the date).",
    ),
    max_shortlist_rows: int = typer.Option(
        20,
        "--max-shortlist-rows",
        min=1,
        max=200,
        help="Max rows to show in the scanner shortlist table.",
    ),
) -> None:
    """Render a read-only daily dashboard from briefing JSON + artifacts."""
    console = Console(width=200)
    try:
        paths = resolve_briefing_paths(reports_dir, report_date)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        artifact = load_briefing_artifact(paths.json_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] failed to load briefing JSON: {exc}")
        raise typer.Exit(1) from exc

    console.print(f"Briefing JSON: {paths.json_path}")
    render_dashboard(
        artifact=artifact,
        console=console,
        reports_dir=reports_dir,
        scanner_run_dir=scanner_run_dir,
        scanner_run_id=scanner_run_id,
        max_shortlist_rows=max_shortlist_rows,
    )


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

    store = cli_deps.build_snapshot_store(cache_dir)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    next_earnings_date = safe_next_earnings_date(earnings_store, position.symbol)

    try:
        as_of_date = store.resolve_date(position.symbol, as_of)
        df = store.load_day(position.symbol, as_of_date)
        meta = store.load_meta(position.symbol, as_of_date)
        spot = _spot_from_meta(meta)
        if spot is None:
            raise ValueError("missing spot price in meta.json (run snapshot-options first)")

        if isinstance(position, MultiLegPosition):
            report = compute_roll_plan_multileg(
                df,
                symbol=position.symbol,
                as_of=as_of_date,
                spot=spot,
                position=position,
                horizon_months=horizon_months,
                min_open_interest=min_oi,
                min_volume=min_vol,
                top=top,
                include_bad_quotes=include_bad_quotes,
                max_debit=max_debit,
                min_credit=min_credit,
            )
            render_roll_plan_multileg_console(console, report)
        else:
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
    store = cli_deps.build_earnings_store(cache_dir)
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
            try:
                ev = cli_deps.build_provider().get_next_earnings_event(sym)
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

    store = cli_deps.build_earnings_store(cache_dir)
    provider = cli_deps.build_provider()

    ok = 0
    err = 0
    unknown = 0

    console.print(f"Refreshing earnings for {len(symbols)} symbol(s)...")
    for sym in sorted(symbols):
        try:
            ev = provider.get_next_earnings_event(sym)
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
    _ensure_pandas()
    portfolio = load_portfolio(portfolio_path)
    rp = portfolio.risk_profile

    if symbol:
        symbols = [symbol.strip().upper()]
    else:
        wl = load_watchlists(watchlists_path)
        symbols = wl.get(watchlist)
        if not symbols:
            raise typer.BadParameter(f"Watchlist '{watchlist}' is empty or missing in {watchlists_path}")

    provider = cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    derived_store = cli_deps.build_derived_store(derived_dir)
    confluence_cfg = None
    confluence_cfg_error = None
    technicals_cfg = None
    technicals_cfg_error = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    try:
        technicals_cfg = load_technical_backtesting_config()
    except TechnicalConfigError as exc:
        technicals_cfg_error = str(exc)
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

    def _format_confluence(score: ConfluenceScore) -> tuple[str, str, str]:
        total = f"{score.total:.0f}"
        coverage = f"{score.coverage * 100.0:.0f}%"
        parts = []
        for comp in score.components:
            if comp.score is None:
                continue
            if abs(comp.score) < 1e-9:
                continue
            parts.append(f"{comp.name} {comp.score:+.0f}")
        detail = ", ".join(parts) if parts else "neutral"
        return total, coverage, detail

    if confluence_cfg_error:
        emit(f"[yellow]Warning:[/yellow] confluence config unavailable: {confluence_cfg_error}")
    if technicals_cfg_error:
        emit(f"[yellow]Warning:[/yellow] technicals config unavailable: {technicals_cfg_error}")

    cached_history: dict[str, pd.DataFrame] = {}
    cached_setup: dict[str, UnderlyingSetup] = {}
    cached_extension_pct: dict[str, float | None] = {}
    pre_confluence: dict[str, ConfluenceScore] = {}

    for sym in symbols:
        history = candle_store.get_daily_history(sym, period=period)
        cached_history[sym] = history
        setup = analyze_underlying(sym, history=history, risk_profile=rp)
        cached_setup[sym] = setup

        ext_pct = None
        if technicals_cfg is not None and history is not None and not history.empty:
            try:
                ext_result = compute_current_extension_percentile(history, technicals_cfg)
                ext_pct = ext_result.percentile
            except Exception:  # noqa: BLE001
                ext_pct = None
        cached_extension_pct[sym] = ext_pct

        inputs = build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
        pre_confluence[sym] = score_confluence(inputs, cfg=confluence_cfg)

    if len(symbols) > 1:
        def _sort_key(sym: str) -> tuple[float, float, str]:
            score = pre_confluence.get(sym)
            coverage = score.coverage if score is not None else -1.0
            total = score.total if score is not None else -1.0
            return (-coverage, -total, sym)

        symbols = sorted(symbols, key=_sort_key)

    for sym in symbols:
        symbol_buffer = None
        symbol_console = None
        if save:
            import io

            symbol_buffer = io.StringIO()
            symbol_console = Console(file=symbol_buffer, width=200, force_terminal=False)

        history = cached_history.get(sym)
        if history is None:
            history = candle_store.get_daily_history(sym, period=period)
        if not history.empty:
            last_ts = history.index.max()
            # Candle store normalizes to tz-naive DatetimeIndex.
            symbol_candle_dates[sym] = last_ts.date()
            symbol_candle_datetimes[sym] = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
        as_of_date = symbol_candle_dates.get(sym)
        next_earnings_date = safe_next_earnings_date(earnings_store, sym)
        setup = cached_setup.get(sym)
        if setup is None:
            setup = analyze_underlying(sym, history=history, risk_profile=rp)
        ext_pct = cached_extension_pct.get(sym)

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

        expiry_strs = [d.isoformat() for d in provider.list_option_expiries(sym)]
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
            confluence_score = score_confluence(
                build_confluence_inputs(
                    setup,
                    extension_percentile=ext_pct,
                    vol_context=None,
                ),
                cfg=confluence_cfg,
            )
            total, coverage, detail = _format_confluence(confluence_score)
            emit(f"  - Confluence score: {total} (coverage {coverage})")
            emit(f"    - Components: {detail}")
            if confluence_score.warnings:
                emit(f"    - Confluence warnings: {', '.join(confluence_score.warnings)}")
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
            chain = provider.get_options_chain(sym, short_exp)
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
            chain = provider.get_options_chain(sym, long_exp)
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

        confluence_score = score_confluence(
            build_confluence_inputs(
                setup,
                extension_percentile=ext_pct,
                vol_context=vol_context,
            ),
            cfg=confluence_cfg,
        )
        total, coverage, detail = _format_confluence(confluence_score)
        emit(f"  - Confluence score: {total} (coverage {coverage})")
        emit(f"    - Components: {detail}")
        if confluence_score.warnings:
            emit(f"    - Confluence warnings: {', '.join(confluence_score.warnings)}")

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

    provider = cli_deps.build_provider()
    store = cli_deps.build_candle_store(candle_cache_dir, provider=provider)
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
    _ensure_pandas()
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
        snapshot_store = cli_deps.build_snapshot_store(snapshots_dir)

    provider = None if offline else cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(cache_dir, provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))

    history_by_symbol: dict[str, pd.DataFrame] = {}
    last_price_by_symbol: dict[str, float | None] = {}
    as_of_by_symbol: dict[str, date | None] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}
    snapshot_day_by_symbol: dict[str, pd.DataFrame] = {}
    chain_cache: dict[tuple[str, date], OptionsChain] = {}
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

    single_metrics: list[PositionMetrics] = []
    all_metrics: list[PositionMetrics] = []
    multi_leg_summaries: list[MultiLegSummary] = []
    advice_by_id: dict[str, Advice] = {}
    offline_missing: list[str] = []

    for p in portfolio.positions:
        try:
            if isinstance(p, MultiLegPosition):
                leg_metrics: list[PositionMetrics] = []
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

                for idx, leg in enumerate(p.legs, start=1):
                    snapshot_row = None
                    if offline:
                        snap_date = as_of_by_symbol.get(p.symbol)
                        df_snap = snapshot_day_by_symbol.get(p.symbol, pd.DataFrame())
                        row = None

                        if snap_date is None:
                            if not missing_as_of_warned:
                                offline_missing.append(f"{p.id}: missing offline as-of date for {p.symbol}")
                                missing_as_of_warned = True
                        elif df_snap.empty:
                            if not missing_day_warned:
                                offline_missing.append(
                                    f"{p.id}: missing snapshot day data for {p.symbol} (as-of {snap_date.isoformat()})"
                                )
                                missing_day_warned = True
                        else:
                            row = find_snapshot_row(
                                df_snap,
                                expiry=leg.expiry,
                                strike=leg.strike,
                                option_type=leg.option_type,
                            )
                            if row is None:
                                offline_missing.append(
                                    f"{p.id}: missing snapshot row for {p.symbol} {leg.expiry.isoformat()} "
                                    f"{leg.option_type} {leg.strike:g} (as-of {snap_date.isoformat()})"
                                )

                        snapshot_row = row if row is not None else {}
                    else:
                        row = None
                        if provider is not None:
                            key = (p.symbol, leg.expiry)
                            chain = chain_cache.get(key)
                            if chain is None:
                                chain = provider.get_options_chain(p.symbol, leg.expiry)
                                chain_cache[key] = chain
                            df_chain = chain.calls if leg.option_type == "call" else chain.puts
                            row = contract_row_by_strike(df_chain, leg.strike)
                        snapshot_row = row if row is not None else {}

                    leg_position = Position(
                        id=f"{p.id}:leg{idx}",
                        symbol=p.symbol,
                        option_type=leg.option_type,
                        expiry=leg.expiry,
                        strike=leg.strike,
                        contracts=leg.contracts,
                        cost_basis=0.0,
                        opened_at=p.opened_at,
                    )

                    metrics = _position_metrics(
                        provider,
                        leg_position,
                        risk_profile=portfolio.risk_profile,
                        underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                        underlying_last_price=last_price_by_symbol.get(p.symbol),
                        as_of=as_of_by_symbol.get(p.symbol),
                        next_earnings_date=next_earnings_by_symbol.get(p.symbol),
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
                if p.net_debit is None:
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
                if net_mark is not None and p.net_debit is not None:
                    net_pnl_abs = net_mark - p.net_debit
                    if p.net_debit > 0:
                        net_pnl_pct = net_pnl_abs / p.net_debit

                dte_min = min(dte_vals) if dte_vals else None
                dte_max = max(dte_vals) if dte_vals else None

                multi_leg_summaries.append(
                    MultiLegSummary(
                        position=p,
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
                provider,
                p,
                risk_profile=portfolio.risk_profile,
                underlying_history=history_by_symbol.get(p.symbol, pd.DataFrame()),
                underlying_last_price=last_price_by_symbol.get(p.symbol),
                as_of=as_of_by_symbol.get(p.symbol),
                next_earnings_date=next_earnings_by_symbol.get(p.symbol),
                snapshot_row=snapshot_row,
            )
            single_metrics.append(metrics)
            all_metrics.append(metrics)
            advice_by_id[p.id] = advise(metrics, portfolio)
        except DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")

    if not single_metrics and not multi_leg_summaries:
        raise typer.Exit(1)

    if single_metrics:
        render_positions(console, portfolio, single_metrics, advice_by_id)
    if multi_leg_summaries:
        render_multi_leg_positions(console, multi_leg_summaries)

    exposure = compute_portfolio_exposure(all_metrics)
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
