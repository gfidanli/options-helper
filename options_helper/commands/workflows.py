from __future__ import annotations

from contextlib import contextmanager
import json
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.greeks import add_black_scholes_greeks_to_chain
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.portfolio_risk import PortfolioExposure, compute_portfolio_exposure, run_stress
from options_helper.analysis.quote_quality import compute_quote_quality
from options_helper.analysis.research import (
    Direction,
    UnderlyingSetup,
    analyze_underlying,
    build_confluence_inputs,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.commands.common import _build_stress_scenarios, _parse_date
from options_helper.commands.position_metrics import _extract_float, _position_metrics
from options_helper.data.candles import close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.earnings import EarningsRecord, safe_next_earnings_date
from options_helper.data.market_types import DataFetchError
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.technical_backtesting_config import (
    ConfigError as TechnicalConfigError,
    load_technical_backtesting_config,
)
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.models import MultiLegPosition, OptionType, Position
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobParameterError,
    run_snapshot_options_job,
)
from options_helper.reporting import MultiLegSummary, render_multi_leg_positions, render_positions, render_summary
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


pd: object | None = None

JOB_SNAPSHOT_OPTIONS = "snapshot_options"
ASSET_OPTIONS_SNAPSHOTS = "options_snapshots"

NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)

_SAVED_SNAPSHOT_RE = re.compile(r"^([A-Z0-9._-]+)\s+\d{4}-\d{2}-\d{2}: saved\b")
_WARNING_SYMBOL_RE = re.compile(r"warning:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)
_ERROR_SYMBOL_RE = re.compile(r"error:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _snapshot_status_by_symbol(*, symbols: list[str], messages: list[str]) -> dict[str, str]:
    status_by_symbol = {sym.upper(): "skipped" for sym in symbols}
    for message in messages:
        plain = _strip_rich_markup(message)
        saved_match = _SAVED_SNAPSHOT_RE.match(plain)
        if saved_match:
            status_by_symbol[saved_match.group(1).upper()] = "success"
            continue

        error_match = _ERROR_SYMBOL_RE.search(plain)
        if error_match:
            status_by_symbol[error_match.group(1).upper()] = "failed"
            continue

        if "skipping snapshot" not in plain.lower():
            continue
        warning_match = _WARNING_SYMBOL_RE.search(plain)
        if warning_match and status_by_symbol.get(warning_match.group(1).upper()) != "success":
            status_by_symbol[warning_match.group(1).upper()] = "skipped"
    return status_by_symbol


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


def register(app: typer.Typer) -> None:
    app.command("daily")(daily_performance)
    app.command("snapshot-options")(snapshot_options)
    app.command("earnings")(earnings)
    app.command("refresh-earnings")(refresh_earnings)
    app.command("research")(research)
    app.command("refresh-candles")(refresh_candles)
    app.command("analyze")(analyze)
    app.command("watch")(watch)


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
    with _observed_run(
        console=console,
        job_name=JOB_SNAPSHOT_OPTIONS,
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
            result = run_snapshot_options_job(
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
        except VisibilityJobParameterError as exc:
            if exc.param_hint:
                raise typer.BadParameter(str(exc), param_hint=exc.param_hint) from exc
            raise typer.BadParameter(str(exc)) from exc

        for message in result.messages:
            console.print(message)

        if result.no_symbols:
            run_logger.log_asset_skipped(
                asset_key=ASSET_OPTIONS_SNAPSHOTS,
                asset_kind="file",
                partition_key="ALL",
                extra={"reason": "no_symbols"},
            )
            raise typer.Exit(0)

        max_data_date = max(result.dates_used) if result.dates_used else None
        status_by_symbol = _snapshot_status_by_symbol(symbols=result.symbols, messages=result.messages)

        for sym in result.symbols:
            status = status_by_symbol.get(sym.upper(), "skipped")
            if status == "success":
                run_logger.log_asset_success(
                    asset_key=ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    min_event_ts=max_data_date,
                    max_event_ts=max_data_date,
                )
                if max_data_date is not None:
                    run_logger.upsert_watermark(
                        asset_key=ASSET_OPTIONS_SNAPSHOTS,
                        scope_key=sym.upper(),
                        watermark_ts=max_data_date,
                    )
            elif status == "failed":
                run_logger.log_asset_failure(
                    asset_key=ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    extra={"reason": "snapshot_failed"},
                )
            else:
                run_logger.log_asset_skipped(
                    asset_key=ASSET_OPTIONS_SNAPSHOTS,
                    asset_kind="file",
                    partition_key=sym.upper(),
                    extra={"reason": "snapshot_skipped"},
                )

        if max_data_date is not None:
            run_logger.upsert_watermark(
                asset_key=ASSET_OPTIONS_SNAPSHOTS,
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

    record: EarningsRecord | None = None

    if set_date is not None:
        d = _parse_date(set_date)
        record = EarningsRecord.manual(symbol=sym, next_earnings_date=d, note="Set via CLI --set.")
        out_path = store.save(record)
        console.print(f"Saved: {out_path}")
    else:
        record = store.load(sym)
        if refresh or record is None:
            # Earnings data is sourced from Yahoo via yfinance (best-effort).
            provider = cli_deps.build_provider("yahoo")
            try:
                ev = provider.get_next_earnings_event(sym)
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

    store = cli_deps.build_earnings_store(cache_dir)
    # Earnings data is sourced from Yahoo via yfinance (best-effort).
    provider = cli_deps.build_provider("yahoo")

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
    chain_cache: dict[tuple[str, date], object] = {}
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
