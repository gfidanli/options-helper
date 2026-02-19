from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.research_analyze_runtime import (
    _AnalyzeResults,
    _build_analyze_symbol_data,
    _process_multileg_position,
    _process_single_position,
    _validate_analyze_interval,
)
from options_helper.commands.workflows.research_runtime_runner import run_research_command
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists


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
    run_research_command(dict(locals()))


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


def _collect_analyze_results(
    *,
    portfolio: Any,
    workflows_pkg: Any,
    symbol_data: Any,
    provider: Any,
    offline: bool,
    console: Console,
    pd: Any,
) -> _AnalyzeResults:
    results = _AnalyzeResults([], [], [], {}, [])
    for position in portfolio.positions:
        try:
            if isinstance(position, legacy.MultiLegPosition):
                summary, leg_metrics = _process_multileg_position(
                    position=position,
                    portfolio=portfolio,
                    workflows_pkg=workflows_pkg,
                    symbol_data=symbol_data,
                    provider=provider,
                    offline=offline,
                    offline_missing=results.offline_missing,
                    pd=pd,
                )
                results.multi_leg_summaries.append(summary)
                results.all_metrics.extend(leg_metrics)
                continue
            metrics, advice = _process_single_position(
                position=position,
                portfolio=portfolio,
                workflows_pkg=workflows_pkg,
                symbol_data=symbol_data,
                provider=provider,
                offline=offline,
                offline_missing=results.offline_missing,
                pd=pd,
            )
            results.single_metrics.append(metrics)
            results.all_metrics.append(metrics)
            results.advice_by_id[position.id] = advice
        except legacy.DataFetchError as exc:
            console.print(f"[red]Data error:[/red] {exc}")
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")
    return results


def run_analyze_command(params: dict[str, Any]) -> None:
    legacy._ensure_pandas()
    pd = legacy.pd
    assert pd is not None

    _validate_analyze_interval(params["interval"])

    import options_helper.commands.workflows as workflows_pkg

    portfolio = load_portfolio(params["portfolio_path"])
    console = Console()
    legacy.render_summary(console, portfolio)

    if not portfolio.positions:
        console.print("No positions.")
        raise typer.Exit(0)

    offline = bool(params["offline"])
    snapshot_store = cli_deps.build_snapshot_store(params["snapshots_dir"]) if offline else None
    provider = None if offline else cli_deps.build_provider()
    candle_store = cli_deps.build_candle_store(params["cache_dir"], provider=provider)
    earnings_store = cli_deps.build_earnings_store(Path("data/earnings"))
    symbol_data = _build_analyze_symbol_data(
        symbols={position.symbol for position in portfolio.positions},
        offline=offline,
        as_of=params["as_of"],
        period=params["period"],
        candle_store=candle_store,
        snapshot_store=snapshot_store,
        earnings_store=earnings_store,
        workflows_pkg=workflows_pkg,
        console=console,
        pd=pd,
    )
    results = _collect_analyze_results(
        portfolio=portfolio,
        workflows_pkg=workflows_pkg,
        symbol_data=symbol_data,
        provider=provider,
        offline=offline,
        console=console,
        pd=pd,
    )
    if not results.single_metrics and not results.multi_leg_summaries:
        raise typer.Exit(1)

    if results.single_metrics:
        legacy.render_positions(console, portfolio, results.single_metrics, results.advice_by_id)
    if results.multi_leg_summaries:
        legacy.render_multi_leg_positions(console, results.multi_leg_summaries)

    exposure = legacy.compute_portfolio_exposure(results.all_metrics)
    legacy._render_portfolio_risk(
        console,
        exposure,
        stress_spot_pct=params["stress_spot_pct"],
        stress_vol_pp=params["stress_vol_pp"],
        stress_days=params["stress_days"],
    )
    if results.offline_missing:
        for msg in results.offline_missing:
            console.print(f"[yellow]Warning:[/yellow] {msg}")
        if params["offline_strict"]:
            raise typer.Exit(1)


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
    run_analyze_command(dict(locals()))


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
