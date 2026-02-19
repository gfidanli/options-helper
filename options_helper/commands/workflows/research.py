from __future__ import annotations

from datetime import date
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
    legacy._ensure_pandas()
    pd = legacy.pd
    assert pd is not None

    import options_helper.commands.workflows as workflows_pkg

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
        confluence_cfg = legacy.load_confluence_config()
    except legacy.ConfluenceConfigError as exc:
        confluence_cfg_error = str(exc)
    try:
        technicals_cfg = legacy.load_technical_backtesting_config()
    except legacy.TechnicalConfigError as exc:
        technicals_cfg_error = str(exc)
    console = Console()

    from rich.table import Table

    report_buffer = None
    report_console = None
    symbol_outputs: dict[str, str] = {}
    symbol_candle_dates: dict[str, date] = {}
    symbol_candle_datetimes: dict[str, Any] = {}
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

    def _format_confluence(score: Any) -> tuple[str, str, str]:
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

    cached_history: dict[str, Any] = {}
    cached_setup: dict[str, Any] = {}
    cached_extension_pct: dict[str, float | None] = {}
    pre_confluence: dict[str, Any] = {}

    for sym in symbols:
        history = candle_store.get_daily_history(sym, period=period)
        cached_history[sym] = history
        setup = legacy.analyze_underlying(sym, history=history, risk_profile=rp)
        cached_setup[sym] = setup

        ext_pct = None
        if technicals_cfg is not None and history is not None and not history.empty:
            try:
                ext_result = legacy.compute_current_extension_percentile(history, technicals_cfg)
                ext_pct = ext_result.percentile
            except Exception:  # noqa: BLE001
                ext_pct = None
        cached_extension_pct[sym] = ext_pct

        inputs = legacy.build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
        pre_confluence[sym] = legacy.score_confluence(inputs, cfg=confluence_cfg)

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
            symbol_candle_dates[sym] = last_ts.date()
            symbol_candle_datetimes[sym] = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
        as_of_date = symbol_candle_dates.get(sym)
        next_earnings_date = workflows_pkg.safe_next_earnings_date(earnings_store, sym)
        setup = cached_setup.get(sym)
        if setup is None:
            setup = legacy.analyze_underlying(sym, history=history, risk_profile=rp)
        ext_pct = cached_extension_pct.get(sym)

        emit(f"\n[bold]{sym}[/bold] — setup: {setup.direction.value}")
        for reason in setup.reasons:
            emit(f"  - {reason}")

        if setup.spot is None:
            emit("  - No spot price; skipping option selection.")
            continue

        levels = legacy.suggest_trade_levels(setup, history=history, risk_profile=rp)
        if levels.entry is not None:
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
        short_exp = legacy.choose_expiry(
            expiry_strs, min_dte=short_min_dte, max_dte=short_max_dte, target_dte=60, today=expiry_as_of
        )
        long_exp = legacy.choose_expiry(
            expiry_strs, min_dte=long_min_dte, max_dte=long_max_dte, target_dte=540, today=expiry_as_of
        )
        if long_exp is None:
            parsed = []
            for expiry_str in expiry_strs:
                try:
                    exp = date.fromisoformat(expiry_str)
                except ValueError:
                    continue
                dte = (exp - expiry_as_of).days
                parsed.append((dte, exp))
            parsed = [item for item in parsed if item[0] >= long_min_dte]
            if parsed:
                _, long_exp = max(parsed, key=lambda item: item[0])

        if setup.direction == legacy.Direction.NEUTRAL:
            confluence_score = legacy.score_confluence(
                legacy.build_confluence_inputs(
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

        opt_type = "call" if setup.direction == legacy.Direction.BULLISH else "put"
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
                vol_context = legacy.compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.40 if opt_type == "call" else -0.40
            short_pick = legacy.select_option_candidate(
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
                vol_context = legacy.compute_volatility_context(
                    history=history,
                    spot=setup.spot,
                    calls=chain.calls,
                    puts=chain.puts,
                    derived_history=derived_history,
                )
            df = chain.calls if opt_type == "call" else chain.puts
            target_delta = 0.70 if opt_type == "call" else -0.70
            long_pick = legacy.select_option_candidate(
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

        confluence_score = legacy.score_confluence(
            legacy.build_confluence_inputs(
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

    def _render_ticker_entry(*, sym: str, candle_day: date, run_dt: Any, body: str) -> str:
        run_ts = run_dt.strftime("%Y-%m-%d %H:%M:%S")
        header = f"=== {candle_day.isoformat()} ===\nrun_at: {run_ts}\ncandles_through: {candle_day.isoformat()}\n"
        return f"{header}\n{body.strip()}\n"

    def _parse_ticker_entries(text: str) -> dict[str, str]:
        import re

        pattern = re.compile(r"^=== (\d{4}-\d{2}-\d{2}) ===$", re.M)
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
        out = "\n\n".join(entries[day] for day in ordered_days).rstrip() + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(out, encoding="utf-8")

    if save and report_buffer is not None:
        run_dt = workflows_pkg.datetime.now()
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
