from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import options_helper.cli_deps as cli_deps
from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.research_runtime import (
    _ResearchCaches,
    _ResearchOutputs,
    _ResearchRuntime,
    _SymbolContext,
    _HorizonSpec,
    _resolve_symbols,
    _load_configs,
    _new_outputs,
    _emit,
    _format_confluence,
    _precompute_symbol_caches,
    _sort_symbols_by_confluence,
    _symbol_output_sink,
    _record_symbol_candle_info,
    _emit_levels,
    _choose_long_expiry_fallback,
    _emit_neutral_result,
    _new_option_table,
    _process_horizon_candidate,
    _finalize_symbol_output,
)
from options_helper.storage import load_portfolio


def _render_ticker_entry(*, candle_day: date, run_dt: Any, body: str) -> str:
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


def _save_research_reports(runtime: _ResearchRuntime, outputs: _ResearchOutputs) -> None:
    if not runtime.save or outputs.report_buffer is None:
        return
    run_dt = runtime.workflows_pkg.datetime.now()
    candle_dt = max(outputs.symbol_candle_datetimes.values()) if outputs.symbol_candle_datetimes else run_dt
    candle_day = candle_dt.date()
    candle_stamp = candle_dt.strftime("%Y-%m-%d_%H%M%S")
    run_stamp = run_dt.strftime("%Y-%m-%d_%H%M%S")
    runtime.output_dir.mkdir(parents=True, exist_ok=True)

    out_path = runtime.output_dir / f"research-{candle_stamp}-{run_stamp}.txt"
    header = (
        f"run_at: {run_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"candles_through: {candle_dt.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"symbols: {', '.join(runtime.symbols)}\n\n"
    )
    out_path.write_text(header + outputs.report_buffer.getvalue().lstrip(), encoding="utf-8")

    tickers_dir = runtime.output_dir / "tickers"
    for sym, body in outputs.symbol_outputs.items():
        sym_day = outputs.symbol_candle_dates.get(sym) or candle_day
        entry = _render_ticker_entry(candle_day=sym_day, run_dt=run_dt, body=body)
        _upsert_ticker_entry(path=tickers_dir / f"{sym}.txt", candle_day=sym_day, new_entry=entry)
    outputs.console.print(f"\nSaved research report to {out_path}", soft_wrap=True)


def _process_symbol(
    *,
    runtime: _ResearchRuntime,
    outputs: _ResearchOutputs,
    caches: _ResearchCaches,
    pd: Any,
    sym: str,
) -> None:
    symbol_buffer, symbol_console = _symbol_output_sink(runtime.save)
    def emit(*args: Any, **kwargs: Any) -> None:
        _emit(outputs, symbol_console, *args, **kwargs)

    history = caches.cached_history.get(sym)
    if history is None:
        history = runtime.candle_store.get_daily_history(sym, period=runtime.period)
    as_of_date = _record_symbol_candle_info(outputs, sym, history)
    next_earnings_date = runtime.workflows_pkg.safe_next_earnings_date(runtime.earnings_store, sym)
    setup = caches.cached_setup.get(sym) or legacy.analyze_underlying(sym, history=history, risk_profile=runtime.rp)
    ext_pct = caches.cached_extension_pct.get(sym)
    emit(f"\n[bold]{sym}[/bold] — setup: {setup.direction.value}")
    for reason in setup.reasons:
        emit(f"  - {reason}")
    if setup.spot is None:
        emit("  - No spot price; skipping option selection.")
        return _finalize_symbol_output(outputs, sym=sym, symbol_buffer=symbol_buffer)

    _emit_levels(emit, setup=setup, history=history, rp=runtime.rp)
    expiry_strs = [d.isoformat() for d in runtime.provider.list_option_expiries(sym)]
    if not expiry_strs:
        emit("  - No listed option expirations found.")
        return _finalize_symbol_output(outputs, sym=sym, symbol_buffer=symbol_buffer)
    expiry_as_of = as_of_date or date.today()
    short_exp = legacy.choose_expiry(expiry_strs, min_dte=runtime.short_min_dte, max_dte=runtime.short_max_dte, target_dte=60, today=expiry_as_of)
    long_exp = legacy.choose_expiry(expiry_strs, min_dte=runtime.long_min_dte, max_dte=runtime.long_max_dte, target_dte=540, today=expiry_as_of)
    if long_exp is None:
        long_exp = _choose_long_expiry_fallback(expiry_strs, expiry_as_of=expiry_as_of, long_min_dte=runtime.long_min_dte)
    if setup.direction == legacy.Direction.NEUTRAL:
        _emit_neutral_result(emit, setup=setup, ext_pct=ext_pct, confluence_cfg=runtime.confluence_cfg)
        return _finalize_symbol_output(outputs, sym=sym, symbol_buffer=symbol_buffer)

    table = _new_option_table(sym)
    context = _SymbolContext(sym, "call" if setup.direction == legacy.Direction.BULLISH else "put", setup, history, runtime.derived_store.load(sym), expiry_as_of, next_earnings_date)
    vol_context = _process_horizon_candidate(
        runtime=runtime,
        context=context,
        spec=_HorizonSpec("30–90d", short_exp, 0.40 if context.opt_type == "call" else -0.40, f"  - No expiries found in {runtime.short_min_dte}-{runtime.short_max_dte} DTE range.", None),
        table=table,
        emit=emit,
        vol_context=None,
    )
    vol_context = _process_horizon_candidate(
        runtime=runtime,
        context=context,
        spec=_HorizonSpec("LEAPS", long_exp, 0.70 if context.opt_type == "call" else -0.70, f"  - No expiries found in {runtime.long_min_dte}-{runtime.long_max_dte} DTE range.", "Longer DTE reduces theta pressure."),
        table=table,
        emit=emit,
        vol_context=vol_context,
    )
    score = legacy.score_confluence(legacy.build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=vol_context), cfg=runtime.confluence_cfg)
    total, coverage, detail = _format_confluence(score)
    emit(f"  - Confluence score: {total} (coverage {coverage})")
    emit(f"    - Components: {detail}")
    if score.warnings:
        emit(f"    - Confluence warnings: {', '.join(score.warnings)}")
    emit(table)
    _finalize_symbol_output(outputs, sym=sym, symbol_buffer=symbol_buffer)


def run_research_command(params: dict[str, Any]) -> None:
    legacy._ensure_pandas()
    pd = legacy.pd
    assert pd is not None

    import options_helper.commands.workflows as workflows_pkg

    portfolio = load_portfolio(params["portfolio_path"])
    symbols = _resolve_symbols(symbol=params["symbol"], watchlists_path=params["watchlists_path"], watchlist=params["watchlist"])
    confluence_cfg, confluence_cfg_error, technicals_cfg, technicals_cfg_error = _load_configs()
    provider = cli_deps.build_provider()
    runtime = _ResearchRuntime(
        rp=portfolio.risk_profile,
        symbols=symbols,
        provider=provider,
        candle_store=cli_deps.build_candle_store(params["candle_cache_dir"], provider=provider),
        earnings_store=cli_deps.build_earnings_store(Path("data/earnings")),
        derived_store=cli_deps.build_derived_store(params["derived_dir"]),
        confluence_cfg=confluence_cfg,
        technicals_cfg=technicals_cfg,
        confluence_cfg_error=confluence_cfg_error,
        technicals_cfg_error=technicals_cfg_error,
        save=bool(params["save"]),
        output_dir=params["output_dir"],
        period=params["period"],
        window_pct=float(params["window_pct"]),
        short_min_dte=int(params["short_min_dte"]),
        short_max_dte=int(params["short_max_dte"]),
        long_min_dte=int(params["long_min_dte"]),
        long_max_dte=int(params["long_max_dte"]),
        include_bad_quotes=bool(params["include_bad_quotes"]),
        workflows_pkg=workflows_pkg,
    )
    outputs = _new_outputs(save=runtime.save)
    if runtime.confluence_cfg_error:
        _emit(outputs, None, f"[yellow]Warning:[/yellow] confluence config unavailable: {runtime.confluence_cfg_error}")
    if runtime.technicals_cfg_error:
        _emit(outputs, None, f"[yellow]Warning:[/yellow] technicals config unavailable: {runtime.technicals_cfg_error}")
    caches = _precompute_symbol_caches(runtime, pd)
    runtime = replace(runtime, symbols=_sort_symbols_by_confluence(runtime.symbols, caches.pre_confluence))
    for sym in runtime.symbols:
        _process_symbol(runtime=runtime, outputs=outputs, caches=caches, pd=pd, sym=sym)
    _save_research_reports(runtime, outputs)
