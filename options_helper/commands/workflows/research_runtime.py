from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

from options_helper.commands import workflows_legacy as legacy
from options_helper.watchlists import load_watchlists
from rich.console import Console
from rich.table import Table
import typer


@dataclass(frozen=True)
class _ResearchRuntime:
    rp: Any
    symbols: list[str]
    provider: Any
    candle_store: Any
    earnings_store: Any
    derived_store: Any
    confluence_cfg: Any
    technicals_cfg: Any
    confluence_cfg_error: str | None
    technicals_cfg_error: str | None
    save: bool
    output_dir: Path
    period: str
    window_pct: float
    short_min_dte: int
    short_max_dte: int
    long_min_dte: int
    long_max_dte: int
    include_bad_quotes: bool
    workflows_pkg: Any


@dataclass
class _ResearchOutputs:
    console: Console
    report_buffer: Any | None
    report_console: Console | None
    symbol_outputs: dict[str, str]
    symbol_candle_dates: dict[str, date]
    symbol_candle_datetimes: dict[str, Any]


@dataclass(frozen=True)
class _ResearchCaches:
    cached_history: dict[str, Any]
    cached_setup: dict[str, Any]
    cached_extension_pct: dict[str, float | None]
    pre_confluence: dict[str, Any]


@dataclass(frozen=True)
class _SymbolContext:
    sym: str
    opt_type: str
    setup: Any
    history: Any
    derived_history: Any
    expiry_as_of: date
    next_earnings_date: date | None


@dataclass(frozen=True)
class _HorizonSpec:
    label: str
    expiry: date | None
    target_delta: float
    no_expiry_message: str
    extra_rationale: str | None


def _resolve_symbols(
    *,
    symbol: str | None,
    watchlists_path: Path,
    watchlist: str,
) -> list[str]:
    if symbol:
        return [symbol.strip().upper()]
    wl = load_watchlists(watchlists_path)
    symbols = wl.get(watchlist)
    if not symbols:
        raise typer.BadParameter(f"Watchlist '{watchlist}' is empty or missing in {watchlists_path}")
    return symbols


def _load_configs() -> tuple[Any, str | None, Any, str | None]:
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
    return confluence_cfg, confluence_cfg_error, technicals_cfg, technicals_cfg_error


def _new_outputs(*, save: bool) -> _ResearchOutputs:
    console = Console()
    if not save:
        return _ResearchOutputs(console, None, None, {}, {}, {})
    import io

    report_buffer = io.StringIO()
    report_console = Console(file=report_buffer, width=200, force_terminal=False)
    return _ResearchOutputs(console, report_buffer, report_console, {}, {}, {})


def _emit(
    outputs: _ResearchOutputs,
    symbol_console: Console | None,
    *args: Any,
    **kwargs: Any,
) -> None:
    outputs.console.print(*args, **kwargs)
    if outputs.report_console is not None:
        outputs.report_console.print(*args, **kwargs)
    if symbol_console is not None:
        symbol_console.print(*args, **kwargs)


def _format_confluence(score: Any) -> tuple[str, str, str]:
    total = f"{score.total:.0f}"
    coverage = f"{score.coverage * 100.0:.0f}%"
    parts = []
    for comp in score.components:
        if comp.score is None or abs(comp.score) < 1e-9:
            continue
        parts.append(f"{comp.name} {comp.score:+.0f}")
    detail = ", ".join(parts) if parts else "neutral"
    return total, coverage, detail


def _precompute_symbol_caches(runtime: _ResearchRuntime, pd: Any) -> _ResearchCaches:
    cached_history: dict[str, Any] = {}
    cached_setup: dict[str, Any] = {}
    cached_extension_pct: dict[str, float | None] = {}
    pre_confluence: dict[str, Any] = {}
    for sym in runtime.symbols:
        history = runtime.candle_store.get_daily_history(sym, period=runtime.period)
        cached_history[sym] = history
        setup = legacy.analyze_underlying(sym, history=history, risk_profile=runtime.rp)
        cached_setup[sym] = setup
        ext_pct = None
        if runtime.technicals_cfg is not None and history is not None and not history.empty:
            try:
                ext = legacy.compute_current_extension_percentile(history, runtime.technicals_cfg)
                ext_pct = ext.percentile
            except Exception:  # noqa: BLE001
                ext_pct = None
        cached_extension_pct[sym] = ext_pct
        inputs = legacy.build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None)
        pre_confluence[sym] = legacy.score_confluence(inputs, cfg=runtime.confluence_cfg)
    return _ResearchCaches(cached_history, cached_setup, cached_extension_pct, pre_confluence)


def _sort_symbols_by_confluence(symbols: list[str], pre_confluence: dict[str, Any]) -> list[str]:
    if len(symbols) <= 1:
        return symbols

    def _sort_key(sym: str) -> tuple[float, float, str]:
        score = pre_confluence.get(sym)
        coverage = score.coverage if score is not None else -1.0
        total = score.total if score is not None else -1.0
        return (-coverage, -total, sym)

    return sorted(symbols, key=_sort_key)


def _symbol_output_sink(save: bool) -> tuple[Any | None, Console | None]:
    if not save:
        return None, None
    import io

    symbol_buffer = io.StringIO()
    symbol_console = Console(file=symbol_buffer, width=200, force_terminal=False)
    return symbol_buffer, symbol_console


def _record_symbol_candle_info(outputs: _ResearchOutputs, sym: str, history: Any) -> date | None:
    if history.empty:
        return None
    last_ts = history.index.max()
    outputs.symbol_candle_dates[sym] = last_ts.date()
    outputs.symbol_candle_datetimes[sym] = last_ts.to_pydatetime() if hasattr(last_ts, "to_pydatetime") else last_ts
    return outputs.symbol_candle_dates[sym]


def _emit_levels(emit: Callable[..., None], setup: Any, history: Any, rp: Any) -> None:
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


def _choose_long_expiry_fallback(expiry_strs: list[str], *, expiry_as_of: date, long_min_dte: int) -> date | None:
    parsed: list[tuple[int, date]] = []
    for expiry_str in expiry_strs:
        try:
            exp = date.fromisoformat(expiry_str)
        except ValueError:
            continue
        dte = (exp - expiry_as_of).days
        if dte >= long_min_dte:
            parsed.append((dte, exp))
    if not parsed:
        return None
    _, picked = max(parsed, key=lambda item: item[0])
    return picked


def _emit_neutral_result(
    emit: Callable[..., None],
    *,
    setup: Any,
    ext_pct: float | None,
    confluence_cfg: Any,
) -> None:
    confluence_score = legacy.score_confluence(
        legacy.build_confluence_inputs(setup, extension_percentile=ext_pct, vol_context=None),
        cfg=confluence_cfg,
    )
    total, coverage, detail = _format_confluence(confluence_score)
    emit(f"  - Confluence score: {total} (coverage {coverage})")
    emit(f"    - Components: {detail}")
    if confluence_score.warnings:
        emit(f"    - Confluence warnings: {', '.join(confluence_score.warnings)}")
    emit("  - No strong directional setup; skipping contract recommendations.")


def _new_option_table(sym: str) -> Table:
    table = Table(title=f"{sym} option ideas (best-effort)")
    for column, justify in [
        ("Horizon", None),
        ("Expiry", None),
        ("DTE", "right"),
        ("Type", None),
        ("Strike", "right"),
        ("Entry", "right"),
        ("Δ", "right"),
        ("IV", "right"),
        ("IV/RV20", "right"),
        ("IV pct", "right"),
        ("OI", "right"),
        ("Vol", "right"),
        ("Spr%", "right"),
        ("Exec", "right"),
        ("Quality", "right"),
        ("Stale", "right"),
        ("Why", None),
    ]:
        table.add_column(column) if justify is None else table.add_column(column, justify=justify)
    return table


def _fmt_stale(age_days: int | None) -> str:
    if age_days is None:
        return "-"
    age = int(age_days)
    return f"{age}d" if age > 5 else "-"


def _fmt_iv_rv(ctx: Any) -> str:
    if ctx is None or ctx.iv_rv_20d is None:
        return "-"
    return f"{ctx.iv_rv_20d:.2f}x"


def _fmt_iv_pct(ctx: Any) -> str:
    if ctx is None or ctx.iv_percentile is None:
        return "-"
    return f"{ctx.iv_percentile:.0f}"


def _process_horizon_candidate(
    *,
    runtime: _ResearchRuntime,
    context: _SymbolContext,
    spec: _HorizonSpec,
    table: Table,
    emit: Callable[..., None],
    vol_context: Any | None,
) -> Any | None:
    if spec.expiry is None:
        emit(spec.no_expiry_message)
        return vol_context
    chain = runtime.provider.get_options_chain(context.sym, spec.expiry)
    if vol_context is None:
        vol_context = legacy.compute_volatility_context(
            history=context.history,
            spot=context.setup.spot,
            calls=chain.calls,
            puts=chain.puts,
            derived_history=context.derived_history,
        )
    df = chain.calls if context.opt_type == "call" else chain.puts
    pick = legacy.select_option_candidate(
        df,
        symbol=context.sym,
        option_type=context.opt_type,
        expiry=spec.expiry,
        spot=context.setup.spot,
        target_delta=spec.target_delta,
        window_pct=runtime.window_pct,
        min_open_interest=runtime.rp.min_open_interest,
        min_volume=runtime.rp.min_volume,
        as_of=context.expiry_as_of,
        next_earnings_date=context.next_earnings_date,
        earnings_warn_days=runtime.rp.earnings_warn_days,
        earnings_avoid_days=runtime.rp.earnings_avoid_days,
        include_bad_quotes=runtime.include_bad_quotes,
    )
    if pick is None:
        return vol_context
    if pick.exclude:
        warn = ", ".join(pick.warnings) if pick.warnings else "earnings_unknown"
        emit(f"  - Excluded {spec.label} candidate due to earnings_avoid_days ({warn}).")
        return vol_context

    why_parts = pick.rationale[:2]
    if spec.extra_rationale is not None:
        why_parts.append(spec.extra_rationale)
    why = "; ".join(why_parts)
    if pick.warnings:
        why = f"{why}; Warnings: {', '.join(pick.warnings)}"
    table.add_row(
        spec.label,
        pick.expiry.isoformat(),
        str(pick.dte),
        pick.option_type,
        f"{pick.strike:g}",
        "-" if pick.mark is None else f"${pick.mark:.2f}",
        "-" if pick.delta is None else f"{pick.delta:+.2f}",
        "-" if pick.iv is None else f"{pick.iv:.1%}",
        _fmt_iv_rv(vol_context),
        _fmt_iv_pct(vol_context),
        "-" if pick.open_interest is None else str(pick.open_interest),
        "-" if pick.volume is None else str(pick.volume),
        "-" if pick.spread_pct is None else f"{pick.spread_pct:.1%}",
        "-" if pick.execution_quality is None else pick.execution_quality,
        "-" if pick.quality_label is None else pick.quality_label,
        _fmt_stale(pick.last_trade_age_days),
        why,
    )
    if pick.warnings:
        emit(f"  - Earnings warnings ({spec.label}): {', '.join(pick.warnings)}")
    if pick.quality_warnings:
        emit(f"  - Quote warnings ({spec.label}): {', '.join(pick.quality_warnings)}")
    return vol_context


def _finalize_symbol_output(outputs: _ResearchOutputs, *, sym: str, symbol_buffer: Any | None) -> None:
    if symbol_buffer is None:
        return
    outputs.symbol_outputs[sym] = symbol_buffer.getvalue().lstrip()
