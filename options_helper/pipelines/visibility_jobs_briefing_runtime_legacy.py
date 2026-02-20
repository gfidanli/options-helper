from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rich.console import RenderableType
from rich.markdown import Markdown

from options_helper.reporting_briefing import BriefingSymbolSection

from .visibility_jobs_briefing_artifacts_legacy import _write_briefing_artifacts
from .visibility_jobs_briefing_models_legacy import _RuntimePrep
from .visibility_jobs_briefing_portfolio_legacy import _collect_portfolio_outputs, _portfolio_risk_outputs
from .visibility_jobs_briefing_selection_legacy import (
    _load_optional_configs,
    _load_watchlist_symbols,
    _positions_by_symbol,
    _resolve_symbol_selection,
)
from .visibility_jobs_briefing_symbol_state_legacy import _build_symbol_section


def _prepare_runtime(
    *,
    portfolio_path: Path,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: str | None,
    compare: str,
    cache_dir: Path,
    candle_cache_dir: Path,
    technicals_config: Path,
    derived_dir: Path,
    snapshot_store_builder: Callable[[Path], Any],
    derived_store_builder: Callable[[Path], Any],
    candle_store_builder: Callable[..., Any],
    earnings_store_builder: Callable[[Path], Any],
    portfolio_loader: Callable[[Path], Any],
    watchlists_loader: Callable[[Path], Any],
    active_snapshot_store_fn: Callable[[Any], Any],
    filesystem_compatible_derived_store_fn: Callable[[Path, Any], Any],
    filesystem_compatible_candle_store_fn: Callable[[Path, Any], Any],
    execution_error_factory: Callable[..., Exception],
    renderables: list[RenderableType],
) -> _RuntimePrep:
    portfolio = portfolio_loader(portfolio_path)
    positions_by_symbol = _positions_by_symbol(portfolio.positions)
    portfolio_symbols = sorted({position.symbol.upper() for position in portfolio.positions})
    watch_symbols, watchlist_symbols_by_name = _load_watchlist_symbols(
        watchlist=watchlist,
        watchlists_path=watchlists_path,
        watchlists_loader=watchlists_loader,
        renderables=renderables,
    )
    selection = _resolve_symbol_selection(
        portfolio_symbols=portfolio_symbols,
        watch_symbols=watch_symbols,
        watchlist_symbols_by_name=watchlist_symbols_by_name,
        watchlist=watchlist,
        symbol=symbol,
    )
    if not selection.symbols:
        raise execution_error_factory("no symbols selected (empty portfolio and no watchlists)")

    technicals_cfg, technicals_cfg_error, confluence_cfg = _load_optional_configs(
        technicals_config=technicals_config,
        renderables=renderables,
    )
    compare_norm = compare.strip().lower()
    return _RuntimePrep(
        portfolio=portfolio,
        selection=selection,
        positions_by_symbol=positions_by_symbol,
        store=active_snapshot_store_fn(snapshot_store_builder(cache_dir)),
        derived_store=filesystem_compatible_derived_store_fn(derived_dir, derived_store_builder(derived_dir)),
        candle_store=filesystem_compatible_candle_store_fn(candle_cache_dir, candle_store_builder(candle_cache_dir)),
        earnings_store=earnings_store_builder(Path("data/earnings")),
        technicals_cfg=technicals_cfg,
        technicals_cfg_error=technicals_cfg_error,
        confluence_cfg=confluence_cfg,
        compare_norm=compare_norm,
        compare_enabled=compare_norm not in {"none", "off", "false", "0"},
    )


def _process_symbols(
    *,
    runtime: _RuntimePrep,
    as_of: str,
    top: int,
    update_derived: bool,
    safe_next_earnings_date_fn: Callable[..., date | None],
    execution_error_factory: Callable[..., Exception],
) -> tuple[list[BriefingSymbolSection], dict[str, tuple[date, pd.DataFrame]], dict[str, pd.DataFrame], dict[str, date | None], list[date]]:
    sections: list[BriefingSymbolSection] = []
    day_cache: dict[str, tuple[date, pd.DataFrame]] = {}
    candles_by_symbol: dict[str, pd.DataFrame] = {}
    next_earnings_by_symbol: dict[str, date | None] = {}
    resolved_to_dates: list[date] = []

    for sym in runtime.selection.symbols:
        result = _build_symbol_section(
            sym=sym,
            as_of=as_of,
            compare_enabled=runtime.compare_enabled,
            compare_norm=runtime.compare_norm,
            top=top,
            store=runtime.store,
            candle_store=runtime.candle_store,
            derived_store=runtime.derived_store,
            earnings_store=runtime.earnings_store,
            technicals_cfg=runtime.technicals_cfg,
            technicals_cfg_error=runtime.technicals_cfg_error,
            confluence_cfg=runtime.confluence_cfg,
            risk_profile=runtime.portfolio.risk_profile,
            positions_for_symbol=runtime.positions_by_symbol.get(sym, []),
            update_derived=update_derived,
            safe_next_earnings_date_fn=safe_next_earnings_date_fn,
            execution_error_factory=execution_error_factory,
        )
        if result.day_entry is not None:
            day_cache[sym] = result.day_entry
            resolved_to_dates.append(result.day_entry[0])
        candles_by_symbol[sym] = result.candles
        next_earnings_by_symbol[sym] = result.next_earnings_date
        sections.append(result.section)
    return sections, day_cache, candles_by_symbol, next_earnings_by_symbol, resolved_to_dates


def _build_symbol_outputs(
    *,
    runtime: _RuntimePrep,
    as_of: str,
    top: int,
    update_derived: bool,
    safe_next_earnings_date_fn: Callable[..., date | None],
    execution_error_factory: Callable[..., Exception],
) -> tuple[list[BriefingSymbolSection], dict[str, tuple[date, pd.DataFrame]], dict[str, pd.DataFrame], dict[str, date | None], str]:
    sections, day_cache, candles_by_symbol, next_earnings_by_symbol, resolved_to_dates = _process_symbols(
        runtime=runtime,
        as_of=as_of,
        top=top,
        update_derived=update_derived,
        safe_next_earnings_date_fn=safe_next_earnings_date_fn,
        execution_error_factory=execution_error_factory,
    )
    if not resolved_to_dates:
        raise execution_error_factory("no snapshots found for selected symbols")
    report_date = max(resolved_to_dates).isoformat()
    return sections, day_cache, candles_by_symbol, next_earnings_by_symbol, report_date


def _run_briefing_job_runtime(*, params: dict[str, Any]) -> Any:
    renderables: list[RenderableType] = []
    runtime = _prepare_runtime(
        portfolio_path=params["portfolio_path"],
        watchlists_path=params["watchlists_path"],
        watchlist=params["watchlist"],
        symbol=params["symbol"],
        compare=params["compare"],
        cache_dir=params["cache_dir"],
        candle_cache_dir=params["candle_cache_dir"],
        technicals_config=params["technicals_config"],
        derived_dir=params["derived_dir"],
        snapshot_store_builder=params["snapshot_store_builder"],
        derived_store_builder=params["derived_store_builder"],
        candle_store_builder=params["candle_store_builder"],
        earnings_store_builder=params["earnings_store_builder"],
        portfolio_loader=params["portfolio_loader"],
        watchlists_loader=params["watchlists_loader"],
        active_snapshot_store_fn=params["active_snapshot_store_fn"],
        filesystem_compatible_derived_store_fn=params["filesystem_compatible_derived_store_fn"],
        filesystem_compatible_candle_store_fn=params["filesystem_compatible_candle_store_fn"],
        execution_error_factory=params["execution_error_factory"],
        renderables=renderables,
    )
    sections, day_cache, candles_by_symbol, next_earnings_by_symbol, report_date = _build_symbol_outputs(
        runtime=runtime,
        as_of=params["as_of"],
        top=params["top"],
        update_derived=params["update_derived"],
        safe_next_earnings_date_fn=params["safe_next_earnings_date_fn"],
        execution_error_factory=params["execution_error_factory"],
    )
    portfolio_outputs = _collect_portfolio_outputs(
        portfolio_positions=runtime.portfolio.positions,
        risk_profile=runtime.portfolio.risk_profile,
        day_cache=day_cache,
        candles_by_symbol=candles_by_symbol,
        next_earnings_by_symbol=next_earnings_by_symbol,
        candle_store=runtime.candle_store,
        renderables=renderables,
    )
    portfolio_exposure, portfolio_stress = _portfolio_risk_outputs(portfolio_outputs)
    markdown, markdown_path, json_path = _write_briefing_artifacts(
        report_date=report_date,
        portfolio_path=params["portfolio_path"],
        sections=sections,
        portfolio_outputs=portfolio_outputs,
        top=params["top"],
        out=params["out"],
        write_json=params["write_json"],
        strict=params["strict"],
        technicals_config=params["technicals_config"],
        selection=runtime.selection,
        portfolio_exposure=portfolio_exposure,
        portfolio_stress=portfolio_stress,
        renderables=renderables,
    )
    if params["print_to_console"]:
        renderables.append(Markdown(markdown))
    return params["result_factory"](
        report_date=report_date,
        markdown=markdown,
        markdown_path=markdown_path,
        json_path=json_path,
        renderables=renderables,
    )


def run_briefing_job_impl(
    *,
    portfolio_path: Path,
    watchlists_path: Path,
    watchlist: list[str],
    symbol: str | None,
    as_of: str,
    compare: str,
    cache_dir: Path,
    candle_cache_dir: Path,
    technicals_config: Path,
    out: Path | None,
    print_to_console: bool,
    write_json: bool,
    strict: bool,
    update_derived: bool,
    derived_dir: Path,
    top: int,
    snapshot_store_builder: Callable[[Path], Any],
    derived_store_builder: Callable[[Path], Any],
    candle_store_builder: Callable[..., Any],
    earnings_store_builder: Callable[[Path], Any],
    safe_next_earnings_date_fn: Callable[..., date | None],
    portfolio_loader: Callable[[Path], Any],
    watchlists_loader: Callable[[Path], Any],
    active_snapshot_store_fn: Callable[[Any], Any],
    filesystem_compatible_derived_store_fn: Callable[[Path, Any], Any],
    filesystem_compatible_candle_store_fn: Callable[[Path, Any], Any],
    execution_error_factory: Callable[..., Exception],
    result_factory: Callable[..., Any],
) -> Any:
    return _run_briefing_job_runtime(params=locals().copy())
