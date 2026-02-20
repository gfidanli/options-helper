from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import RenderableType

from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.technical_backtesting_config import load_technical_backtesting_config

from .visibility_jobs_briefing_models_legacy import _SymbolSelection

if TYPE_CHECKING:
    from options_helper.models import Position


def _load_watchlist_symbols(
    *,
    watchlist: list[str],
    watchlists_path: Path,
    watchlists_loader: Callable[[Path], Any],
    renderables: list[RenderableType],
) -> tuple[list[str], dict[str, list[str]]]:
    if not watchlist:
        return [], {}

    watch_symbols: list[str] = []
    symbols_by_name: dict[str, list[str]] = {}
    try:
        watchlists = watchlists_loader(watchlists_path)
        for name in watchlist:
            symbols_for_name = watchlists.get(name)
            watch_symbols.extend(symbols_for_name)
            symbols_by_name[name] = symbols_for_name
    except Exception as exc:  # noqa: BLE001
        renderables.append(f"[yellow]Warning:[/yellow] failed to load watchlists: {exc}")
    return watch_symbols, symbols_by_name


def _resolve_symbol_selection(
    *,
    portfolio_symbols: list[str],
    watch_symbols: list[str],
    watchlist_symbols_by_name: dict[str, list[str]],
    watchlist: list[str],
    symbol: str | None,
) -> _SymbolSelection:
    symbols = sorted(set(portfolio_symbols).union({value.upper() for value in watch_symbols if value}))
    if symbol is not None:
        symbols = [symbol.upper().strip()]

    if symbol is not None:
        symbol_sources_map: dict[str, set[str]] = {}
        resolved_symbol = symbols[0] if symbols else ""
        if resolved_symbol:
            if resolved_symbol in portfolio_symbols:
                symbol_sources_map.setdefault(resolved_symbol, set()).add("portfolio")
            symbol_sources_map.setdefault(resolved_symbol, set()).add("manual")
        watchlists_payload: list[dict[str, object]] = []
    else:
        symbol_sources_map = {value: {"portfolio"} for value in portfolio_symbols}
        for name, symbols_for_name in watchlist_symbols_by_name.items():
            for symbol_name in symbols_for_name:
                symbol_sources_map.setdefault(symbol_name, set()).add(f"watchlist:{name}")
        watchlists_payload = [
            {"name": name, "symbols": watchlist_symbols_by_name.get(name, [])}
            for name in watchlist
            if name in watchlist_symbols_by_name
        ]

    symbol_sources_payload = [
        {"symbol": symbol_name, "sources": sorted(symbol_sources_map.get(symbol_name, set()))}
        for symbol_name in symbols
    ]
    return _SymbolSelection(
        symbols=symbols,
        symbol_sources_payload=symbol_sources_payload,
        watchlists_payload=watchlists_payload,
    )


def _load_optional_configs(
    *,
    technicals_config: Path,
    renderables: list[RenderableType],
) -> tuple[dict[str, Any] | None, str | None, Any | None]:
    technicals_cfg: dict[str, Any] | None = None
    technicals_cfg_error: str | None = None
    try:
        technicals_cfg = load_technical_backtesting_config(technicals_config)
    except Exception as exc:  # noqa: BLE001
        technicals_cfg_error = str(exc)

    confluence_cfg = None
    try:
        confluence_cfg = load_confluence_config()
    except ConfluenceConfigError as exc:
        renderables.append(f"[yellow]Warning:[/yellow] confluence config unavailable: {exc}")
    return technicals_cfg, technicals_cfg_error, confluence_cfg


def _positions_by_symbol(portfolio_positions: list[Position]) -> dict[str, list[Position]]:
    grouped: dict[str, list[Position]] = {}
    for position in portfolio_positions:
        grouped.setdefault(position.symbol.upper(), []).append(position)
    return grouped
