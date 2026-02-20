from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

import pandas as pd

from options_helper.reporting_briefing import BriefingSymbolSection
from options_helper.technicals_backtesting.snapshot import TechnicalSnapshot

if TYPE_CHECKING:
    from options_helper.models import Position


@dataclass(frozen=True)
class _SymbolSelection:
    symbols: list[str]
    symbol_sources_payload: list[dict[str, object]]
    watchlists_payload: list[dict[str, object]]


@dataclass(frozen=True)
class _SymbolResult:
    section: BriefingSymbolSection
    day_entry: tuple[date, pd.DataFrame] | None
    candles: pd.DataFrame
    next_earnings_date: date | None


@dataclass(frozen=True)
class _PortfolioOutputs:
    rows_payload: list[dict[str, object]]
    table_markdown: str | None
    metrics: list[Any]


@dataclass(frozen=True)
class _SymbolState:
    day_entry: tuple[date, pd.DataFrame] | None
    candles: pd.DataFrame
    chain: Any
    compare_report: Any
    flow_net: pd.DataFrame | None
    technicals: TechnicalSnapshot | None
    confluence_score: Any
    quote_quality: Any
    derived_updated: bool
    derived_row: Any
    warnings: list[str]
    errors: list[str]


@dataclass(frozen=True)
class _RuntimePrep:
    portfolio: Any
    selection: _SymbolSelection
    positions_by_symbol: dict[str, list[Position]]
    store: Any
    derived_store: Any
    candle_store: Any
    earnings_store: Any
    technicals_cfg: dict[str, Any] | None
    technicals_cfg_error: str | None
    confluence_cfg: Any
    compare_norm: str
    compare_enabled: bool
