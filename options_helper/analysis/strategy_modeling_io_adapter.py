"""Explicit I/O adapter seam for strategy-modeling analysis modules.

This module is the only analysis-layer location allowed to import
`options_helper.data.strategy_modeling_io` directly.
"""

from __future__ import annotations

from options_helper.data.strategy_modeling_io import (
    AdjustedDataFallbackMode,
    DailySourceMode,
    StrategyModelingDailyLoadResult,
    StrategyModelingIntradayLoadResult,
    StrategyModelingIntradayPreflightResult,
    StrategyModelingUniverseLoadResult,
    build_required_intraday_sessions,
    list_strategy_modeling_universe,
    load_daily_ohlc_history,
    load_required_intraday_bars,
    normalize_symbol,
)

__all__ = [
    "AdjustedDataFallbackMode",
    "DailySourceMode",
    "StrategyModelingDailyLoadResult",
    "StrategyModelingIntradayLoadResult",
    "StrategyModelingIntradayPreflightResult",
    "StrategyModelingUniverseLoadResult",
    "build_required_intraday_sessions",
    "list_strategy_modeling_universe",
    "load_daily_ohlc_history",
    "load_required_intraday_bars",
    "normalize_symbol",
]
