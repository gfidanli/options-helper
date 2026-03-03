from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from options_helper.commands.technicals.backtest_batch_runtime_costs import (
    merge_batch_backtest_cost_overrides,
)
from options_helper.technicals_backtesting.backtest.batch_runner import (
    BatchBacktestResult,
    Clock,
    ProgressCallback,
    run_backtest_batch,
)


def _require_mapping(
    *,
    mapping: Mapping[str, object],
    key: str,
    context: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must define mapping '{key}'.")
    return value


def _resolve_strategy_cfg(
    *,
    cfg: Mapping[str, object],
    strategy: str,
) -> Mapping[str, object]:
    strategies = _require_mapping(mapping=cfg, key="strategies", context="technical config")
    strategy_cfg = strategies.get(strategy)
    if not isinstance(strategy_cfg, Mapping):
        raise ValueError(f"Unknown strategy: {strategy}")
    if not bool(strategy_cfg.get("enabled", False)):
        raise ValueError(f"Strategy is disabled in config: {strategy}")
    return strategy_cfg


def _resolve_strategy_defaults(strategy_cfg: Mapping[str, object]) -> dict[str, object]:
    defaults = strategy_cfg.get("defaults") or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("Strategy config defaults must be a mapping.")
    return dict(defaults)


def _select_strategy_feature_frame(
    *,
    features: pd.DataFrame,
    needed_columns: Sequence[str],
) -> pd.DataFrame:
    columns: list[str] = ["Open", "High", "Low", "Close"]
    if "Volume" in features.columns:
        columns.append("Volume")
    columns.extend(str(column) for column in needed_columns)
    selected = [column for column in dict.fromkeys(columns) if column in features.columns]
    return features.loc[:, selected]


def run_technicals_backtest_batch_runtime(
    *,
    symbols: str | Sequence[str],
    strategy: str,
    cfg: Mapping[str, object],
    cache_dir: Path,
    cli_commission: float | None = None,
    cli_slippage_bps: float | None = None,
    progress_callback: ProgressCallback | None = None,
    clock: Clock | None = None,
) -> BatchBacktestResult:
    from options_helper.data.technical_backtesting_io import load_ohlc_from_cache
    from options_helper.technicals_backtesting.backtest.runner import run_backtest
    from options_helper.technicals_backtesting.feature_selection import required_feature_columns_for_strategy
    from options_helper.technicals_backtesting.pipeline import compute_features, warmup_bars
    from options_helper.technicals_backtesting.strategies.registry import get_strategy

    backtest_cfg = _require_mapping(mapping=cfg, key="backtest", context="technical config")
    strategy_cfg = _resolve_strategy_cfg(cfg=cfg, strategy=strategy)
    needed_columns = tuple(required_feature_columns_for_strategy(strategy, dict(strategy_cfg)))
    strategy_defaults = _resolve_strategy_defaults(strategy_cfg)
    resolved_backtest_cfg = merge_batch_backtest_cost_overrides(
        global_backtest_cfg=backtest_cfg,
        strategy_cfg=strategy_cfg,
        cli_commission=cli_commission,
        cli_slippage_bps=cli_slippage_bps,
    )
    warmup = int(warmup_bars(cfg))
    strategy_class = get_strategy(strategy)

    def _load_ohlc(symbol_token: str) -> pd.DataFrame:
        return load_ohlc_from_cache(
            symbol_token,
            cache_dir,
            backfill_if_missing=True,
            period="max",
            raise_on_backfill_error=True,
        )

    def _compute_features(_symbol: str, ohlc: pd.DataFrame) -> pd.DataFrame:
        return compute_features(ohlc, cfg)

    def _select_features(_symbol: str, features: pd.DataFrame) -> pd.DataFrame:
        return _select_strategy_feature_frame(features=features, needed_columns=needed_columns)

    def _run_strategy(_symbol: str, strategy_features: pd.DataFrame) -> object:
        return run_backtest(
            strategy_features,
            strategy_class,
            resolved_backtest_cfg,
            strategy_defaults,
            warmup_bars=warmup,
            indicator_cols=needed_columns,
        )

    return run_backtest_batch(
        symbols=symbols,
        load_ohlc=_load_ohlc,
        compute_features=_compute_features,
        select_strategy_features=_select_features,
        run_strategy_backtest=_run_strategy,
        progress_callback=progress_callback,
        clock=clock,
    )


__all__ = [
    "run_technicals_backtest_batch_runtime",
]
