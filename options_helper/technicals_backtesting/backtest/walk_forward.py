from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from options_helper.technicals_backtesting.backtest.metrics import has_min_trades, score_stats
from options_helper.technicals_backtesting.backtest.optimizer import optimize_params
from options_helper.technicals_backtesting.backtest.runner import run_backtest


@dataclass(frozen=True)
class WalkForwardResult:
    params: dict[str, Any]
    folds: list[dict[str, Any]]
    stability: dict[str, Any]
    used_defaults: bool
    reason: str | None


def _generate_splits(
    idx: pd.DatetimeIndex,
    train_years: float,
    validate_months: float,
    step_months: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if idx.empty:
        return []
    start = idx.min()
    last = idx.max()

    train_offset = pd.DateOffset(years=train_years)
    validate_offset = pd.DateOffset(months=validate_months)
    step_offset = pd.DateOffset(months=step_months)

    splits: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    while True:
        train_end = start + train_offset
        val_start = train_end + timedelta(days=1)
        val_end = val_start + validate_offset - timedelta(days=1)
        if val_end > last:
            break
        splits.append((start, train_end, val_start, val_end))
        start = start + step_offset
    return splits


def walk_forward_optimize(
    df_features: pd.DataFrame,
    StrategyClass: type,
    bt_cfg: dict,
    search_space: dict,
    constraints: list[str],
    maximize: str,
    method: str,
    sambo_cfg: dict,
    custom_score_cfg: dict,
    walk_cfg: dict,
    defaults: dict,
    *,
    warmup_bars: int = 0,
    min_train_bars: int = 0,
    return_heatmap: bool = False,
) -> WalkForwardResult:
    if df_features.empty:
        return WalkForwardResult(defaults, [], {"stable": False}, True, "empty_data")

    total_years = (df_features.index.max() - df_features.index.min()).days / 365.25
    if total_years < walk_cfg["min_history_years"]:
        return WalkForwardResult(defaults, [], {"stable": False}, True, "insufficient_history")

    splits = _generate_splits(
        df_features.index,
        walk_cfg["train_years"],
        walk_cfg["validate_months"],
        walk_cfg["step_months"],
    )

    folds: list[dict[str, Any]] = []
    for train_start, train_end, val_start, val_end in splits:
        train_df = df_features.loc[train_start:train_end]
        validate_df = df_features.loc[val_start:val_end]
        if min_train_bars and len(train_df) < min_train_bars:
            continue

        best_params, train_stats, heatmap = optimize_params(
            train_df,
            StrategyClass,
            bt_cfg,
            search_space,
            constraints,
            maximize,
            method,
            sambo_cfg,
            custom_score_cfg,
            warmup_bars=warmup_bars,
            return_heatmap=return_heatmap,
        )
        validate_stats = run_backtest(
            validate_df,
            StrategyClass,
            bt_cfg,
            best_params,
            warmup_bars=warmup_bars,
        )
        validate_score = score_stats(validate_stats, maximize, custom_score_cfg)

        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "validate_start": val_start,
                "validate_end": val_end,
                "best_params": best_params,
                "train_stats": train_stats,
                "validate_stats": validate_stats,
                "validate_score": validate_score,
                "heatmap": heatmap,
            }
        )

    if not folds:
        return WalkForwardResult(defaults, [], {"stable": False}, True, "no_folds")

    scores = np.array([f["validate_score"] for f in folds], dtype=float)
    mean_score = float(np.nanmean(scores))
    std_score = float(np.nanstd(scores))
    cv = float(std_score / abs(mean_score)) if mean_score != 0 else float("inf")
    stability_cfg = walk_cfg.get("selection", {}).get("stability") or walk_cfg.get("stability", {})
    max_cv = float(stability_cfg.get("max_validate_score_cv", 1.0))

    valid_trade_folds = [has_min_trades(f["validate_stats"], custom_score_cfg) for f in folds]
    stable = cv <= max_cv and all(valid_trade_folds)

    params_by_key: dict[tuple[tuple[str, Any], ...], list[float]] = {}
    for fold in folds:
        key = tuple(sorted(fold["best_params"].items()))
        params_by_key.setdefault(key, []).append(float(fold["validate_score"]))
    best_key = max(params_by_key, key=lambda k: np.mean(params_by_key[k]))
    selected_params = dict(best_key)

    stability = {
        "validate_score_mean": mean_score,
        "validate_score_std": std_score,
        "validate_score_cv": cv,
        "threshold": max_cv,
        "stable": stable,
    }

    if not stable:
        return WalkForwardResult(defaults, folds, stability, True, "unstable_or_low_trades")

    return WalkForwardResult(selected_params, folds, stability, False, None)
