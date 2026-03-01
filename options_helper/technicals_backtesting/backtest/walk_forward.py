from __future__ import annotations

from dataclasses import dataclass
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


SplitRange = tuple[int, int, int, int]


def _walk_cfg_int(walk_cfg: dict, key: str, default: int = 0) -> int:
    value = walk_cfg.get(key, default)
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _use_bar_splits(walk_cfg: dict) -> bool:
    return _walk_cfg_int(walk_cfg, "train_bars") > 0


def _generate_bar_splits(
    idx: pd.DatetimeIndex,
    train_bars: int,
    validate_bars: int,
    step_bars: int,
) -> list[SplitRange]:
    if idx.empty or train_bars <= 0 or validate_bars <= 0:
        return []
    if step_bars <= 0:
        return []

    n_rows = len(idx)
    splits: list[SplitRange] = []
    start_pos = 0
    while True:
        train_end_pos = start_pos + train_bars - 1
        val_start_pos = train_end_pos + 1
        val_end_pos = val_start_pos + validate_bars - 1
        if val_end_pos >= n_rows:
            break
        splits.append((start_pos, train_end_pos, val_start_pos, val_end_pos))
        start_pos += step_bars
        if start_pos >= n_rows:
            break
    return splits


def _generate_date_splits(
    idx: pd.DatetimeIndex,
    train_years: float,
    validate_months: float,
    step_months: float,
) -> list[SplitRange]:
    if idx.empty:
        return []
    if train_years <= 0 or validate_months <= 0 or step_months <= 0:
        return []

    train_offset = pd.DateOffset(years=train_years)
    validate_offset = pd.DateOffset(months=validate_months)
    step_offset = pd.DateOffset(months=step_months)

    splits: list[SplitRange] = []
    start_pos = 0
    last_pos = len(idx) - 1
    last_ts = idx[last_pos]

    while True:
        if start_pos > last_pos:
            break
        start_ts = idx[start_pos]
        train_end_target = start_ts + train_offset
        if train_end_target > last_ts:
            break
        train_end_pos = int(idx.searchsorted(train_end_target, side="right")) - 1
        if train_end_pos < start_pos:
            break

        val_start_pos = train_end_pos + 1
        if val_start_pos > last_pos:
            break
        val_start_ts = idx[val_start_pos]
        val_end_target = val_start_ts + validate_offset
        if val_end_target > last_ts:
            break
        val_end_pos = int(idx.searchsorted(val_end_target, side="left")) - 1
        if val_end_pos < val_start_pos:
            break

        splits.append((start_pos, train_end_pos, val_start_pos, val_end_pos))

        next_start_target = start_ts + step_offset
        next_start_pos = int(idx.searchsorted(next_start_target, side="left"))
        if next_start_pos <= start_pos:
            next_start_pos = start_pos + 1
        start_pos = next_start_pos
    return splits


def _defaults_result(defaults: dict, reason: str) -> WalkForwardResult:
    return WalkForwardResult(defaults, [], {"stable": False}, True, reason)


def _prepare_walk_forward_data(
    *,
    df_features: pd.DataFrame,
    walk_cfg: dict,
    defaults: dict,
    warmup_bars: int,
) -> tuple[pd.DataFrame | None, WalkForwardResult | None]:
    if df_features.empty:
        return None, _defaults_result(defaults, "empty_data")
    df_used = df_features.iloc[warmup_bars:] if warmup_bars > 0 else df_features
    if df_used.empty:
        return None, _defaults_result(defaults, "empty_after_warmup")
    total_bars = len(df_used)
    min_history_bars = _walk_cfg_int(walk_cfg, "min_history_bars")
    if _use_bar_splits(walk_cfg):
        train_bars = _walk_cfg_int(walk_cfg, "train_bars")
        validate_bars = _walk_cfg_int(walk_cfg, "validate_bars")
        required_bars = max(min_history_bars, train_bars + validate_bars)
        if required_bars > 0 and total_bars < required_bars:
            return None, _defaults_result(defaults, "insufficient_history")
        return df_used, None

    if min_history_bars > 0 and total_bars < min_history_bars:
        return None, _defaults_result(defaults, "insufficient_history")
    min_history_years = float(walk_cfg.get("min_history_years", 0.0) or 0.0)
    if min_history_years > 0:
        total_years = (df_used.index.max() - df_used.index.min()).days / 365.25
        if total_years < min_history_years:
            return None, _defaults_result(defaults, "insufficient_history")
    return df_used, None


def _generate_splits(idx: pd.DatetimeIndex, walk_cfg: dict) -> list[SplitRange]:
    if _use_bar_splits(walk_cfg):
        train_bars = _walk_cfg_int(walk_cfg, "train_bars")
        validate_bars = _walk_cfg_int(walk_cfg, "validate_bars")
        step_bars = _walk_cfg_int(walk_cfg, "step_bars", validate_bars)
        return _generate_bar_splits(idx, train_bars, validate_bars, step_bars)
    return _generate_date_splits(
        idx,
        float(walk_cfg.get("train_years", 0.0) or 0.0),
        float(walk_cfg.get("validate_months", 0.0) or 0.0),
        float(walk_cfg.get("step_months", 0.0) or 0.0),
    )


def _run_walk_forward_fold(
    *,
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    StrategyClass: type,
    bt_cfg: dict,
    search_space: dict,
    constraints: list[str],
    maximize: str,
    method: str,
    sambo_cfg: dict,
    custom_score_cfg: dict,
    return_heatmap: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], float, Any]:
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
        warmup_bars=0,
        return_heatmap=return_heatmap,
    )
    validate_stats = run_backtest(
        validate_df,
        StrategyClass,
        bt_cfg,
        best_params,
        warmup_bars=0,
    )
    validate_score = score_stats(validate_stats, maximize, custom_score_cfg)
    return best_params, train_stats, validate_stats, validate_score, heatmap


def _build_folds(
    *,
    df_used: pd.DataFrame,
    splits: list[SplitRange],
    StrategyClass: type,
    bt_cfg: dict,
    search_space: dict,
    constraints: list[str],
    maximize: str,
    method: str,
    sambo_cfg: dict,
    custom_score_cfg: dict,
    min_train_bars: int,
    return_heatmap: bool,
) -> list[dict[str, Any]]:
    folds: list[dict[str, Any]] = []
    index = df_used.index
    for train_start_pos, train_end_pos, val_start_pos, val_end_pos in splits:
        train_df = df_used.iloc[train_start_pos : train_end_pos + 1]
        validate_df = df_used.iloc[val_start_pos : val_end_pos + 1]
        if min_train_bars and len(train_df) < min_train_bars:
            continue
        best_params, train_stats, validate_stats, validate_score, heatmap = _run_walk_forward_fold(
            train_df=train_df,
            validate_df=validate_df,
            StrategyClass=StrategyClass,
            bt_cfg=bt_cfg,
            search_space=search_space,
            constraints=constraints,
            maximize=maximize,
            method=method,
            sambo_cfg=sambo_cfg,
            custom_score_cfg=custom_score_cfg,
            return_heatmap=return_heatmap,
        )
        folds.append(
            {
                "train_start": index[train_start_pos],
                "train_end": index[train_end_pos],
                "validate_start": index[val_start_pos],
                "validate_end": index[val_end_pos],
                "best_params": best_params,
                "train_stats": train_stats,
                "validate_stats": validate_stats,
                "validate_score": validate_score,
                "heatmap": heatmap,
            }
        )
    return folds


def _compute_stability(folds: list[dict[str, Any]], *, walk_cfg: dict, custom_score_cfg: dict) -> dict[str, Any]:
    scores = np.array([f["validate_score"] for f in folds], dtype=float)
    mean_score = float(np.nanmean(scores))
    std_score = float(np.nanstd(scores))
    cv = float(std_score / abs(mean_score)) if mean_score != 0 else float("inf")
    stability_cfg = walk_cfg.get("selection", {}).get("stability") or walk_cfg.get("stability", {})
    max_cv = float(stability_cfg.get("max_validate_score_cv", 1.0))
    valid_trade_folds = [has_min_trades(f["validate_stats"], custom_score_cfg) for f in folds]
    stable = cv <= max_cv and all(valid_trade_folds)
    return {
        "validate_score_mean": mean_score,
        "validate_score_std": std_score,
        "validate_score_cv": cv,
        "threshold": max_cv,
        "stable": stable,
    }


def _select_params_from_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    params_by_key: dict[tuple[tuple[str, Any], ...], list[float]] = {}
    for fold in folds:
        key = tuple(sorted(fold["best_params"].items()))
        params_by_key.setdefault(key, []).append(float(fold["validate_score"]))
    best_key = max(params_by_key, key=lambda key: np.mean(params_by_key[key]))
    return dict(best_key)


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
    df_used, early_result = _prepare_walk_forward_data(
        df_features=df_features,
        walk_cfg=walk_cfg,
        defaults=defaults,
        warmup_bars=warmup_bars,
    )
    if early_result is not None:
        return early_result
    assert df_used is not None
    splits = _generate_splits(df_used.index, walk_cfg)
    folds = _build_folds(
        df_used=df_used,
        splits=splits,
        StrategyClass=StrategyClass,
        bt_cfg=bt_cfg,
        search_space=search_space,
        constraints=constraints,
        maximize=maximize,
        method=method,
        sambo_cfg=sambo_cfg,
        custom_score_cfg=custom_score_cfg,
        min_train_bars=min_train_bars,
        return_heatmap=return_heatmap,
    )

    if not folds:
        return _defaults_result(defaults, "no_folds")
    stability = _compute_stability(folds, walk_cfg=walk_cfg, custom_score_cfg=custom_score_cfg)
    if not bool(stability["stable"]):
        return WalkForwardResult(defaults, folds, stability, True, "unstable_or_low_trades")
    return WalkForwardResult(_select_params_from_folds(folds), folds, stability, False, None)
