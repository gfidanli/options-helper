from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pandas as pd
from backtesting import Backtest, Strategy

from options_helper.technicals_backtesting.backtest.metrics import compute_custom_score, has_min_trades
from options_helper.technicals_backtesting.constraints import ConstraintError, evaluate_constraint


@dataclass(frozen=True)
class _Choice:
    """
    Wrapper to force SAMBO to treat numeric parameters as categorical choices.

    backtesting.py's SAMBO integration converts integer/float arrays into continuous
    ranges. Wrapping values keeps dtype=object so only these discrete values are explored.
    """

    value: Any

    def __int__(self) -> int:  # pragma: no cover - exercised indirectly
        return int(self.value)

    def __float__(self) -> float:  # pragma: no cover - exercised indirectly
        return float(self.value)


def _unwrap_value(val: Any) -> Any:
    if isinstance(val, _Choice):
        return val.value
    return val


def _prepare_search_space(search_space: dict, *, method: str) -> dict:
    if method != "sambo":
        return search_space
    prepared: dict[str, Any] = {}
    for k, values in search_space.items():
        # Keep bools as-is (they already become categorical); wrap ints/floats.
        if isinstance(values, (list, tuple)):
            out = []
            for v in values:
                if isinstance(v, bool):
                    out.append(v)
                elif isinstance(v, (int, float)):
                    out.append(_Choice(v))
                else:
                    out.append(v)
            prepared[k] = out
        else:
            prepared[k] = values
    return prepared


def _bt_kwargs(bt_cfg: dict) -> dict:
    sig = inspect.signature(Backtest.__init__)
    params = {
        "cash": bt_cfg["cash"],
        "commission": bt_cfg["commission"],
        "trade_on_close": bt_cfg["trade_on_close"],
        "exclusive_orders": bt_cfg["exclusive_orders"],
        "hedging": bt_cfg["hedging"],
        "margin": bt_cfg["margin"],
    }
    if "spread" in sig.parameters:
        params["spread"] = bt_cfg.get("slippage_bps", 0.0) / 10000.0
    return params


def _constraint_func(constraints: list[str]):
    def _inner(*args: Any, **kwargs: Any) -> bool:
        params: dict[str, Any] = {}
        if kwargs:
            params = {k: _unwrap_value(v) for k, v in kwargs.items()}
        elif args and isinstance(args[0], dict):
            params = {k: _unwrap_value(v) for k, v in args[0].items()}
        for expr in constraints:
            try:
                if not evaluate_constraint(expr, params):
                    return False
            except ConstraintError:
                return False
        return True

    return _inner


def optimize_params(
    df_features: pd.DataFrame,
    StrategyClass: type[Strategy],
    bt_cfg: dict,
    search_space: dict,
    constraints: list[str],
    maximize: str,
    method: str,
    sambo_cfg: dict,
    custom_score_cfg: dict,
    *,
    warmup_bars: int = 0,
    return_heatmap: bool = False,
) -> tuple[dict, pd.Series, pd.Series | None]:
    if df_features.empty:
        raise ValueError("Empty feature frame for optimization")

    df_bt = df_features.iloc[warmup_bars:] if warmup_bars > 0 else df_features
    df_bt = df_bt.dropna(subset=["Open", "High", "Low", "Close"])
    if df_bt.empty:
        raise ValueError("Backtest frame is empty after warmup filtering")

    bt = Backtest(df_bt, StrategyClass, **_bt_kwargs(bt_cfg))

    maximize_arg: Any = maximize
    if maximize == "custom_score":
        def _score(stats: Any) -> float:
            if not has_min_trades(stats, custom_score_cfg):
                return -1.0e12
            return compute_custom_score(stats, custom_score_cfg)

        maximize_arg = _score

    opt_kwargs: dict[str, Any] = {
        "maximize": maximize_arg,
        "method": method,
        "constraint": _constraint_func(constraints),
        "return_heatmap": return_heatmap,
    }

    sig = inspect.signature(bt.optimize)
    if "max_tries" in sig.parameters:
        opt_kwargs["max_tries"] = sambo_cfg.get("max_tries")
    if "random_state" in sig.parameters:
        opt_kwargs["random_state"] = sambo_cfg.get("random_state")

    prepared_space = _prepare_search_space(search_space, method=method)
    stats_or_tuple = bt.optimize(**prepared_space, **opt_kwargs)
    if return_heatmap:
        stats, heatmap = stats_or_tuple
    else:
        stats, heatmap = stats_or_tuple, None

    best_params = dict(stats._strategy.__dict__)
    best_params = {
        k: _unwrap_value(v)
        for k, v in best_params.items()
        if not k.startswith("_") and k in search_space
    }
    return best_params, stats, heatmap
