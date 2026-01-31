from __future__ import annotations

from typing import Any


def compute_custom_score(stats: Any, custom_cfg: dict) -> float:
    return_key = custom_cfg["return_key"]
    dd_key = custom_cfg["max_drawdown_key"]
    trades_key = custom_cfg["trades_key"]
    lam = custom_cfg["weights"]["drawdown_lambda"]
    mu = custom_cfg["weights"]["turnover_mu"]

    ret = float(stats.get(return_key, 0.0))
    dd = float(stats.get(dd_key, 0.0))
    trades = float(stats.get(trades_key, 0.0))
    score = ret - lam * abs(dd) - mu * trades
    return float(score)


def score_stats(stats: Any, maximize: str, custom_cfg: dict) -> float:
    if maximize == "custom_score":
        return compute_custom_score(stats, custom_cfg)
    try:
        return float(stats.get(maximize, 0.0))
    except Exception:  # noqa: BLE001
        return 0.0


def has_min_trades(stats: Any, custom_cfg: dict) -> bool:
    min_trades = int(custom_cfg.get("min_trades", 0))
    trades_key = custom_cfg["trades_key"]
    try:
        trades = float(stats.get(trades_key, 0.0))
    except Exception:  # noqa: BLE001
        trades = 0.0
    return trades >= min_trades
