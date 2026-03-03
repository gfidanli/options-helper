from __future__ import annotations

from typing import Mapping


_COST_KEYS: tuple[str, str] = ("commission", "slippage_bps")


def resolve_batch_backtest_cost_overrides(
    *,
    global_backtest_cfg: Mapping[str, object],
    strategy_cfg: Mapping[str, object] | None = None,
    cli_commission: float | None = None,
    cli_slippage_bps: float | None = None,
) -> dict[str, float]:
    """Resolve per-run cost inputs using CLI > strategy override > global config."""

    strategy_overrides = {}
    if strategy_cfg is not None:
        candidate = strategy_cfg.get("cost_overrides")
        if isinstance(candidate, Mapping):
            strategy_overrides = candidate

    cli_overrides = {
        "commission": cli_commission,
        "slippage_bps": cli_slippage_bps,
    }

    resolved: dict[str, float] = {}
    for key in _COST_KEYS:
        cli_value = cli_overrides[key]
        if cli_value is not None:
            resolved[key] = float(cli_value)
            continue

        strategy_value = strategy_overrides.get(key)
        if strategy_value is not None:
            resolved[key] = float(strategy_value)
            continue

        resolved[key] = float(global_backtest_cfg[key])
    return resolved


def merge_batch_backtest_cost_overrides(
    *,
    global_backtest_cfg: Mapping[str, object],
    strategy_cfg: Mapping[str, object] | None = None,
    cli_commission: float | None = None,
    cli_slippage_bps: float | None = None,
) -> dict[str, object]:
    """Return a backtest config copy with resolved batch-cost values applied."""

    merged = dict(global_backtest_cfg)
    merged.update(
        resolve_batch_backtest_cost_overrides(
            global_backtest_cfg=global_backtest_cfg,
            strategy_cfg=strategy_cfg,
            cli_commission=cli_commission,
            cli_slippage_bps=cli_slippage_bps,
        )
    )
    return merged


__all__ = [
    "merge_batch_backtest_cost_overrides",
    "resolve_batch_backtest_cost_overrides",
]
