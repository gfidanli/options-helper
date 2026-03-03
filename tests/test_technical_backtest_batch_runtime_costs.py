from __future__ import annotations

from copy import deepcopy

import pytest

from options_helper.commands.technicals.backtest_batch_runtime_costs import (
    merge_batch_backtest_cost_overrides,
    resolve_batch_backtest_cost_overrides,
)


@pytest.mark.parametrize(
    ("cli_commission", "cli_slippage_bps", "strategy_cfg", "expected"),
    [
        (
            None,
            None,
            None,
            {"commission": 0.001, "slippage_bps": 3.0},
        ),
        (
            None,
            None,
            {"cost_overrides": {"commission": 0.0025, "slippage_bps": 7.0}},
            {"commission": 0.0025, "slippage_bps": 7.0},
        ),
        (
            0.004,
            1.5,
            {"cost_overrides": {"commission": 0.0025, "slippage_bps": 7.0}},
            {"commission": 0.004, "slippage_bps": 1.5},
        ),
        (
            0.0035,
            None,
            {"cost_overrides": {"commission": 0.0025, "slippage_bps": 7.0}},
            {"commission": 0.0035, "slippage_bps": 7.0},
        ),
        (
            None,
            2.25,
            {"cost_overrides": {"commission": 0.0025, "slippage_bps": 7.0}},
            {"commission": 0.0025, "slippage_bps": 2.25},
        ),
        (
            None,
            None,
            {"cost_overrides": {"commission": 0.0025}},
            {"commission": 0.0025, "slippage_bps": 3.0},
        ),
        (
            None,
            None,
            {"cost_overrides": {"slippage_bps": 9.0}},
            {"commission": 0.001, "slippage_bps": 9.0},
        ),
        (
            0.0,
            0.0,
            {"cost_overrides": {"commission": 0.0025, "slippage_bps": 7.0}},
            {"commission": 0.0, "slippage_bps": 0.0},
        ),
        (
            None,
            None,
            {"defaults": {"lookback_high": 10}},
            {"commission": 0.001, "slippage_bps": 3.0},
        ),
    ],
)
def test_resolve_batch_backtest_cost_overrides_precedence(
    cli_commission: float | None,
    cli_slippage_bps: float | None,
    strategy_cfg: dict | None,
    expected: dict[str, float],
) -> None:
    global_backtest_cfg = {"cash": 100000.0, "commission": 0.001, "slippage_bps": 3.0}
    global_before = deepcopy(global_backtest_cfg)
    strategy_before = deepcopy(strategy_cfg)

    resolved = resolve_batch_backtest_cost_overrides(
        global_backtest_cfg=global_backtest_cfg,
        strategy_cfg=strategy_cfg,
        cli_commission=cli_commission,
        cli_slippage_bps=cli_slippage_bps,
    )

    assert resolved == expected
    assert global_backtest_cfg == global_before
    assert strategy_cfg == strategy_before


def test_merge_batch_backtest_cost_overrides_preserves_non_cost_keys() -> None:
    global_backtest_cfg = {
        "cash": 100000.0,
        "commission": 0.001,
        "slippage_bps": 3.0,
        "trade_on_close": False,
    }
    strategy_cfg = {"cost_overrides": {"commission": 0.002}}

    merged = merge_batch_backtest_cost_overrides(
        global_backtest_cfg=global_backtest_cfg,
        strategy_cfg=strategy_cfg,
        cli_slippage_bps=1.25,
    )

    assert merged == {
        "cash": 100000.0,
        "commission": 0.002,
        "slippage_bps": 1.25,
        "trade_on_close": False,
    }
    assert merged is not global_backtest_cfg
