from __future__ import annotations

from options_helper.technicals_backtesting.constraints import evaluate_constraint


def test_constraint_evaluator() -> None:
    params = {"p_entry": 0.1, "p_exit": 0.5, "stop_mult_atr": 0.0}
    assert evaluate_constraint("p_entry < p_exit", params)
    assert not evaluate_constraint("p_entry > p_exit", params)
    assert evaluate_constraint("stop_mult_atr >= 0.0", params)
    assert not evaluate_constraint("stop_mult_atr > 1.0", params)
