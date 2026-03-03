from __future__ import annotations

import pytest

from options_helper.analysis.regime_tactic import map_regime_to_tactic


def test_map_regime_to_tactic_prefers_breakout_when_trend_aligns() -> None:
    result = map_regime_to_tactic("trend_up", "trend_up", direction="long")

    assert result.tactic == "breakout"
    assert result.support_model == "ema"
    assert result.rationale


def test_map_regime_to_tactic_prefers_undercut_reclaim_in_choppy_sideways_context() -> None:
    result = map_regime_to_tactic("choppy", "sideways", direction="long")

    assert result.tactic == "undercut_reclaim"
    assert result.support_model == "static"
    assert any("Range-bound" in item for item in result.rationale)


@pytest.mark.parametrize(
    ("market_regime", "symbol_regime", "expected_tactic", "expected_support_model"),
    [
        ("mixed", "trend_up", "undercut_reclaim", "static"),
        ("trend_up", "mixed", "undercut_reclaim", "static"),
        ("trend_up", "trend_down", "avoid", "static"),
    ],
)
def test_map_regime_to_tactic_handles_mixed_edge_cases_deterministically(
    market_regime: str,
    symbol_regime: str,
    expected_tactic: str,
    expected_support_model: str,
) -> None:
    first = map_regime_to_tactic(market_regime, symbol_regime, direction="long")
    second = map_regime_to_tactic(market_regime, symbol_regime, direction="long")

    assert first == second
    assert first.tactic == expected_tactic
    assert first.support_model == expected_support_model


def test_map_regime_to_tactic_orients_trend_for_short_direction() -> None:
    result = map_regime_to_tactic("trend_down", "trend_down", direction="short")

    assert result.tactic == "breakout"
    assert result.support_model == "ema"
