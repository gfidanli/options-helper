from __future__ import annotations

import pytest

from options_helper.analysis.confluence import ConfluenceInputs, score_confluence


def test_confluence_score_all_components_aligned() -> None:
    inputs = ConfluenceInputs(
        weekly_trend="up",
        extension_percentile=2.0,
        rsi_divergence="bullish",
        flow_delta_oi_notional=2_500_000.0,
        iv_rv_20d=1.4,
    )
    result = score_confluence(inputs)

    assert result.coverage == pytest.approx(1.0)
    assert result.total == pytest.approx(96.4, rel=1e-3)
    assert not result.warnings

    comps = {c.name: c for c in result.components}
    assert comps["weekly_trend"].score == 25.0
    assert comps["extension"].score == 10.0
    assert comps["flow_alignment"].score == 20.0
    assert comps["rsi_divergence"].score == 5.0
    assert comps["iv_regime"].score == 5.0


def test_confluence_score_missing_components_reduces_coverage() -> None:
    inputs = ConfluenceInputs(weekly_trend="up")
    result = score_confluence(inputs)

    assert result.total == pytest.approx(67.86, rel=1e-3)
    assert result.coverage == pytest.approx(25.0 / 70.0, rel=1e-4)
    assert "partial_coverage" in result.warnings

    comps = {c.name: c for c in result.components}
    assert comps["weekly_trend"].score == 25.0
    assert comps["extension"].score is None
    assert comps["flow_alignment"].score is None


def test_confluence_score_conflicting_signals() -> None:
    inputs = ConfluenceInputs(
        weekly_trend="up",
        extension_percentile=98.0,
        rsi_divergence="bearish",
        flow_delta_oi_notional=-3_000_000.0,
        iv_rv_20d=0.7,
    )
    result = score_confluence(inputs)

    assert result.total == pytest.approx(35.71, rel=1e-3)
    assert result.coverage == pytest.approx(1.0)
