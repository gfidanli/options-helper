from __future__ import annotations

from options_helper.analysis.scanner_rank import ScannerRankInputs, rank_scanner_candidate


def test_rank_scanner_extension_only_low_tail() -> None:
    cfg = {
        "weights": {
            "extension": 10.0,
            "weekly_trend": 0.0,
            "rsi_divergence": 0.0,
            "liquidity": 0.0,
            "iv_regime": 0.0,
            "flow": 0.0,
        },
        "extension": {"tail_low": 5.0, "tail_high": 95.0},
        "top_reasons": 3,
    }
    result = rank_scanner_candidate(ScannerRankInputs(extension_percentile=1.0), cfg)
    assert result.score > 50.0
    assert result.coverage == 1.0
    assert result.top_reasons
    assert "Extension tail" in result.top_reasons[0]


def test_rank_scanner_missing_inputs_is_neutral() -> None:
    cfg = {
        "weights": {
            "extension": 10.0,
            "weekly_trend": 0.0,
            "rsi_divergence": 0.0,
            "liquidity": 0.0,
            "iv_regime": 0.0,
            "flow": 0.0,
        },
        "extension": {"tail_low": 5.0, "tail_high": 95.0},
    }
    result = rank_scanner_candidate(ScannerRankInputs(), cfg)
    assert result.score == 50.0
    assert result.coverage == 0.0
    assert "no_components_scored" in result.warnings
    assert result.top_reasons == []
