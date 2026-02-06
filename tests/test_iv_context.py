from __future__ import annotations

from options_helper.analysis.iv_context import classify_iv_rv


def test_classify_iv_rv_expensive_fair_cheap() -> None:
    expensive = classify_iv_rv(1.4, low=0.8, high=1.2)
    assert expensive is not None
    assert expensive.label == "expensive"
    assert expensive.reason == "IV/RV high (premium rich)."

    fair = classify_iv_rv(1.0, low=0.8, high=1.2)
    assert fair is not None
    assert fair.label == "fair"
    assert fair.reason == "IV/RV neutral."

    cheap = classify_iv_rv(0.7, low=0.8, high=1.2)
    assert cheap is not None
    assert cheap.label == "cheap"
    assert cheap.reason == "IV/RV low (premium cheap)."


def test_classify_iv_rv_handles_missing_and_invalid_thresholds() -> None:
    assert classify_iv_rv(None, low=0.8, high=1.2) is None
    assert classify_iv_rv(1.0, low=1.2, high=0.8) is None

