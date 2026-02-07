from __future__ import annotations

import pandas as pd

from options_helper.analysis.zero_dte_policy import (
    ZeroDTEPolicyConfig,
    recommend_zero_dte_put_strikes,
)


def _probability_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06", "2026-02-06"],
            "decision_ts": ["2026-02-06T15:30:00Z"] * 3,
            "entry_anchor_ts": ["2026-02-06T15:31:00Z"] * 3,
            "strike_return": [-0.01, -0.02, -0.03],
            "breach_probability": [0.040, 0.018, 0.008],
            "breach_probability_ci_low": [0.030, 0.012, 0.004],
            "breach_probability_ci_high": [0.060, 0.028, 0.015],
            "sample_size": [120, 120, 120],
        }
    )


def _snapshot_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06", "2026-02-06"],
            "decision_ts": ["2026-02-06T15:30:00Z"] * 3,
            "entry_anchor_ts": ["2026-02-06T15:31:00Z"] * 3,
            "target_strike_return": [-0.01, -0.02, -0.03],
            "strike_price": [99.0, 98.0, 97.0],
            "entry_premium": [2.00, 1.20, 0.80],
            "quote_quality_status": ["good", "good", "good"],
            "skip_reason": [None, None, None],
        }
    )


def test_policy_maps_tighter_risk_tiers_to_deeper_strikes() -> None:
    recommendations = recommend_zero_dte_put_strikes(
        _probability_rows(),
        _snapshot_rows(),
        risk_tiers=[0.01, 0.02, 0.05],
        config=ZeroDTEPolicyConfig(top_k_per_tier=1),
    )

    assert recommendations["policy_status"].eq("ok").all()
    assert recommendations["fallback_used"].eq(False).all()
    by_tier = {row["risk_tier"]: row["strike_return"] for _, row in recommendations.iterrows()}
    assert by_tier[0.01] == -0.03
    assert by_tier[0.02] == -0.02
    assert by_tier[0.05] == -0.01


def test_policy_emits_fallback_when_no_strike_meets_tier() -> None:
    recommendations = recommend_zero_dte_put_strikes(
        _probability_rows(),
        _snapshot_rows(),
        risk_tiers=[0.005],
        config=ZeroDTEPolicyConfig(top_k_per_tier=1),
    )

    row = recommendations.iloc[0]
    assert row["policy_status"] == "fallback"
    assert row["policy_reason"] == "fallback_no_candidate_within_risk_tier"
    assert bool(row["fallback_used"]) is True
    assert row["strike_return"] == -0.03


def test_policy_emits_deterministic_skip_reason_for_bad_quotes() -> None:
    bad_quotes = _snapshot_rows().copy()
    bad_quotes["quote_quality_status"] = ["wide", "wide", "stale"]
    bad_quotes["skip_reason"] = ["bad_quote_quality", "bad_quote_quality", "bad_quote_quality"]

    recommendations = recommend_zero_dte_put_strikes(
        _probability_rows(),
        bad_quotes,
        risk_tiers=[0.02],
        config=ZeroDTEPolicyConfig(top_k_per_tier=1),
    )

    row = recommendations.iloc[0]
    assert row["policy_status"] == "skip"
    assert row["policy_reason"] == "bad_quote_quality"
    assert pd.isna(row["strike_return"])
