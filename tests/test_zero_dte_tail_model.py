from __future__ import annotations

import pandas as pd

from options_helper.analysis.zero_dte_tail_model import (
    ZeroDTETailModelConfig,
    fit_zero_dte_tail_model,
    score_zero_dte_tail_model,
)


def _training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": [
                "2026-02-01",
                "2026-02-01",
                "2026-02-01",
                "2026-02-02",
                "2026-02-02",
                "2026-02-02",
                "2026-02-03",
            ],
            "time_of_day_bucket": ["open", "open", "open", "open", "open", "open", "midday"],
            "intraday_return": [-0.015, 0.005, 0.006, 0.004, 0.008, 0.009, -0.011],
            "iv_regime": ["low", "low", "low", "low", "low", "low", "high"],
            "close_return_from_entry": [-0.030, -0.005, -0.006, -0.002, -0.008, -0.009, -0.012],
            "feature_status": ["ok", "ok", "ok", "ok", "ok", "ok", "ok"],
            "label_status": ["ok", "ok", "ok", "ok", "ok", "ok", "ok"],
        }
    )


def test_tail_model_scores_monotonic_probabilities_across_deeper_strikes() -> None:
    model = fit_zero_dte_tail_model(
        _training_rows(),
        strike_returns=[-0.005, -0.01, -0.02],
        config=ZeroDTETailModelConfig(min_bucket_samples=2),
    )
    state = pd.DataFrame(
        {
            "session_date": ["2026-02-06"],
            "decision_ts": ["2026-02-06T15:30:00Z"],
            "time_of_day_bucket": ["open"],
            "intraday_return": [-0.014],
            "iv_regime": ["low"],
        }
    )

    first = score_zero_dte_tail_model(model, state)
    second = score_zero_dte_tail_model(model, state)

    assert first.equals(second)
    ordered = first.sort_values(by="strike_return", ascending=True, kind="mergesort")
    probs = ordered["breach_probability"].to_list()
    assert probs[0] <= probs[1] <= probs[2]
    assert (ordered["breach_probability_ci_low"] <= ordered["breach_probability"]).all()
    assert (ordered["breach_probability"] <= ordered["breach_probability_ci_high"]).all()


def test_tail_model_exposes_low_sample_and_fallback_diagnostics() -> None:
    model = fit_zero_dte_tail_model(
        _training_rows(),
        strike_returns=[-0.01],
        config=ZeroDTETailModelConfig(min_bucket_samples=3),
    )
    state = pd.DataFrame(
        {
            "session_date": ["2026-02-06", "2026-02-06", "2026-02-06"],
            "decision_ts": [
                "2026-02-06T15:30:00Z",
                "2026-02-06T15:30:00Z",
                "2026-02-06T15:30:00Z",
            ],
            "time_of_day_bucket": ["open", "open", "late"],
            "intraday_return": [-0.015, -0.025, -0.015],
            "iv_regime": ["low", "low", "unknown"],
        }
    )

    scored = score_zero_dte_tail_model(model, state)
    first, second, third = scored.iloc[0], scored.iloc[1], scored.iloc[2]

    assert first["sample_size"] == 1
    assert bool(first["low_sample_bin"]) is True
    assert first["fallback_source"] == "shrunk_local"

    assert second["sample_size"] == 0
    assert second["parent_sample_size"] > 0
    assert second["fallback_source"] == "parent"

    assert third["sample_size"] == 0
    assert third["parent_sample_size"] == 0
    assert third["fallback_source"] == "global"
