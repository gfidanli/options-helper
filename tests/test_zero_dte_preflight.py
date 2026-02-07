from __future__ import annotations

import pandas as pd

from options_helper.analysis.zero_dte_preflight import (
    ZeroDTEPreflightConfig,
    run_zero_dte_preflight,
)


def test_run_zero_dte_preflight_reports_failures_with_clear_diagnostics() -> None:
    features = pd.DataFrame(
        {
            "session_date": ["2026-02-03", "2026-02-03", "2026-02-03", "2026-02-03"],
            "time_of_day_bucket": ["open", "open", "midday", "midday"],
            "iv_regime": ["low", "low", "high", "high"],
            "feature_status": ["ok", "ok", "ok", "ok"],
        }
    )
    labels = pd.DataFrame(
        {
            "close_return_from_entry": [0.01, None, None, 0.02],
        }
    )
    strike_snapshot = pd.DataFrame(
        {
            "quote_quality_status": ["good", "stale", "good", "wide"],
            "entry_premium": [1.0, 1.1, 0.0, 1.2],
            "skip_reason": [None, "bad_quote_quality", None, "bad_quote_quality"],
        }
    )

    result = run_zero_dte_preflight(
        features,
        labels,
        strike_snapshot,
        config=ZeroDTEPreflightConfig(
            min_sessions=2,
            min_feature_rows=5,
            min_rows_per_time_bucket=3,
            min_rows_per_iv_regime=3,
            min_label_coverage_rate=0.75,
            min_quote_quality_pass_rate=0.8,
        ),
    )

    assert result.passed is False
    messages = {diagnostic.code: diagnostic.message for diagnostic in result.diagnostics if not diagnostic.ok}
    assert "sessions" in messages
    assert "feature_rows" in messages
    assert "time_bucket_open" in messages
    assert "iv_regime_high" in messages
    assert "label_coverage_rate" in messages
    assert "quote_quality_pass_rate" in messages


def test_run_zero_dte_preflight_passes_when_thresholds_met() -> None:
    features = pd.DataFrame(
        {
            "session_date": [
                "2026-02-03",
                "2026-02-03",
                "2026-02-04",
                "2026-02-04",
                "2026-02-05",
                "2026-02-05",
            ],
            "time_of_day_bucket": ["open", "midday", "open", "midday", "open", "midday"],
            "iv_regime": ["low", "low", "high", "high", "high", "high"],
            "feature_status": ["ok", "ok", "ok", "ok", "ok", "ok"],
        }
    )
    labels = pd.DataFrame(
        {
            "close_return_from_entry": [0.01, 0.02, 0.01, 0.0, -0.01, 0.02],
        }
    )
    strike_snapshot = pd.DataFrame(
        {
            "quote_quality_status": ["good", "unknown", "good", "good", "unknown", "good"],
            "entry_premium": [1.0, 1.1, 0.9, 1.2, 0.8, 1.0],
            "skip_reason": [None, None, None, None, None, None],
        }
    )

    result = run_zero_dte_preflight(
        features,
        labels,
        strike_snapshot,
        config=ZeroDTEPreflightConfig(
            min_sessions=3,
            min_feature_rows=6,
            min_rows_per_time_bucket=3,
            min_rows_per_iv_regime=2,
            min_label_coverage_rate=0.95,
            min_quote_quality_pass_rate=0.95,
        ),
    )

    assert result.passed is True
    assert all(diagnostic.ok for diagnostic in result.diagnostics)
    assert result.metrics["feature_rows"] == 6.0
    assert result.metrics["session_count"] == 3.0


def test_run_zero_dte_preflight_handles_missing_quote_snapshot() -> None:
    features = pd.DataFrame(
        {
            "session_date": ["2026-02-03"],
            "time_of_day_bucket": ["open"],
            "feature_status": ["ok"],
        }
    )
    labels = pd.DataFrame(
        {
            "close_return_from_entry": [0.01],
        }
    )

    result = run_zero_dte_preflight(
        features,
        labels,
        strike_snapshot=pd.DataFrame(),
        config=ZeroDTEPreflightConfig(
            min_sessions=1,
            min_feature_rows=1,
            min_rows_per_time_bucket=1,
            min_rows_per_iv_regime=1,
            min_label_coverage_rate=0.5,
            min_quote_quality_pass_rate=0.5,
        ),
    )

    failed_codes = {diagnostic.code for diagnostic in result.diagnostics if not diagnostic.ok}
    assert failed_codes == {"quote_quality_pass_rate"}
