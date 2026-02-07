from __future__ import annotations

import pytest
import pandas as pd

from options_helper.analysis.zero_dte_calibration import (
    ZeroDTECalibrationConfig,
    compute_zero_dte_calibration,
)


def test_calibration_metrics_match_known_values() -> None:
    frame = pd.DataFrame(
        {
            "breach_probability": [0.1, 0.2, 0.8, 0.9],
            "breach_observed": [0, 0, 1, 1],
        }
    )
    result = compute_zero_dte_calibration(
        frame,
        config=ZeroDTECalibrationConfig(num_bins=2),
    )

    assert result.sample_size == 4
    assert result.brier_score == pytest.approx(0.025)
    assert result.predicted_mean == pytest.approx(0.5)
    assert result.observed_rate == pytest.approx(0.5)
    assert result.sharpness == pytest.approx(0.125)
    assert result.expected_calibration_error == pytest.approx(0.15)

    reliability = result.reliability_bins
    assert reliability["sample_size"].to_list() == [2, 2]
    assert reliability["predicted_mean"].to_list() == pytest.approx([0.15, 0.85])
    assert reliability["observed_rate"].to_list() == pytest.approx([0.0, 1.0])


def test_calibration_accepts_external_outcomes_and_drops_invalid_rows() -> None:
    predictions = pd.DataFrame({"breach_probability": [0.1, 1.2, None, 0.6]})
    outcomes = pd.Series([0, 1, 1, "true"])

    result = compute_zero_dte_calibration(
        predictions,
        outcomes=outcomes,
        config=ZeroDTECalibrationConfig(num_bins=2),
    )

    assert result.sample_size == 2
    assert result.brier_score == pytest.approx(0.085)
    assert result.reliability_bins["sample_size"].sum() == 2
    assert result.reliability_bins.shape[0] == 2

