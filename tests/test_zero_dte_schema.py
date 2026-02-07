from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from options_helper.schemas.zero_dte_put_study import (
    DecisionAnchorMetadata,
    DecisionMode,
    QuoteQualityStatus,
    SkipReason,
    ZeroDteProbabilityRow,
    ZeroDtePutStudyArtifact,
    ZeroDteSimulationRow,
    ZeroDteStrikeLadderRow,
    ZeroDteStudyAssumptions,
)


def _anchor(*, with_entry_anchor: bool = True) -> DecisionAnchorMetadata:
    return DecisionAnchorMetadata(
        session_date=date(2026, 2, 6),
        decision_ts=datetime(2026, 2, 6, 15, 30, tzinfo=timezone.utc),
        decision_bar_completed_ts=datetime(2026, 2, 6, 15, 30, tzinfo=timezone.utc),
        close_label_ts=datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc),
        decision_mode=DecisionMode.FIXED_TIME,
        entry_anchor_ts=(
            datetime(2026, 2, 6, 15, 31, tzinfo=timezone.utc) if with_entry_anchor else None
        ),
    )


def test_zero_dte_study_assumptions_locked_defaults() -> None:
    assumptions = ZeroDteStudyAssumptions()

    assert assumptions.proxy_underlying_symbol == "SPY"
    assert assumptions.target_underlying_symbol == "SPXW"
    assert assumptions.benchmark_decision_mode == DecisionMode.FIXED_TIME
    assert assumptions.benchmark_fixed_time_et == "10:30"
    assert assumptions.fill_model.value == "bid"
    assert assumptions.risk_tier_breach_probabilities == (0.005, 0.01, 0.02, 0.05)
    assert [mode.value for mode in assumptions.exit_modes] == ["hold_to_close", "adaptive_exit"]


def test_zero_dte_study_assumptions_reject_unsorted_risk_tiers() -> None:
    with pytest.raises(ValidationError):
        ZeroDteStudyAssumptions(risk_tier_breach_probabilities=(0.02, 0.01))


def test_probability_row_requires_anchor_metadata() -> None:
    with pytest.raises(ValidationError):
        ZeroDteProbabilityRow(
            symbol="SPY",
            risk_tier=0.01,
            strike_return=-0.01,
            breach_probability=0.2,
            sample_size=42,
            model_version="v1",
            assumptions_hash="abc123",
            quote_quality_status=QuoteQualityStatus.GOOD,
        )


def test_probability_row_rejects_invalid_quote_quality_status() -> None:
    with pytest.raises(ValidationError):
        ZeroDteProbabilityRow(
            symbol="SPY",
            risk_tier=0.01,
            strike_return=-0.01,
            breach_probability=0.2,
            sample_size=42,
            model_version="v1",
            assumptions_hash="abc123",
            quote_quality_status="invalid_status",
            anchor=_anchor(),
        )


def test_rows_require_no_entry_anchor_skip_reason_when_anchor_missing() -> None:
    anchor_without_entry = _anchor(with_entry_anchor=False)
    with pytest.raises(ValidationError):
        ZeroDteStrikeLadderRow(
            symbol="SPY",
            risk_tier=0.01,
            ladder_rank=1,
            strike_price=575.0,
            strike_return=-0.01,
            breach_probability=0.2,
            quote_quality_status=QuoteQualityStatus.MISSING,
            anchor=anchor_without_entry,
        )

    row = ZeroDteSimulationRow(
        symbol="SPY",
        risk_tier=0.01,
        exit_mode="hold_to_close",
        quote_quality_status=QuoteQualityStatus.MISSING,
        skip_reason=SkipReason.NO_ENTRY_ANCHOR,
        anchor=anchor_without_entry,
    )
    assert row.skip_reason == SkipReason.NO_ENTRY_ANCHOR


def test_zero_dte_artifact_json_round_trip_stable() -> None:
    artifact = ZeroDtePutStudyArtifact(
        as_of=date(2026, 2, 6),
        probability_rows=[
            ZeroDteProbabilityRow(
                symbol="SPY",
                risk_tier=0.01,
                strike_return=-0.01,
                breach_probability=0.2,
                breach_probability_ci_low=0.15,
                breach_probability_ci_high=0.25,
                sample_size=42,
                model_version="v1",
                assumptions_hash="abc123",
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=_anchor(),
            )
        ],
    )

    payload = artifact.model_dump_json()
    restored = ZeroDtePutStudyArtifact.model_validate_json(payload)
    assert restored.to_dict() == artifact.to_dict()

