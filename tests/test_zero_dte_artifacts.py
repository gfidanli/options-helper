from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from options_helper.data.zero_dte_artifacts import (
    ModelStateMissingError,
    ModelStateStaleError,
    ZeroDteArtifactStore,
    build_model_snapshot,
    compute_assumptions_hash,
    compute_snapshot_hash,
)
from options_helper.schemas.zero_dte_put_study import (
    ExitMode,
    ModelPromotionAction,
    ZeroDteBacktestSummaryRow,
    ZeroDteCalibrationTableRow,
    ZeroDteForwardProbabilityCurveRow,
    ZeroDteForwardSnapshotRow,
    ZeroDteForwardStrikeLadderRow,
    ZeroDteModelCompatibilityMetadata,
    ZeroDteTradeLedgerRow,
)


def test_snapshot_hash_is_stable_for_equivalent_payloads() -> None:
    model_payload_a = {"bucket": {"a": 1, "b": 2}, "weights": [0.2, 0.8]}
    model_payload_b = {"weights": [0.2, 0.8], "bucket": {"b": 2, "a": 1}}

    hash_a = compute_snapshot_hash(
        model_version="v1",
        trained_through_session=date(2026, 2, 6),
        assumptions_hash="abc123",
        model_payload=model_payload_a,
    )
    hash_b = compute_snapshot_hash(
        model_version="v1",
        trained_through_session=date(2026, 2, 6),
        assumptions_hash="abc123",
        model_payload=model_payload_b,
    )

    assert hash_a == hash_b



def test_active_model_resolution_promotion_and_rollback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ZeroDteArtifactStore(tmp_path)
    assumptions_hash = compute_assumptions_hash({"fixed_time": "10:30", "fill_model": "bid"})
    compatibility = ZeroDteModelCompatibilityMetadata(
        artifact_schema_version=1,
        feature_contract_version="f1",
        policy_contract_version="p1",
    )

    snap_v1 = build_model_snapshot(
        model_version="v1",
        trained_through_session=date(2026, 2, 4),
        assumptions_hash=assumptions_hash,
        model_payload={"global_rate": 0.11},
        compatibility=compatibility,
    )
    snap_v2 = build_model_snapshot(
        model_version="v2",
        trained_through_session=date(2026, 2, 5),
        assumptions_hash=assumptions_hash,
        model_payload={"global_rate": 0.14},
        compatibility=compatibility,
    )

    store.register_model_snapshot(snap_v1)
    store.register_model_snapshot(snap_v2)

    store.promote_model(
        "v1",
        as_of=date(2026, 2, 4),
        promoted_at=datetime(2026, 2, 4, 21, 0, tzinfo=timezone.utc),
        reason="baseline promotion",
    )

    resolved = store.resolve_active_model(
        scoring_session=date(2026, 2, 5),
        expected_assumptions_hash=assumptions_hash,
        required_compatibility=compatibility,
    )
    assert resolved.model_version == "v1"

    store.promote_model(
        "v2",
        as_of=date(2026, 2, 5),
        promoted_at=datetime(2026, 2, 5, 21, 0, tzinfo=timezone.utc),
        reason="newer fit",
    )

    resolved = store.resolve_active_model(
        scoring_session=date(2026, 2, 6),
        expected_assumptions_hash=assumptions_hash,
        required_compatibility=compatibility,
    )
    assert resolved.model_version == "v2"

    store.rollback_model(
        as_of=date(2026, 2, 6),
        promoted_at=datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc),
        reason="rollback regression",
    )

    resolved = store.resolve_active_model(
        scoring_session=date(2026, 2, 7),
        expected_assumptions_hash=assumptions_hash,
        required_compatibility=compatibility,
    )
    assert resolved.model_version == "v1"

    registry = store.load_model_registry()
    assert registry.active_model_version == "v1"
    assert registry.previous_active_model_version == "v2"
    assert [event.action for event in registry.promotion_history] == [
        ModelPromotionAction.PROMOTE,
        ModelPromotionAction.PROMOTE,
        ModelPromotionAction.ROLLBACK,
    ]



def test_resolve_active_model_rejects_missing_and_stale_snapshots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ZeroDteArtifactStore(tmp_path)
    assumptions_hash = compute_assumptions_hash({"fixed_time": "10:30"})

    stale_snapshot = build_model_snapshot(
        model_version="v1",
        trained_through_session=date(2026, 2, 1),
        assumptions_hash=assumptions_hash,
        model_payload={"global_rate": 0.1},
    )
    store.register_model_snapshot(stale_snapshot)
    store.promote_model("v1", as_of=date(2026, 2, 1))

    with pytest.raises(ModelStateStaleError):
        store.resolve_active_model(
            scoring_session=date(2026, 2, 8),
            min_trained_through_session=date(2026, 2, 5),
            expected_assumptions_hash=assumptions_hash,
        )

    store.model_snapshot_path("v1").unlink()
    with pytest.raises(ModelStateMissingError):
        store.resolve_active_model(
            scoring_session=date(2026, 2, 8),
            expected_assumptions_hash=assumptions_hash,
        )



def test_artifact_writers_upsert_rows_idempotently(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = ZeroDteArtifactStore(tmp_path)
    decision_ts = datetime(2026, 2, 7, 15, 30, tzinfo=timezone.utc)
    entry_ts = datetime(2026, 2, 7, 15, 31, tzinfo=timezone.utc)
    assumptions_hash = "hash-1"

    probability_row = ZeroDteForwardProbabilityCurveRow(
        symbol="SPY",
        session_date=date(2026, 2, 7),
        decision_ts=decision_ts,
        risk_tier=0.01,
        model_version="v2",
        assumptions_hash=assumptions_hash,
        strike_return=-0.01,
        breach_probability=0.12,
        sample_size=250,
    )
    strike_row = ZeroDteForwardStrikeLadderRow(
        symbol="SPY",
        session_date=date(2026, 2, 7),
        decision_ts=decision_ts,
        risk_tier=0.01,
        model_version="v2",
        assumptions_hash=assumptions_hash,
        ladder_rank=1,
        strike_price=570.0,
        strike_return=-0.01,
        breach_probability=0.12,
        premium_estimate=0.44,
    )
    calibration_row = ZeroDteCalibrationTableRow(
        model_version="v2",
        assumptions_hash=assumptions_hash,
        risk_tier=0.01,
        probability_bin="0.1-0.2",
        predicted_probability=0.12,
        observed_frequency=0.1,
        sample_size=100,
    )
    backtest_row = ZeroDteBacktestSummaryRow(
        model_version="v2",
        assumptions_hash=assumptions_hash,
        session_date=date(2026, 2, 7),
        risk_tier=0.01,
        exit_mode=ExitMode.HOLD_TO_CLOSE,
        trade_count=1,
        win_rate=1.0,
        net_pnl=45.0,
        brier_score=0.11,
    )
    snapshot_row = ZeroDteForwardSnapshotRow(
        symbol="SPY",
        session_date=date(2026, 2, 7),
        decision_ts=decision_ts,
        risk_tier=0.01,
        model_version="v2",
        assumptions_hash=assumptions_hash,
        recommended_contract_symbol="SPY260207P00570000",
        recommended_strike=570.0,
        recommended_premium=0.44,
    )
    trade_row = ZeroDteTradeLedgerRow(
        symbol="SPY",
        session_date=date(2026, 2, 7),
        decision_ts=decision_ts,
        risk_tier=0.01,
        model_version="v2",
        assumptions_hash=assumptions_hash,
        exit_mode=ExitMode.HOLD_TO_CLOSE,
        contract_symbol="SPY260207P00570000",
        entry_ts=entry_ts,
        entry_premium=0.44,
        status="open",
    )

    first_probability = store.upsert_probability_curves(as_of=date(2026, 2, 7), rows=[probability_row])
    second_probability = store.upsert_probability_curves(as_of=date(2026, 2, 7), rows=[probability_row])
    assert len(first_probability.probability_rows) == 1
    assert len(second_probability.probability_rows) == 1

    first_ladder = store.upsert_strike_ladders(as_of=date(2026, 2, 7), rows=[strike_row])
    second_ladder = store.upsert_strike_ladders(as_of=date(2026, 2, 7), rows=[strike_row])
    assert len(first_ladder.strike_ladder_rows) == 1
    assert len(second_ladder.strike_ladder_rows) == 1

    first_calibration = store.upsert_calibration_tables(as_of=date(2026, 2, 7), rows=[calibration_row])
    second_calibration = store.upsert_calibration_tables(as_of=date(2026, 2, 7), rows=[calibration_row])
    assert len(first_calibration.calibration_rows) == 1
    assert len(second_calibration.calibration_rows) == 1

    first_backtest = store.upsert_backtest_summaries(as_of=date(2026, 2, 7), rows=[backtest_row])
    second_backtest = store.upsert_backtest_summaries(as_of=date(2026, 2, 7), rows=[backtest_row])
    assert len(first_backtest.backtest_rows) == 1
    assert len(second_backtest.backtest_rows) == 1

    first_snapshots = store.upsert_forward_snapshots(as_of=date(2026, 2, 7), rows=[snapshot_row])
    second_snapshots = store.upsert_forward_snapshots(as_of=date(2026, 2, 7), rows=[snapshot_row])
    assert len(first_snapshots.snapshot_rows) == 1
    assert len(second_snapshots.snapshot_rows) == 1

    first_ledger = store.upsert_trade_ledgers(as_of=date(2026, 2, 7), rows=[trade_row])
    second_ledger = store.upsert_trade_ledgers(as_of=date(2026, 2, 7), rows=[trade_row])
    assert len(first_ledger.trade_rows) == 1
    assert len(second_ledger.trade_rows) == 1

    updated_probability = probability_row.model_copy(update={"breach_probability": 0.15})
    updated_ladder = strike_row.model_copy(update={"premium_estimate": 0.5})
    updated_calibration = calibration_row.model_copy(update={"observed_frequency": 0.14})
    updated_backtest = backtest_row.model_copy(update={"net_pnl": 52.0})
    updated_snapshot = snapshot_row.model_copy(update={"recommended_premium": 0.5})
    updated_trade = trade_row.model_copy(
        update={
            "status": "closed",
            "exit_ts": datetime(2026, 2, 7, 20, 59, tzinfo=timezone.utc),
            "exit_premium": 0.1,
            "pnl_per_contract": 34.0,
        }
    )

    assert (
        store.upsert_probability_curves(as_of=date(2026, 2, 7), rows=[updated_probability])
        .probability_rows[0]
        .breach_probability
        == pytest.approx(0.15)
    )
    assert (
        store.upsert_strike_ladders(as_of=date(2026, 2, 7), rows=[updated_ladder])
        .strike_ladder_rows[0]
        .premium_estimate
        == pytest.approx(0.5)
    )
    assert (
        store.upsert_calibration_tables(as_of=date(2026, 2, 7), rows=[updated_calibration])
        .calibration_rows[0]
        .observed_frequency
        == pytest.approx(0.14)
    )
    assert (
        store.upsert_backtest_summaries(as_of=date(2026, 2, 7), rows=[updated_backtest])
        .backtest_rows[0]
        .net_pnl
        == pytest.approx(52.0)
    )
    assert (
        store.upsert_forward_snapshots(as_of=date(2026, 2, 7), rows=[updated_snapshot])
        .snapshot_rows[0]
        .recommended_premium
        == pytest.approx(0.5)
    )
    assert (
        store.upsert_trade_ledgers(as_of=date(2026, 2, 7), rows=[updated_trade])
        .trade_rows[0]
        .status
        == "closed"
    )
