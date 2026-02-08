from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from options_helper.schemas.zero_dte_put_study import (
    DecisionAnchorMetadata,
    DecisionMode,
    ExitMode,
    FillModel,
    QuoteQualityStatus,
    ZeroDteProbabilityRow,
    ZeroDtePutStudyArtifact,
    ZeroDteSimulationRow,
    ZeroDteStrikeLadderRow,
)


def _seed_zero_dte_artifacts(root: Path, *, symbol: str = "SPY") -> None:
    symbol_dir = root / "zero_dte_put_study" / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)

    anchor_fixed = DecisionAnchorMetadata(
        session_date=date(2026, 2, 6),
        decision_ts=datetime(2026, 2, 6, 15, 30, tzinfo=timezone.utc),
        decision_bar_completed_ts=datetime(2026, 2, 6, 15, 30, tzinfo=timezone.utc),
        close_label_ts=datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc),
        entry_anchor_ts=datetime(2026, 2, 6, 15, 31, tzinfo=timezone.utc),
        decision_mode=DecisionMode.FIXED_TIME,
    )
    anchor_rolling = DecisionAnchorMetadata(
        session_date=date(2026, 2, 6),
        decision_ts=datetime(2026, 2, 6, 17, 0, tzinfo=timezone.utc),
        decision_bar_completed_ts=datetime(2026, 2, 6, 17, 0, tzinfo=timezone.utc),
        close_label_ts=datetime(2026, 2, 6, 21, 0, tzinfo=timezone.utc),
        entry_anchor_ts=datetime(2026, 2, 6, 17, 1, tzinfo=timezone.utc),
        decision_mode=DecisionMode.ROLLING,
    )

    artifact = ZeroDtePutStudyArtifact(
        as_of=date(2026, 2, 6),
        probability_rows=[
            ZeroDteProbabilityRow(
                symbol=symbol,
                risk_tier=0.01,
                strike_return=-0.01,
                breach_probability=0.12,
                breach_probability_ci_low=0.08,
                breach_probability_ci_high=0.18,
                sample_size=120,
                model_version="wf_1",
                assumptions_hash="hash-1",
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
            ),
            ZeroDteProbabilityRow(
                symbol=symbol,
                risk_tier=0.01,
                strike_return=-0.02,
                breach_probability=0.04,
                breach_probability_ci_low=0.02,
                breach_probability_ci_high=0.07,
                sample_size=120,
                model_version="wf_1",
                assumptions_hash="hash-1",
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
            ),
            ZeroDteProbabilityRow(
                symbol=symbol,
                risk_tier=0.02,
                strike_return=-0.01,
                breach_probability=0.19,
                breach_probability_ci_low=0.14,
                breach_probability_ci_high=0.24,
                sample_size=90,
                model_version="wf_1",
                assumptions_hash="hash-1",
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_rolling,
            ),
        ],
        strike_ladder_rows=[
            ZeroDteStrikeLadderRow(
                symbol=symbol,
                risk_tier=0.01,
                ladder_rank=1,
                strike_price=590.0,
                strike_return=-0.01,
                breach_probability=0.12,
                premium_estimate=1.25,
                fill_model=FillModel.BID,
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
            ),
            ZeroDteStrikeLadderRow(
                symbol=symbol,
                risk_tier=0.01,
                ladder_rank=2,
                strike_price=585.0,
                strike_return=-0.02,
                breach_probability=0.04,
                premium_estimate=0.86,
                fill_model=FillModel.BID,
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
            ),
        ],
        simulation_rows=[
            ZeroDteSimulationRow(
                symbol=symbol,
                risk_tier=0.01,
                exit_mode=ExitMode.HOLD_TO_CLOSE,
                fill_model=FillModel.BID,
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
                pnl_per_contract=0.65,
                max_loss_proxy=100.0,
                entry_premium=1.25,
                exit_premium=0.60,
            ),
            ZeroDteSimulationRow(
                symbol=symbol,
                risk_tier=0.01,
                exit_mode=ExitMode.ADAPTIVE_EXIT,
                fill_model=FillModel.BID,
                quote_quality_status=QuoteQualityStatus.GOOD,
                anchor=anchor_fixed,
                pnl_per_contract=-0.15,
                max_loss_proxy=100.0,
                entry_premium=1.25,
                exit_premium=1.40,
            ),
        ],
    )
    (symbol_dir / "2026-02-06.json").write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    (symbol_dir / "active_model.json").write_text("{}", encoding="utf-8")

    forward_rows = [
        {
            "symbol": symbol,
            "session_date": "2026-02-07",
            "decision_ts": "2026-02-07T15:30:00+00:00",
            "risk_tier": 0.01,
            "ladder_rank": 1,
            "strike_return": -0.01,
            "strike_price": 592.0,
            "breach_probability": 0.10,
            "breach_probability_ci_low": 0.07,
            "breach_probability_ci_high": 0.15,
            "premium_estimate": 1.1,
            "policy_status": "ok",
            "policy_reason": None,
            "reconciliation_status": "finalized",
            "realized_close_return_from_entry": -0.008,
            "model_version": "active_2026-02-06",
            "assumptions_hash": "hash-1",
        },
        {
            "symbol": symbol,
            "session_date": "2026-02-07",
            "decision_ts": "2026-02-07T15:30:00+00:00",
            "risk_tier": 0.01,
            "ladder_rank": 2,
            "strike_return": -0.02,
            "strike_price": 585.0,
            "breach_probability": 0.04,
            "breach_probability_ci_low": 0.02,
            "breach_probability_ci_high": 0.06,
            "premium_estimate": 0.7,
            "policy_status": "ok",
            "policy_reason": None,
            "reconciliation_status": "finalized",
            "realized_close_return_from_entry": -0.025,
            "model_version": "active_2026-02-06",
            "assumptions_hash": "hash-1",
        },
        {
            "symbol": symbol,
            "session_date": "2026-02-08",
            "decision_ts": "2026-02-08T15:30:00+00:00",
            "risk_tier": 0.02,
            "ladder_rank": 1,
            "strike_return": -0.01,
            "strike_price": 590.0,
            "breach_probability": 0.18,
            "breach_probability_ci_low": 0.14,
            "breach_probability_ci_high": 0.24,
            "premium_estimate": 1.0,
            "policy_status": "ok",
            "policy_reason": None,
            "reconciliation_status": "pending_close",
            "realized_close_return_from_entry": None,
            "model_version": "active_2026-02-06",
            "assumptions_hash": "hash-1",
        },
    ]
    with (symbol_dir / "forward_snapshots.jsonl").open("w", encoding="utf-8") as handle:
        for row in forward_rows:
            handle.write(json.dumps(row) + "\n")


def test_zero_dte_page_helpers_contracts(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import zero_dte_put_page as page

    _seed_zero_dte_artifacts(tmp_path)
    page._list_symbols_cached.clear()
    page._load_latest_study_payload.clear()
    page._load_forward_rows.clear()

    symbols, note = page.list_zero_dte_symbols(reports_root=tmp_path)
    assert note is None
    assert symbols == ["SPY"]

    probabilities, probability_notes = page.load_probability_surface("SPY", reports_root=tmp_path)
    assert probability_notes == []
    assert not probabilities.empty
    assert list(probabilities.columns) == page._PROBABILITY_COLUMNS
    assert set(probabilities["decision_time_et"]) == {"10:30", "12:00"}

    ladder, ladder_notes = page.load_strike_table(
        "SPY",
        reports_root=tmp_path,
        fill_model="bid",
    )
    assert ladder_notes == []
    assert not ladder.empty
    assert list(ladder.columns) == page._STRIKE_COLUMNS
    assert ladder["fill_model"].str.lower().eq("bid").all()

    walk_summary, walk_notes = page.load_walk_forward_summary("SPY", reports_root=tmp_path)
    assert walk_notes == []
    assert not walk_summary.empty
    assert list(walk_summary.columns) == page._WALK_FORWARD_COLUMNS
    assert set(walk_summary["exit_mode"]) == {"adaptive_exit", "hold_to_close"}

    calibration, calibration_notes = page.load_calibration_curves("SPY", reports_root=tmp_path, bins=5)
    assert not calibration.empty
    assert "forward_test" in set(calibration["source"])
    assert list(calibration.columns) == page._CALIBRATION_COLUMNS
    assert calibration_notes == []

    forward, forward_notes = page.load_forward_snapshots("SPY", reports_root=tmp_path)
    assert forward_notes == []
    assert not forward.empty
    assert list(forward.columns) == page._FORWARD_COLUMNS
    assert set(forward["reconciliation_status"]) == {"finalized", "pending_close"}


def test_zero_dte_page_helpers_missing_or_malformed_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import zero_dte_put_page as page

    missing_root = tmp_path / "missing"
    page._list_symbols_cached.clear()
    symbols, note = page.list_zero_dte_symbols(reports_root=missing_root)
    assert symbols == []
    assert note is not None
    assert "not found" in note.lower()

    probabilities, probability_notes = page.load_probability_surface("SPY", reports_root=missing_root)
    assert probabilities.empty
    assert probability_notes

    symbol_dir = tmp_path / "zero_dte_put_study" / "SPY"
    symbol_dir.mkdir(parents=True, exist_ok=True)
    (symbol_dir / "2026-02-08.json").write_text("{invalid", encoding="utf-8")

    page._load_latest_study_payload.clear()
    bad_probabilities, bad_notes = page.load_probability_surface("SPY", reports_root=tmp_path)
    assert bad_probabilities.empty
    assert any("parse" in str(item).lower() or "readable" in str(item).lower() for item in bad_notes)

    page._load_forward_rows.clear()
    forward, forward_notes = page.load_forward_snapshots("SPY", reports_root=tmp_path)
    assert forward.empty
    assert forward_notes
    assert "not found" in " ".join(forward_notes).lower()


def test_zero_dte_page_filters_apply_consistently(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import zero_dte_put_page as page

    _seed_zero_dte_artifacts(tmp_path)
    page._load_latest_study_payload.clear()
    page._load_forward_rows.clear()

    filtered, _ = page.load_probability_surface(
        "SPY",
        reports_root=tmp_path,
        decision_mode="fixed_time",
        decision_time_et="10:30",
        risk_tier=0.01,
        max_strike_distance_pct=1.1,
    )
    assert not filtered.empty
    assert filtered["decision_mode"].eq("fixed_time").all()
    assert filtered["decision_time_et"].eq("10:30").all()
    assert pd.to_numeric(filtered["risk_tier"], errors="coerce").eq(0.01).all()
    assert pd.to_numeric(filtered["strike_return"], errors="coerce").abs().le(0.011).all()

