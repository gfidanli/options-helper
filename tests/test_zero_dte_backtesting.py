from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.backtesting.zero_dte_put import (
    ZeroDTEPutSimulatorConfig,
    simulate_zero_dte_put_outcomes,
)
from options_helper.backtesting.zero_dte_walk_forward import (
    ZeroDTEWalkForwardConfig,
    run_zero_dte_walk_forward,
)


def test_simulator_hold_to_close_scenarios() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(
                session=date(2026, 2, 6),
                decision_ts="2026-02-06T15:30:00Z",
                entry_anchor_ts="2026-02-06T15:31:00Z",
                close_label_ts="2026-02-06T21:00:00Z",
                risk_tier=0.005,
                strike_return=-0.01,
                strike_price=100.0,
                premium=2.0,
                close_price=103.0,
            ),
            _candidate_row(
                session=date(2026, 2, 6),
                decision_ts="2026-02-06T15:31:00Z",
                entry_anchor_ts="2026-02-06T15:32:00Z",
                close_label_ts="2026-02-06T21:00:00Z",
                risk_tier=0.01,
                strike_return=-0.01,
                strike_price=100.0,
                premium=2.0,
                close_price=99.0,
            ),
            _candidate_row(
                session=date(2026, 2, 6),
                decision_ts="2026-02-06T15:32:00Z",
                entry_anchor_ts="2026-02-06T15:33:00Z",
                close_label_ts="2026-02-06T21:00:00Z",
                risk_tier=0.02,
                strike_return=-0.01,
                strike_price=100.0,
                premium=2.0,
                close_price=97.0,
            ),
            _candidate_row(
                session=date(2026, 2, 6),
                decision_ts="2026-02-06T15:33:00Z",
                entry_anchor_ts="2026-02-06T15:34:00Z",
                close_label_ts="2026-02-06T21:00:00Z",
                risk_tier=0.05,
                strike_return=-0.01,
                strike_price=100.0,
                premium=2.0,
                close_price=90.0,
            ),
        ]
    )

    out = simulate_zero_dte_put_outcomes(
        candidates,
        config=ZeroDTEPutSimulatorConfig(
            sizing_mode="fixed_contracts",
            fixed_contracts=1,
            max_concurrent_positions=10,
            max_concurrent_positions_per_day=10,
        ),
        exit_modes=("hold_to_close",),
    )

    assert len(out) == 4
    assert out["status"].eq("filled").all()
    assert out["exit_reason"].eq("hold_to_close").all()

    by_tier = out.set_index("risk_tier")
    assert by_tier.loc[0.005, "close_intrinsic"] == pytest.approx(0.0)
    assert by_tier.loc[0.005, "pnl_total"] == pytest.approx(200.0)
    assert by_tier.loc[0.01, "close_intrinsic"] == pytest.approx(1.0)
    assert by_tier.loc[0.01, "pnl_total"] == pytest.approx(100.0)
    assert by_tier.loc[0.02, "close_intrinsic"] == pytest.approx(3.0)
    assert by_tier.loc[0.02, "pnl_total"] == pytest.approx(-100.0)
    assert by_tier.loc[0.05, "close_intrinsic"] == pytest.approx(10.0)
    assert by_tier.loc[0.05, "pnl_total"] == pytest.approx(-800.0)


def test_simulator_applies_fee_slippage_and_risk_sizing() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(
                session=date(2026, 2, 9),
                decision_ts="2026-02-09T15:30:00Z",
                entry_anchor_ts="2026-02-09T15:31:00Z",
                close_label_ts="2026-02-09T21:00:00Z",
                risk_tier=0.02,
                strike_return=-0.02,
                strike_price=20.0,
                premium=1.0,
                close_price=21.0,
            )
        ]
    )

    out = simulate_zero_dte_put_outcomes(
        candidates,
        config=ZeroDTEPutSimulatorConfig(
            initial_equity=10000.0,
            sizing_mode="risk_pct_of_equity",
            risk_pct_of_equity=0.2,
            entry_slippage_bps=100.0,
            exit_slippage_bps=50.0,
            entry_fee_per_contract=1.0,
            exit_fee_per_contract=1.0,
            max_concurrent_positions=1,
            max_concurrent_positions_per_day=1,
        ),
        exit_modes=("hold_to_close",),
    )

    row = out.iloc[0]
    assert row["status"] == "filled"
    assert row["quantity"] == 1
    assert row["entry_premium_net"] == pytest.approx(0.99)
    assert row["exit_premium_net"] == pytest.approx(0.0)
    assert row["pnl_per_contract"] == pytest.approx(97.0)
    assert row["pnl_total"] == pytest.approx(97.0)


def test_simulator_enforces_same_day_concurrency_by_exit_track() -> None:
    candidates = pd.DataFrame(
        [
            _candidate_row(
                session=date(2026, 2, 10),
                decision_ts="2026-02-10T15:30:00Z",
                entry_anchor_ts="2026-02-10T15:31:00Z",
                close_label_ts="2026-02-10T21:00:00Z",
                risk_tier=0.01,
                strike_return=-0.02,
                strike_price=100.0,
                premium=1.5,
                close_price=102.0,
                adaptive_exit_premium=0.3,
                adaptive_exit_ts="2026-02-10T15:31:30Z",
            ),
            _candidate_row(
                session=date(2026, 2, 10),
                decision_ts="2026-02-10T15:32:00Z",
                entry_anchor_ts="2026-02-10T15:32:30Z",
                close_label_ts="2026-02-10T21:00:00Z",
                risk_tier=0.02,
                strike_return=-0.02,
                strike_price=100.0,
                premium=1.5,
                close_price=101.0,
            ),
            _candidate_row(
                session=date(2026, 2, 10),
                decision_ts="2026-02-10T15:33:00Z",
                entry_anchor_ts="2026-02-10T15:33:30Z",
                close_label_ts="2026-02-10T21:00:00Z",
                risk_tier=0.05,
                strike_return=-0.02,
                strike_price=100.0,
                premium=1.5,
                close_price=99.0,
            ),
        ]
    )

    out = simulate_zero_dte_put_outcomes(
        candidates,
        config=ZeroDTEPutSimulatorConfig(
            sizing_mode="fixed_contracts",
            fixed_contracts=1,
            max_concurrent_positions=1,
            max_concurrent_positions_per_day=1,
        ),
    )

    hold = out.loc[out["exit_mode"] == "hold_to_close"].sort_values(by="risk_tier")
    adaptive = out.loc[out["exit_mode"] == "adaptive_exit"].sort_values(by="risk_tier")

    assert hold["status"].to_list() == ["filled", "skipped", "skipped"]
    assert hold["skip_reason"].to_list()[1:] == ["concurrency_cap_total", "concurrency_cap_total"]

    assert adaptive["status"].to_list() == ["filled", "filled", "skipped"]
    assert adaptive["skip_reason"].to_list()[2] == "concurrency_cap_total"


def test_walk_forward_split_boundaries_and_no_future_leakage() -> None:
    rows = _walk_forward_rows()
    cfg = ZeroDTEWalkForwardConfig(
        train_sessions=3,
        test_sessions=2,
        step_sessions=2,
        min_training_rows=3,
        strike_returns=(-0.03, -0.02),
        calibration_bins=3,
        simulator_config=ZeroDTEPutSimulatorConfig(
            sizing_mode="fixed_contracts",
            fixed_contracts=1,
            max_concurrent_positions=10,
            max_concurrent_positions_per_day=10,
        ),
    )

    result = run_zero_dte_walk_forward(rows, config=cfg)

    assert len(result.folds) == 2
    assert [fold["test_sessions"] for fold in result.folds] == [
        [date(2026, 1, 8), date(2026, 1, 9)],
        [date(2026, 1, 12), date(2026, 1, 13)],
    ]
    assert not result.model_snapshots.empty
    assert (result.model_snapshots["trained_through_session"] < result.model_snapshots["session_date"]).all()
    assert result.model_snapshots.iloc[0]["trained_through_session"] == date(2026, 1, 7)
    assert result.model_snapshots.iloc[1]["trained_through_session"] == date(2026, 1, 8)

    assert not result.calibration_summary.empty
    assert not result.trade_summary.empty
    assert set(result.trade_summary["exit_mode"]) == {"hold_to_close", "adaptive_exit"}


def test_walk_forward_reproducible_with_stable_outputs() -> None:
    rows = _walk_forward_rows()
    shuffled = rows.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cfg = ZeroDTEWalkForwardConfig(
        train_sessions=3,
        test_sessions=2,
        step_sessions=2,
        min_training_rows=3,
        strike_returns=(-0.03, -0.02),
        calibration_bins=3,
    )

    first = run_zero_dte_walk_forward(rows, config=cfg)
    second = run_zero_dte_walk_forward(shuffled, config=cfg)

    first_scored = first.scored_rows.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "strike_return"],
        kind="mergesort",
    ).reset_index(drop=True)
    second_scored = second.scored_rows.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "strike_return"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_scored, second_scored)

    first_summary = first.trade_summary.sort_values(
        by=["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime", "exit_mode"],
        kind="mergesort",
    ).reset_index(drop=True)
    second_summary = second.trade_summary.sort_values(
        by=["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime", "exit_mode"],
        kind="mergesort",
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(first_summary, second_summary)


def test_walk_forward_filters_non_causal_entry_anchors() -> None:
    rows = _walk_forward_rows().copy()
    rows["entry_anchor_ts"] = rows["decision_ts"]
    cfg = ZeroDTEWalkForwardConfig(
        train_sessions=3,
        test_sessions=2,
        step_sessions=2,
        min_training_rows=1,
        strike_returns=(-0.03, -0.02),
        calibration_bins=3,
    )

    result = run_zero_dte_walk_forward(rows, config=cfg)

    assert result.scored_rows.empty
    assert result.model_snapshots.empty
    assert all(int(fold["scored_rows"]) == 0 for fold in result.folds)


def _candidate_row(
    *,
    session: date,
    decision_ts: str,
    entry_anchor_ts: str,
    close_label_ts: str,
    risk_tier: float,
    strike_return: float,
    strike_price: float,
    premium: float,
    close_price: float,
    adaptive_exit_premium: float | None = None,
    adaptive_exit_ts: str | None = None,
) -> dict[str, object]:
    return {
        "session_date": session.isoformat(),
        "decision_ts": decision_ts,
        "entry_anchor_ts": entry_anchor_ts,
        "close_label_ts": close_label_ts,
        "risk_tier": risk_tier,
        "decision_mode": "fixed_time",
        "time_of_day_bucket": "open",
        "iv_regime": "low",
        "strike_return": strike_return,
        "strike_price": strike_price,
        "premium_estimate": premium,
        "quote_quality_status": "good",
        "policy_status": "ok",
        "policy_reason": None,
        "close_price": close_price,
        "entry_anchor_price": 100.0,
        "close_return_from_entry": (close_price / 100.0) - 1.0,
        "intraday_min_return_from_entry": -0.02,
        "intraday_max_return_from_entry": 0.01,
        "adaptive_exit_premium": adaptive_exit_premium,
        "adaptive_exit_ts": adaptive_exit_ts,
    }


def _walk_forward_rows() -> pd.DataFrame:
    sessions = pd.date_range("2026-01-05", periods=8, freq="B")
    close_returns = [-0.006, -0.014, -0.022, -0.009, -0.031, -0.011, -0.019, -0.013]
    rows: list[dict[str, object]] = []
    for idx, session in enumerate(sessions):
        session_date = session.date()
        decision_ts = f"{session_date.isoformat()}T15:30:00Z"
        entry_anchor_ts = f"{session_date.isoformat()}T15:31:00Z"
        close_label_ts = f"{session_date.isoformat()}T21:00:00Z"
        close_price = 100.0 * (1.0 + close_returns[idx])
        for risk_tier, strike_return, strike_price, premium in (
            (0.02, -0.02, 98.0, 1.40),
            (0.01, -0.03, 97.0, 0.95),
        ):
            rows.append(
                {
                    "session_date": session_date.isoformat(),
                    "decision_ts": decision_ts,
                    "entry_anchor_ts": entry_anchor_ts,
                    "close_label_ts": close_label_ts,
                    "risk_tier": risk_tier,
                    "decision_mode": "fixed_time",
                    "time_of_day_bucket": "open" if idx % 2 == 0 else "midday",
                    "iv_regime": "high" if idx % 3 == 0 else "low",
                    "intraday_return": close_returns[idx] * 0.5,
                    "strike_return": strike_return,
                    "strike_price": strike_price,
                    "premium_estimate": premium,
                    "quote_quality_status": "good",
                    "policy_status": "ok",
                    "policy_reason": None,
                    "feature_status": "ok",
                    "label_status": "ok",
                    "entry_anchor_price": 100.0,
                    "close_price": close_price,
                    "close_return_from_entry": close_returns[idx],
                    "intraday_min_return_from_entry": close_returns[idx] - 0.004,
                    "intraday_max_return_from_entry": 0.006 if idx % 2 == 0 else 0.002,
                }
            )
    return pd.DataFrame(rows)
