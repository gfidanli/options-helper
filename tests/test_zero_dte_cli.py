from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import market_analysis as market_analysis_command
from options_helper.schemas.zero_dte_put_study import ZeroDtePutStudyArtifact


def test_zero_dte_put_study_registered_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["market-analysis", "zero-dte-put-study", "--help"])
    assert result.exit_code == 0, result.output
    assert "--decision-mode" in result.output
    assert "--risk-tiers" in result.output
    assert "--strike-grid" in result.output


def test_zero_dte_put_study_defaults_binding_and_disclaimer(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _stub_workflow(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return market_analysis_command._ZeroDTEStudyResult(
            artifact=ZeroDtePutStudyArtifact(as_of=date(2026, 2, 6)),
            active_model={
                "schema_version": 1,
                "symbol": "SPY",
                "trained_through_session": "2026-02-06",
                "tail_model": {
                    "config": {},
                    "strike_returns": [-0.02],
                    "training_sample_size": 1,
                    "global_stats": [],
                    "parent_stats": [],
                    "bucket_stats": [],
                },
                "assumptions": ZeroDtePutStudyArtifact(as_of=date(2026, 2, 6)).assumptions.model_dump(
                    mode="json"
                ),
                "assumptions_hash": "abc123",
                "model_version": "active_2026-02-06",
            },
            preflight_passed=True,
            preflight_messages=[],
        )

    monkeypatch.setattr(
        market_analysis_command,
        "_run_zero_dte_put_study_workflow",
        _stub_workflow,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "market-analysis",
            "zero-dte-put-study",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["symbol"] == "SPY"
    assert str(captured["decision_mode"]) == "DecisionMode.FIXED_TIME"
    assert str(captured["fill_model"]) == "FillModel.BID"
    assert tuple(captured["risk_tiers"]) == (0.005, 0.01, 0.02, 0.05)
    assert tuple(captured["strike_grid"]) == (-0.03, -0.02, -0.015, -0.01, -0.005)

    assert "Proxy notice:" in result.output
    assert "Not financial advice" in result.output
    assert (tmp_path / "zero_dte_put_study" / "SPY" / "2026-02-06.json").exists()
    assert (tmp_path / "zero_dte_put_study" / "SPY" / "active_model.json").exists()


def test_zero_dte_put_study_rejects_invalid_risk_tiers() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "market-analysis",
            "zero-dte-put-study",
            "--risk-tiers",
            "0.01,1.2",
        ],
    )
    assert result.exit_code == 1
    assert "Risk tiers must be in (0, 1)." in result.output


def test_zero_dte_forward_snapshot_registered_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["market-analysis", "zero-dte-put-forward-snapshot", "--help"])
    assert result.exit_code == 0, result.output
    assert "--active-model-path" in result.output
    assert "--snapshot-path" in result.output


def test_zero_dte_forward_snapshot_idempotent_upsert_and_reconciliation(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    call_state = {"count": 0}

    def _stub_forward(**kwargs):  # type: ignore[no-untyped-def]
        call_state["count"] += 1
        finalized = call_state["count"] > 1
        rows = [
            {
                "symbol": "SPY",
                "session_date": "2026-02-10",
                "decision_ts": "2026-02-10T15:30:00+00:00",
                "risk_tier": 0.01,
                "model_version": "active_2026-02-09",
                "assumptions_hash": "hash-1",
                "reconciliation_status": "finalized" if finalized else "pending_close",
                "realized_close_return_from_entry": -0.012 if finalized else None,
            }
        ]
        payload = {
            "symbol": "SPY",
            "session_date": "2026-02-10",
            "rows": 1,
            "pending_close_rows": 0 if finalized else 1,
            "finalized_rows": 1 if finalized else 0,
            "disclaimer": market_analysis_command.clean_nan(
                market_analysis_command.ZeroDteDisclaimerMetadata().model_dump(mode="json")
            ),
        }
        return market_analysis_command._ZeroDTEForwardResult(payload=payload, rows=rows)

    monkeypatch.setattr(
        market_analysis_command,
        "_run_zero_dte_forward_snapshot_workflow",
        _stub_forward,
    )

    runner = CliRunner()
    first = runner.invoke(
        app,
        [
            "market-analysis",
            "zero-dte-put-forward-snapshot",
            "--out",
            str(tmp_path),
        ],
    )
    assert first.exit_code == 0, first.output
    assert "pending_close=1" in first.output
    assert "Proxy notice:" in first.output
    assert "Not financial advice" in first.output

    second = runner.invoke(
        app,
        [
            "market-analysis",
            "zero-dte-put-forward-snapshot",
            "--out",
            str(tmp_path),
        ],
    )
    assert second.exit_code == 0, second.output
    assert "finalized=1" in second.output

    snapshot_path = tmp_path / "zero_dte_put_study" / "SPY" / "forward_snapshots.jsonl"
    lines = [line for line in snapshot_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["reconciliation_status"] == "finalized"
    assert row["realized_close_return_from_entry"] == -0.012


def test_zero_dte_forward_snapshot_surfaces_errors(
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    def _stub_forward_error(**kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("No frozen active model found")

    monkeypatch.setattr(
        market_analysis_command,
        "_run_zero_dte_forward_snapshot_workflow",
        _stub_forward_error,
    )
    runner = CliRunner()
    result = runner.invoke(app, ["market-analysis", "zero-dte-put-forward-snapshot"])
    assert result.exit_code == 1
    assert "No frozen active model found" in result.output
