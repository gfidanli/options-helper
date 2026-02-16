from __future__ import annotations

from pathlib import Path

from options_helper.commands.market_analysis.zero_dte_serialization import _fit_and_serialize_active_model
from options_helper.commands.market_analysis.zero_dte_types import _ZeroDTEForwardResult, _ZeroDTEStudyResult
from options_helper.commands.market_analysis.zero_dte_workflow_support import (
    _build_assumptions_with_hash,
    _build_forward_result,
    _build_intraday_store,
    _build_study_artifact,
    _load_forward_candidates,
    _load_forward_model_bundle,
    _load_study_candidate_bundle,
    _run_walk_forward_or_raise,
    _score_forward_candidates,
    _validate_preflight_or_raise,
)
from options_helper.schemas.zero_dte_put_study import DecisionMode, FillModel


def _run_zero_dte_put_study_workflow(
    *,
    symbol: str,
    start_date: str | None, end_date: str | None,
    decision_mode: DecisionMode, decision_times: tuple[str, ...],
    rolling_interval_minutes: int, risk_tiers: tuple[float, ...],
    strike_grid: tuple[float, ...], fill_model: FillModel,
    entry_slippage_bps: float, exit_slippage_bps: float,
    train_sessions: int, test_sessions: int, step_sessions: int,
    min_training_rows: int,
    top_k_per_tier: int, strict_preflight: bool,
    intraday_dir: Path, snapshot_cache_dir: Path, candle_cache_dir: Path,
) -> _ZeroDTEStudyResult:
    end, candidates, features, labels, snapshots, warnings = _load_study_candidate_bundle(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        decision_mode=decision_mode,
        decision_times=decision_times,
        rolling_interval_minutes=rolling_interval_minutes,
        risk_tiers=risk_tiers,
        strike_grid=strike_grid,
        fill_model=fill_model,
        intraday_dir=intraday_dir,
        snapshot_cache_dir=snapshot_cache_dir,
        candle_cache_dir=candle_cache_dir,
    )
    preflight_passed, preflight_messages = _validate_preflight_or_raise(
        features, labels, snapshots, strict_preflight=strict_preflight
    )
    walk_forward = _run_walk_forward_or_raise(
        candidates,
        strike_grid=strike_grid,
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
        step_sessions=step_sessions,
        min_training_rows=min_training_rows,
    )
    assumptions, assumptions_hash = _build_assumptions_with_hash(
        symbol=symbol,
        decision_mode=decision_mode,
        decision_times=decision_times,
        rolling_interval_minutes=rolling_interval_minutes,
        fill_model=fill_model,
        entry_slippage_bps=entry_slippage_bps,
        risk_tiers=risk_tiers,
    )
    artifact = _build_study_artifact(
        walk_forward=walk_forward,
        assumptions=assumptions,
        assumptions_hash=assumptions_hash,
        fill_model=fill_model,
        top_k_per_tier=top_k_per_tier,
        end=end,
        warnings=warnings,
        preflight_messages=preflight_messages,
        risk_tiers=risk_tiers,
    )
    active_model = _fit_and_serialize_active_model(
        candidates,
        symbol=symbol,
        strike_grid=strike_grid,
        assumptions=assumptions,
        assumptions_hash=assumptions_hash,
        fallback_trained_through=artifact.as_of,
    )
    return _ZeroDTEStudyResult(artifact=artifact, active_model=active_model, preflight_passed=preflight_passed, preflight_messages=preflight_messages)


def _run_zero_dte_forward_snapshot_workflow(
    *,
    symbol: str,
    session_date: str | None,
    intraday_dir: Path,
    snapshot_cache_dir: Path,
    candle_cache_dir: Path,
    out: Path,
    active_model_path: Path | None,
) -> _ZeroDTEForwardResult:
    payload, model, assumptions, trained_through, session = _load_forward_model_bundle(
        symbol=symbol,
        session_date=session_date,
        intraday_dir=intraday_dir,
        out=out,
        active_model_path=active_model_path,
    )
    candidates, warnings = _load_forward_candidates(
        symbol=symbol,
        session=session,
        model=model,
        assumptions=assumptions,
        intraday_dir=intraday_dir,
        snapshot_cache_dir=snapshot_cache_dir,
        candle_cache_dir=candle_cache_dir,
    )
    merged = _score_forward_candidates(candidates=candidates, model=model, payload=payload)
    return _build_forward_result(
        symbol=symbol,
        session=session,
        payload=payload,
        trained_through=trained_through,
        warnings=warnings,
        merged=merged,
        assumptions=assumptions,
    )


__all__ = [
    "_build_intraday_store",
    "_run_zero_dte_put_study_workflow",
    "_run_zero_dte_forward_snapshot_workflow",
]
