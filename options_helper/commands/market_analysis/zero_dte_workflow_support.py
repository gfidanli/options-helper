from __future__ import annotations

from datetime import date
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import options_helper.cli_deps as cli_deps
from options_helper.analysis.zero_dte_policy import recommend_zero_dte_put_strikes
from options_helper.analysis.zero_dte_preflight import run_zero_dte_preflight
from options_helper.analysis.zero_dte_tail_model import score_zero_dte_tail_model
from options_helper.backtesting.zero_dte_put import ZeroDTEPutSimulatorConfig
from options_helper.backtesting.zero_dte_walk_forward import ZeroDTEWalkForwardConfig, run_zero_dte_walk_forward
from options_helper.commands.common import _parse_date
from options_helper.commands.market_analysis.core_helpers import _dedupe
from options_helper.commands.market_analysis.core_io import _normalize_daily_history
from options_helper.commands.market_analysis.zero_dte_candidates import (
    _build_zero_dte_candidates,
    _resolve_latest_intraday_session,
    _resolve_zero_dte_study_range,
)
from options_helper.commands.market_analysis.zero_dte_output import _default_zero_dte_active_model_path
from options_helper.commands.market_analysis.zero_dte_serialization import (
    _build_forward_snapshot_rows,
    _build_zero_dte_probability_rows,
    _build_zero_dte_simulation_rows,
    _build_zero_dte_strike_ladder_rows,
    _deserialize_tail_model,
)
from options_helper.commands.market_analysis.zero_dte_types import _ZeroDTEForwardResult
from options_helper.commands.market_analysis.zero_dte_utils import _hash_zero_dte_assumptions, _resolve_as_of_date
from options_helper.data.zero_dte_dataset import ZeroDTEIntradayDatasetLoader
from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import (
    DecisionMode,
    FillModel,
    ZeroDteDisclaimerMetadata,
    ZeroDtePutStudyArtifact,
    ZeroDteStudyAssumptions,
)


def _build_intraday_store(root_dir: Path) -> Any:
    market_analysis_pkg = importlib.import_module("options_helper.commands.market_analysis")
    return market_analysis_pkg.IntradayStore(root_dir)


def _load_study_candidate_bundle(
    *,
    symbol: str,
    start_date: str | None,
    end_date: str | None,
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
    risk_tiers: tuple[float, ...],
    strike_grid: tuple[float, ...],
    fill_model: FillModel,
    intraday_dir: Path,
    snapshot_cache_dir: Path,
    candle_cache_dir: Path,
) -> tuple[date, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    start, end = _resolve_zero_dte_study_range(
        start_date=start_date,
        end_date=end_date,
        intraday_dir=intraday_dir,
        symbol=symbol,
    )
    sessions = [day.date() for day in pd.date_range(start, end, freq="B")]
    if not sessions:
        raise ValueError("No sessions in requested date range.")

    loader = ZeroDTEIntradayDatasetLoader(
        intraday_store=_build_intraday_store(intraday_dir),
        options_snapshot_store=cli_deps.build_snapshot_store(snapshot_cache_dir),
    )
    candles = _normalize_daily_history(cli_deps.build_candle_store(candle_cache_dir).load(symbol))
    candidates, features, labels, snapshots, warnings = _build_zero_dte_candidates(
        loader=loader,
        candles=candles,
        symbol=symbol,
        sessions=sessions,
        decision_mode=decision_mode,
        decision_times=decision_times,
        rolling_interval_minutes=rolling_interval_minutes,
        strike_grid=strike_grid,
        risk_tiers=risk_tiers,
        fill_model=fill_model,
    )
    if candidates.empty:
        raise ValueError("No 0DTE candidate rows were produced for the requested range.")
    return end, candidates, features, labels, snapshots, warnings


def _validate_preflight_or_raise(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    strict_preflight: bool,
) -> tuple[bool, list[str]]:
    preflight = run_zero_dte_preflight(features, labels, snapshots)
    preflight_messages = [item.message for item in preflight.diagnostics if not item.ok]
    if strict_preflight and not preflight.passed:
        raise ValueError("Preflight failed: " + ("; ".join(preflight_messages) or "unknown reason"))
    return preflight.passed, preflight_messages


def _run_walk_forward_or_raise(
    candidates: pd.DataFrame,
    *,
    strike_grid: tuple[float, ...],
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    train_sessions: int,
    test_sessions: int,
    step_sessions: int,
    min_training_rows: int,
) -> Any:
    simulator_cfg = ZeroDTEPutSimulatorConfig(
        entry_slippage_bps=entry_slippage_bps,
        exit_slippage_bps=exit_slippage_bps,
    )
    walk_forward = run_zero_dte_walk_forward(
        candidates,
        config=ZeroDTEWalkForwardConfig(
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            step_sessions=step_sessions,
            min_training_rows=min_training_rows,
            strike_returns=strike_grid,
            simulator_config=simulator_cfg,
        ),
    )
    if walk_forward.scored_rows.empty:
        raise ValueError(
            "Walk-forward produced no scored rows. Expand date range or relax train/test settings."
        )
    return walk_forward


def _build_assumptions_with_hash(
    *,
    symbol: str,
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
    fill_model: FillModel,
    entry_slippage_bps: float,
    risk_tiers: tuple[float, ...],
) -> tuple[ZeroDteStudyAssumptions, str]:
    assumptions = ZeroDteStudyAssumptions(
        proxy_underlying_symbol=symbol,
        benchmark_decision_mode=decision_mode,
        benchmark_fixed_time_et=decision_times[0],
        rolling_interval_minutes=rolling_interval_minutes,
        fill_model=fill_model,
        fill_slippage_bps=entry_slippage_bps,
        risk_tier_breach_probabilities=risk_tiers,
    )
    return assumptions, _hash_zero_dte_assumptions(assumptions)


def _build_study_artifact(
    *,
    walk_forward: Any,
    assumptions: ZeroDteStudyAssumptions,
    assumptions_hash: str,
    fill_model: FillModel,
    top_k_per_tier: int,
    end: date,
    warnings: list[str],
    preflight_messages: list[str],
    risk_tiers: tuple[float, ...],
) -> ZeroDtePutStudyArtifact:
    recommendations = recommend_zero_dte_put_strikes(
        walk_forward.scored_rows,
        walk_forward.scored_rows,
        risk_tiers=risk_tiers,
    )
    return ZeroDtePutStudyArtifact(
        as_of=_resolve_as_of_date(walk_forward.scored_rows, default=end),
        assumptions=assumptions,
        disclaimer=ZeroDteDisclaimerMetadata(),
        probability_rows=_build_zero_dte_probability_rows(
            walk_forward.scored_rows,
            assumptions_hash=assumptions_hash,
        ),
        strike_ladder_rows=_build_zero_dte_strike_ladder_rows(
            recommendations,
            fill_model=fill_model,
            top_k_per_tier=top_k_per_tier,
        ),
        simulation_rows=_build_zero_dte_simulation_rows(
            walk_forward.trade_rows,
            fill_model=fill_model,
        ),
        warnings=_dedupe([*warnings, *preflight_messages]),
    )


def _load_forward_model_bundle(
    *,
    symbol: str,
    session_date: str | None,
    intraday_dir: Path,
    out: Path,
    active_model_path: Path | None,
) -> tuple[dict[str, Any], Any, ZeroDteStudyAssumptions, date, date]:
    model_path = active_model_path or _default_zero_dte_active_model_path(out, symbol)
    if not model_path.exists():
        raise ValueError(f"No frozen active model found at {model_path}.")
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    tail_payload = payload.get("tail_model") or {}
    if not isinstance(tail_payload, dict):
        raise ValueError("Active model payload missing tail_model section.")
    model = _deserialize_tail_model(tail_payload)
    assumptions = ZeroDteStudyAssumptions.model_validate(payload.get("assumptions") or {})
    trained_through = _parse_date(str(payload.get("trained_through_session")))
    session = _parse_date(session_date) if session_date else _resolve_latest_intraday_session(intraday_dir, symbol)
    if session <= trained_through:
        raise ValueError("Forward snapshot requires session_date strictly after trained_through_session.")
    return payload, model, assumptions, trained_through, session


def _load_forward_candidates(
    *,
    symbol: str,
    session: date,
    model: Any,
    assumptions: ZeroDteStudyAssumptions,
    intraday_dir: Path,
    snapshot_cache_dir: Path,
    candle_cache_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    loader = ZeroDTEIntradayDatasetLoader(
        intraday_store=_build_intraday_store(intraday_dir),
        options_snapshot_store=cli_deps.build_snapshot_store(snapshot_cache_dir),
    )
    candles = _normalize_daily_history(cli_deps.build_candle_store(candle_cache_dir).load(symbol))
    candidates, _, _, _, warnings = _build_zero_dte_candidates(
        loader=loader,
        candles=candles,
        symbol=symbol,
        sessions=[session],
        decision_mode=assumptions.benchmark_decision_mode,
        decision_times=(assumptions.benchmark_fixed_time_et,),
        rolling_interval_minutes=assumptions.rolling_interval_minutes,
        strike_grid=tuple(float(v) for v in model.strike_returns),
        risk_tiers=tuple(float(v) for v in assumptions.risk_tier_breach_probabilities),
        fill_model=assumptions.fill_model,
    )
    if candidates.empty:
        raise ValueError("No forward candidate rows produced for requested session.")
    return candidates, warnings


def _score_forward_candidates(
    *,
    candidates: pd.DataFrame,
    model: Any,
    payload: dict[str, Any],
) -> pd.DataFrame:
    state_rows = candidates.loc[
        :,
        ["session_date", "decision_ts", "time_of_day_bucket", "intraday_return", "iv_regime"],
    ].drop_duplicates(subset=["session_date", "decision_ts"], keep="first")
    scored = score_zero_dte_tail_model(model, state_rows, strike_returns=model.strike_returns)
    merged = candidates.merge(
        scored.loc[
            :,
            [
                "session_date",
                "decision_ts",
                "strike_return",
                "breach_probability",
                "breach_probability_ci_low",
                "breach_probability_ci_high",
                "sample_size",
            ],
        ],
        on=["session_date", "decision_ts", "strike_return"],
        how="left",
        sort=True,
    )
    merged["model_version"] = str(payload.get("model_version") or "active")
    merged["assumptions_hash"] = str(payload.get("assumptions_hash") or "")
    return merged


def _build_forward_result(
    *,
    symbol: str,
    session: date,
    payload: dict[str, Any],
    trained_through: date,
    warnings: list[str],
    merged: pd.DataFrame,
    assumptions: ZeroDteStudyAssumptions,
) -> _ZeroDTEForwardResult:
    recommendations = recommend_zero_dte_put_strikes(
        merged,
        merged,
        risk_tiers=tuple(float(v) for v in assumptions.risk_tier_breach_probabilities),
    )
    rows = _build_forward_snapshot_rows(
        recommendations=recommendations,
        scored_rows=merged,
        symbol=symbol,
        model_version=str(payload.get("model_version") or "active"),
        assumptions_hash=str(payload.get("assumptions_hash") or ""),
        trained_through_session=trained_through.isoformat(),
    )
    summary = {
        "symbol": symbol,
        "session_date": session.isoformat(),
        "model_version": str(payload.get("model_version") or "active"),
        "trained_through_session": trained_through.isoformat(),
        "assumptions_hash": str(payload.get("assumptions_hash") or ""),
        "rows": len(rows),
        "pending_close_rows": int(sum(1 for row in rows if row.get("reconciliation_status") == "pending_close")),
        "finalized_rows": int(sum(1 for row in rows if row.get("reconciliation_status") == "finalized")),
        "disclaimer": clean_nan(ZeroDteDisclaimerMetadata().model_dump(mode="json")),
        "warnings": warnings,
    }
    return _ZeroDTEForwardResult(payload=summary, rows=rows)


__all__ = [
    "_build_intraday_store",
    "_load_study_candidate_bundle",
    "_validate_preflight_or_raise",
    "_run_walk_forward_or_raise",
    "_build_assumptions_with_hash",
    "_build_study_artifact",
    "_load_forward_model_bundle",
    "_load_forward_candidates",
    "_score_forward_candidates",
    "_build_forward_result",
]
