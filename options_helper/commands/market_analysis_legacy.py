from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.zero_dte_features import compute_zero_dte_features
from options_helper.analysis.zero_dte_labels import build_zero_dte_labels
from options_helper.analysis.zero_dte_policy import recommend_zero_dte_put_strikes
from options_helper.analysis.zero_dte_preflight import run_zero_dte_preflight
from options_helper.analysis.zero_dte_tail_model import (
    ZeroDTETailModel,
    ZeroDTETailModelConfig,
    fit_zero_dte_tail_model,
    score_zero_dte_tail_model,
)
from options_helper.backtesting.zero_dte_put import ZeroDTEPutSimulatorConfig
from options_helper.backtesting.zero_dte_walk_forward import ZeroDTEWalkForwardConfig, run_zero_dte_walk_forward
from options_helper.commands.common import _parse_date
from options_helper.data.zero_dte_dataset import DEFAULT_PROXY_UNDERLYING, ZeroDTEIntradayDatasetLoader
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.zero_dte_put_study import (
    DecisionAnchorMetadata,
    DecisionMode,
    FillModel,
    QuoteQualityStatus,
    SkipReason,
    ZeroDteDisclaimerMetadata,
    ZeroDteProbabilityRow,
    ZeroDtePutStudyArtifact,
    ZeroDteSimulationRow,
    ZeroDteStrikeLadderRow,
    ZeroDteStudyAssumptions,
)


app = typer.Typer(help="Market analysis utilities (informational only; not financial advice).")


_ZERO_DTE_DEFAULT_STRIKE_GRID: tuple[float, ...] = (-0.03, -0.02, -0.015, -0.01, -0.005)
_ZERO_DTE_FORWARD_KEY_FIELDS: tuple[str, ...] = (
    "symbol",
    "session_date",
    "decision_ts",
    "risk_tier",
    "model_version",
    "assumptions_hash",
)


@dataclass(frozen=True)
class _ZeroDTEStudyResult:
    artifact: ZeroDtePutStudyArtifact
    active_model: dict[str, Any] | None
    preflight_passed: bool
    preflight_messages: list[str]


@dataclass(frozen=True)
class _ZeroDTEForwardResult:
    payload: dict[str, Any]
    rows: list[dict[str, Any]]


def _package_workflows() -> tuple[Any, Any, Any, Any]:
    """Resolve package-level seams for monkeypatch compatibility."""
    market_analysis_pkg = importlib.import_module("options_helper.commands.market_analysis")

    return (
        market_analysis_pkg.get_storage_runtime_config,
        market_analysis_pkg.IntradayStore,
        market_analysis_pkg._run_zero_dte_put_study_workflow,
        market_analysis_pkg._run_zero_dte_forward_snapshot_workflow,
    )


def _build_intraday_store(root_dir: Path) -> Any:
    _, intraday_store_cls, _, _ = _package_workflows()
    return intraday_store_cls(root_dir)


def _core_impl_module() -> Any:
    return importlib.import_module("options_helper.commands.market_analysis.core_impl")


def _core_helpers_module() -> Any:
    return importlib.import_module("options_helper.commands.market_analysis.core_helpers")


def _core_io_module() -> Any:
    return importlib.import_module("options_helper.commands.market_analysis.core_io")


tail_risk = _core_impl_module().tail_risk
iv_surface = _core_impl_module().iv_surface
exposure = _core_impl_module().exposure
levels = _core_impl_module().levels

app.command("tail-risk")(tail_risk)
app.command("iv-surface")(iv_surface)
app.command("exposure")(exposure)
app.command("levels")(levels)

_normalize_symbol = _core_helpers_module()._normalize_symbol
_normalize_output_format = _core_helpers_module()._normalize_output_format
_dedupe = _core_helpers_module()._dedupe
_as_float = _core_helpers_module()._as_float
_normalize_daily_history = _core_io_module()._normalize_daily_history


@app.command("zero-dte-put-study")
def zero_dte_put_study(
    symbol: str = typer.Option(
        DEFAULT_PROXY_UNDERLYING,
        "--symbol",
        help="Proxy underlying symbol (default: SPY).",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Start session date (YYYY-MM-DD). Defaults to 60 calendar days before end date.",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="End session date (YYYY-MM-DD). Defaults to latest available intraday session.",
    ),
    decision_mode: str = typer.Option(
        DecisionMode.FIXED_TIME.value,
        "--decision-mode",
        help="Decision mode: fixed_time|rolling.",
    ),
    decision_times: str = typer.Option(
        "10:30",
        "--decision-times",
        help="Comma-separated decision times in ET (HH:MM) for fixed_time mode.",
    ),
    rolling_interval_minutes: int = typer.Option(
        15,
        "--rolling-interval-minutes",
        min=1,
        help="Rolling decision interval minutes when --decision-mode rolling.",
    ),
    risk_tiers: str = typer.Option(
        "0.005,0.01,0.02,0.05",
        "--risk-tiers",
        help="Comma-separated breach-probability tiers (for example: 0.005,0.01,0.02,0.05).",
    ),
    strike_grid: str = typer.Option(
        "-0.03,-0.02,-0.015,-0.01,-0.005",
        "--strike-grid",
        help="Comma-separated strike returns from entry (negative values).",
    ),
    fill_model: str = typer.Option(
        FillModel.BID.value,
        "--fill-model",
        help="Entry fill model: bid|mid|ask.",
    ),
    entry_slippage_bps: float = typer.Option(
        0.0,
        "--entry-slippage-bps",
        min=0.0,
        help="Entry slippage basis points.",
    ),
    exit_slippage_bps: float = typer.Option(
        0.0,
        "--exit-slippage-bps",
        min=0.0,
        help="Exit slippage basis points.",
    ),
    train_sessions: int = typer.Option(20, "--train-sessions", min=1, help="Walk-forward train sessions."),
    test_sessions: int = typer.Option(5, "--test-sessions", min=1, help="Walk-forward test sessions."),
    step_sessions: int = typer.Option(5, "--step-sessions", min=1, help="Walk-forward step sessions."),
    min_training_rows: int = typer.Option(
        20,
        "--min-training-rows",
        min=1,
        help="Minimum in-sample rows required before a walk-forward fit.",
    ),
    top_k_per_tier: int = typer.Option(
        3,
        "--top-k-per-tier",
        min=1,
        help="Recommended strike count per risk tier.",
    ),
    strict_preflight: bool = typer.Option(
        True,
        "--strict-preflight/--no-strict-preflight",
        help="Fail closed if data sufficiency preflight does not pass.",
    ),
    intraday_dir: Path = typer.Option(
        Path("data/intraday"),
        "--intraday-dir",
        help="Intraday partition root.",
    ),
    snapshot_cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshot-cache-dir",
        help="Options snapshot cache root.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Daily candle cache root.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path = typer.Option(
        Path("data/reports"),
        "--out",
        help="Output root for study artifacts and active model metadata.",
    ),
) -> None:
    """Run SPY-proxy 0DTE put study + walk-forward backtest (informational only)."""
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)
    sym = _normalize_symbol(symbol)

    try:
        _, _, run_study_workflow, _ = _package_workflows()
        parsed_mode = _parse_decision_mode(decision_mode)
        parsed_times = _parse_time_csv(decision_times)
        parsed_risk_tiers = _parse_positive_probability_csv(risk_tiers)
        parsed_strike_grid = _parse_strike_return_csv(strike_grid)
        parsed_fill_model = _parse_fill_model(fill_model)
        result = run_study_workflow(
            symbol=sym,
            start_date=start_date,
            end_date=end_date,
            decision_mode=parsed_mode,
            decision_times=parsed_times,
            rolling_interval_minutes=rolling_interval_minutes,
            risk_tiers=parsed_risk_tiers,
            strike_grid=parsed_strike_grid,
            fill_model=parsed_fill_model,
            entry_slippage_bps=float(entry_slippage_bps),
            exit_slippage_bps=float(exit_slippage_bps),
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            step_sessions=step_sessions,
            min_training_rows=min_training_rows,
            top_k_per_tier=top_k_per_tier,
            strict_preflight=strict_preflight,
            intraday_dir=intraday_dir,
            snapshot_cache_dir=snapshot_cache_dir,
            candle_cache_dir=candle_cache_dir,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if output_fmt == "json":
        console.print(result.artifact.model_dump_json(indent=2))
    else:
        _render_zero_dte_study_console(console, result)

    saved_artifact = _save_zero_dte_study_artifact(out, result.artifact, symbol=sym)
    if result.active_model is not None:
        saved_model = _save_zero_dte_active_model(out, symbol=sym, payload=result.active_model)
    else:
        saved_model = None
    if output_fmt != "json":
        console.print(f"Saved: {saved_artifact}")
        if saved_model is not None:
            console.print(f"Saved: {saved_model}")


@app.command("zero-dte-put-forward-snapshot")
def zero_dte_put_forward_snapshot(
    symbol: str = typer.Option(
        DEFAULT_PROXY_UNDERLYING,
        "--symbol",
        help="Proxy underlying symbol (default: SPY).",
    ),
    session_date: str | None = typer.Option(
        None,
        "--session-date",
        help="Session date (YYYY-MM-DD). Defaults to most recent intraday session for symbol.",
    ),
    intraday_dir: Path = typer.Option(
        Path("data/intraday"),
        "--intraday-dir",
        help="Intraday partition root.",
    ),
    snapshot_cache_dir: Path = typer.Option(
        Path("data/options_snapshots"),
        "--snapshot-cache-dir",
        help="Options snapshot cache root.",
    ),
    candle_cache_dir: Path = typer.Option(
        Path("data/candles"),
        "--candle-cache-dir",
        help="Daily candle cache root.",
    ),
    active_model_path: Path | None = typer.Option(
        None,
        "--active-model-path",
        help="Path to frozen active model JSON. Defaults to {out}/zero_dte_put_study/{SYMBOL}/active_model.json.",
    ),
    snapshot_path: Path | None = typer.Option(
        None,
        "--snapshot-path",
        help="Path to forward snapshot JSONL. Defaults to {out}/zero_dte_put_study/{SYMBOL}/forward_snapshots.jsonl.",
    ),
    format: str = typer.Option("console", "--format", help="Output format: console|json"),
    out: Path = typer.Option(
        Path("data/reports"),
        "--out",
        help="Output root for forward snapshot persistence.",
    ),
) -> None:
    """Score current/most-recent session using frozen active model only."""
    console = Console(width=200)
    output_fmt = _normalize_output_format(format)
    sym = _normalize_symbol(symbol)

    try:
        _, _, _, run_forward_workflow = _package_workflows()
        result = run_forward_workflow(
            symbol=sym,
            session_date=session_date,
            intraday_dir=intraday_dir,
            snapshot_cache_dir=snapshot_cache_dir,
            candle_cache_dir=candle_cache_dir,
            out=out,
            active_model_path=active_model_path,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    effective_snapshot_path = snapshot_path or _default_zero_dte_forward_snapshot_path(out, sym)
    persisted_count = _upsert_forward_snapshot_records(
        effective_snapshot_path,
        rows=result.rows,
        key_fields=_ZERO_DTE_FORWARD_KEY_FIELDS,
    )

    if output_fmt == "json":
        payload = dict(result.payload)
        payload["persisted_rows"] = persisted_count
        payload["snapshot_path"] = str(effective_snapshot_path)
        console.print(json.dumps(clean_nan(payload), indent=2))
    else:
        _render_zero_dte_forward_console(
            console,
            payload=result.payload,
            persisted_rows=persisted_count,
            snapshot_path=effective_snapshot_path,
        )

def _run_zero_dte_put_study_workflow(
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
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    train_sessions: int,
    test_sessions: int,
    step_sessions: int,
    min_training_rows: int,
    top_k_per_tier: int,
    strict_preflight: bool,
    intraday_dir: Path,
    snapshot_cache_dir: Path,
    candle_cache_dir: Path,
) -> _ZeroDTEStudyResult:
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

    preflight = run_zero_dte_preflight(features, labels, snapshots)
    preflight_messages = [item.message for item in preflight.diagnostics if not item.ok]
    if strict_preflight and not preflight.passed:
        raise ValueError("Preflight failed: " + ("; ".join(preflight_messages) or "unknown reason"))

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

    assumptions = ZeroDteStudyAssumptions(
        proxy_underlying_symbol=symbol,
        benchmark_decision_mode=decision_mode,
        benchmark_fixed_time_et=decision_times[0],
        rolling_interval_minutes=rolling_interval_minutes,
        fill_model=fill_model,
        fill_slippage_bps=entry_slippage_bps,
        risk_tier_breach_probabilities=risk_tiers,
    )
    assumptions_hash = _hash_zero_dte_assumptions(assumptions)
    recommendations = recommend_zero_dte_put_strikes(
        walk_forward.scored_rows,
        walk_forward.scored_rows,
        risk_tiers=risk_tiers,
    )

    artifact = ZeroDtePutStudyArtifact(
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
    active_model = _fit_and_serialize_active_model(
        candidates,
        symbol=symbol,
        strike_grid=strike_grid,
        assumptions=assumptions,
        assumptions_hash=assumptions_hash,
        fallback_trained_through=artifact.as_of,
    )
    return _ZeroDTEStudyResult(
        artifact=artifact,
        active_model=active_model,
        preflight_passed=preflight.passed,
        preflight_messages=preflight_messages,
    )


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
        raise ValueError(
            "Forward snapshot requires session_date strictly after trained_through_session."
        )

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

    state_rows = candidates.loc[
        :,
        ["session_date", "decision_ts", "time_of_day_bucket", "intraday_return", "iv_regime"],
    ].drop_duplicates(subset=["session_date", "decision_ts"], keep="first")
    scored = score_zero_dte_tail_model(
        model,
        state_rows,
        strike_returns=model.strike_returns,
    )
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
        "pending_close_rows": int(
            sum(1 for row in rows if row.get("reconciliation_status") == "pending_close")
        ),
        "finalized_rows": int(
            sum(1 for row in rows if row.get("reconciliation_status") == "finalized")
        ),
        "disclaimer": clean_nan(ZeroDteDisclaimerMetadata().model_dump(mode="json")),
        "warnings": warnings,
    }
    return _ZeroDTEForwardResult(payload=summary, rows=rows)


def _build_zero_dte_candidates(
    *,
    loader: ZeroDTEIntradayDatasetLoader,
    candles: pd.DataFrame,
    symbol: str,
    sessions: list[date],
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
    strike_grid: tuple[float, ...],
    risk_tiers: tuple[float, ...],
    fill_model: FillModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    candidate_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    snapshot_parts: list[pd.DataFrame] = []
    warnings: list[str] = []
    for session_date in sessions:
        previous_close = _resolve_previous_close(candles, session_date)
        if previous_close is None:
            warnings.append(f"Missing previous close for {session_date.isoformat()}.")
            continue
        day_times = _resolve_decision_times_for_session(
            session_date=session_date,
            decision_mode=decision_mode,
            decision_times=decision_times,
            rolling_interval_minutes=rolling_interval_minutes,
        )
        dataset = loader.load_day(
            session_date,
            underlying_symbol=symbol,
            decision_times=day_times,
        )
        if dataset.state_rows.empty or dataset.underlying_bars.empty:
            continue
        features = compute_zero_dte_features(
            dataset.state_rows,
            dataset.underlying_bars,
            previous_close=previous_close,
        )
        labels = build_zero_dte_labels(
            dataset.state_rows,
            dataset.underlying_bars,
            market_close_ts=pd.Timestamp(dataset.session.market_close).tz_convert("UTC"),
        )
        snapshots = _build_strike_snapshot_rows(
            loader=loader,
            session_date=session_date,
            labels=labels,
            previous_close=previous_close,
            strike_grid=strike_grid,
            symbol=symbol,
        )
        combined = _assemble_zero_dte_candidate_rows(
            features=features,
            labels=labels,
            snapshots=snapshots,
            fill_model=fill_model,
            decision_mode=decision_mode,
            risk_tiers=risk_tiers,
        )
        if not combined.empty:
            candidate_parts.append(combined)
        feature_parts.append(features)
        label_parts.append(labels)
        snapshot_parts.append(snapshots)
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    features_all = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    labels_all = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    snapshots_all = pd.concat(snapshot_parts, ignore_index=True) if snapshot_parts else pd.DataFrame()
    return candidates, features_all, labels_all, snapshots_all, warnings


def _resolve_zero_dte_study_range(
    *,
    start_date: str | None,
    end_date: str | None,
    intraday_dir: Path,
    symbol: str,
) -> tuple[date, date]:
    end = _parse_date(end_date) if end_date else _resolve_latest_intraday_session(intraday_dir, symbol)
    start = _parse_date(start_date) if start_date else (end - timedelta(days=60))
    if start > end:
        raise ValueError("--start-date must be on or before --end-date")
    return start, end


def _resolve_latest_intraday_session(intraday_dir: Path, symbol: str) -> date:
    root = intraday_dir / "stocks" / "bars" / "1Min" / symbol
    if not root.exists():
        raise ValueError(f"No intraday partitions found for {symbol} at {root}")
    days: list[date] = []
    for file_path in root.glob("*.parquet"):
        try:
            days.append(date.fromisoformat(file_path.stem))
        except ValueError:
            continue
    if not days:
        raise ValueError(f"No intraday session files found for {symbol} at {root}")
    return max(days)


def _resolve_decision_times_for_session(
    *,
    session_date: date,
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
) -> tuple[str, ...]:
    if decision_mode == DecisionMode.FIXED_TIME:
        return decision_times
    out: list[str] = []
    cursor = datetime.combine(session_date, time(9, 30))
    end = datetime.combine(session_date, time(16, 0))
    step = timedelta(minutes=rolling_interval_minutes)
    while cursor <= end:
        out.append(cursor.strftime("%H:%M"))
        cursor += step
    return tuple(out)


def _resolve_previous_close(candles: pd.DataFrame, session_date: date) -> float | None:
    if candles.empty:
        return None
    prior = candles.loc[candles["date"] < pd.Timestamp(session_date)].copy()
    if prior.empty:
        return None
    close = pd.to_numeric(prior["close"], errors="coerce").dropna()
    if close.empty:
        return None
    value = float(close.iloc[-1])
    return value if value > 0 else None


def _build_strike_snapshot_rows(
    *,
    loader: ZeroDTEIntradayDatasetLoader,
    session_date: date,
    labels: pd.DataFrame,
    previous_close: float,
    strike_grid: tuple[float, ...],
    symbol: str,
) -> pd.DataFrame:
    if labels is None or labels.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for _, label_row in labels.iterrows():
        entry_anchor_ts = pd.to_datetime(label_row.get("entry_anchor_ts"), errors="coerce", utc=True)
        decision_ts = pd.to_datetime(label_row.get("decision_ts"), errors="coerce", utc=True)
        if pd.isna(entry_anchor_ts) or pd.isna(decision_ts):
            continue
        snap = loader.load_strike_premium_snapshot(
            session_date,
            previous_close=previous_close,
            strike_returns=strike_grid,
            entry_anchor_ts=entry_anchor_ts,
            underlying_symbol=symbol,
        )
        if snap.empty:
            continue
        snap = snap.copy()
        snap["decision_ts"] = decision_ts
        snap["decision_bar_completed_ts"] = pd.to_datetime(
            label_row.get("decision_bar_completed_ts"),
            errors="coerce",
            utc=True,
        )
        snap["close_label_ts"] = pd.to_datetime(label_row.get("close_label_ts"), errors="coerce", utc=True)
        snap["close_price"] = pd.to_numeric(label_row.get("close_price"), errors="coerce")
        snap["close_return_from_entry"] = pd.to_numeric(
            label_row.get("close_return_from_entry"),
            errors="coerce",
        )
        snap["entry_anchor_price"] = pd.to_numeric(label_row.get("entry_anchor_price"), errors="coerce")
        snap["label_status"] = label_row.get("label_status")
        snap["label_skip_reason"] = label_row.get("skip_reason")
        parts.append(snap)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _assemble_zero_dte_candidate_rows(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: pd.DataFrame,
    fill_model: FillModel,
    decision_mode: DecisionMode,
    risk_tiers: tuple[float, ...],
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty:
        return pd.DataFrame()
    feature_base = features.copy()
    feature_base["session_date"] = pd.to_datetime(feature_base["session_date"], errors="coerce").dt.date
    feature_base["decision_ts"] = pd.to_datetime(feature_base["decision_ts"], errors="coerce", utc=True)
    label_base = labels.copy()
    label_base["session_date"] = pd.to_datetime(label_base["session_date"], errors="coerce").dt.date
    label_base["decision_ts"] = pd.to_datetime(label_base["decision_ts"], errors="coerce", utc=True)
    base = feature_base.merge(
        label_base,
        on=["session_date", "decision_ts"],
        how="outer",
        suffixes=("_feature", "_label"),
    )
    snap = snapshots.copy()
    snap["session_date"] = pd.to_datetime(snap["session_date"], errors="coerce").dt.date
    snap["decision_ts"] = pd.to_datetime(snap["decision_ts"], errors="coerce", utc=True)
    snap["entry_anchor_ts"] = pd.to_datetime(snap["entry_anchor_ts"], errors="coerce", utc=True)
    merged = snap.merge(base, on=["session_date", "decision_ts"], how="left", sort=True)

    for column in (
        "entry_anchor_ts",
        "decision_bar_completed_ts",
        "close_label_ts",
        "close_price",
        "entry_anchor_price",
        "close_return_from_entry",
        "label_status",
        "skip_reason",
    ):
        if column in merged.columns:
            continue
        left = merged.get(f"{column}_x")
        right = merged.get(f"{column}_y")
        if left is None and right is None:
            continue
        if left is None:
            merged[column] = right
        elif right is None:
            merged[column] = left
        else:
            merged[column] = left
            missing_mask = merged[column].isna()
            if bool(missing_mask.any()):
                merged.loc[missing_mask, column] = right.loc[missing_mask]

    merged["strike_return"] = pd.to_numeric(merged["target_strike_return"], errors="coerce")
    merged["target_strike_price"] = pd.to_numeric(merged["target_strike_price"], errors="coerce")
    merged["strike_price"] = pd.to_numeric(merged.get("strike_price"), errors="coerce")
    merged["premium_estimate"] = merged.apply(
        lambda row: _apply_fill_model_to_premium(
            entry_premium=row.get("entry_premium"),
            spread=row.get("spread"),
            fill_model=fill_model,
        ),
        axis=1,
    )
    merged["decision_mode"] = decision_mode.value
    merged["policy_status"] = "ok"
    merged["policy_reason"] = None

    skip_snapshot = merged.get("skip_reason")
    if skip_snapshot is None:
        skip_snapshot = pd.Series([None] * len(merged), index=merged.index, dtype="object")
    skip_label = merged.get("label_skip_reason")
    if skip_label is None:
        skip_label = pd.Series([None] * len(merged), index=merged.index, dtype="object")
    invalid_premium = ~pd.to_numeric(merged["premium_estimate"], errors="coerce").gt(0.0)
    has_snapshot_skip = skip_snapshot.notna() & skip_snapshot.astype(str).str.strip().ne("")
    has_label_skip = skip_label.notna() & skip_label.astype(str).str.strip().ne("")
    has_skip = invalid_premium | has_snapshot_skip | has_label_skip
    merged.loc[has_skip, "policy_status"] = "skip"
    merged.loc[has_skip, "policy_reason"] = merged.loc[has_skip].apply(
        lambda row: _first_text(row.get("skip_reason"), row.get("label_skip_reason"), "invalid_premium_estimate"),
        axis=1,
    )

    merged["feature_status"] = merged.get("feature_status").fillna("unknown")
    merged["label_status"] = merged.get("label_status").fillna("unknown")
    merged["quote_quality_status"] = merged.get("quote_quality_status").fillna(
        QuoteQualityStatus.UNKNOWN.value
    )
    merged["decision_bar_completed_ts"] = pd.to_datetime(
        merged.get("decision_bar_completed_ts"),
        errors="coerce",
        utc=True,
    )
    merged["close_label_ts"] = pd.to_datetime(merged.get("close_label_ts"), errors="coerce", utc=True)

    keep_cols = [
        "session_date",
        "decision_ts",
        "decision_mode",
        "decision_bar_completed_ts",
        "entry_anchor_ts",
        "close_label_ts",
        "time_of_day_bucket",
        "intraday_return",
        "iv_regime",
        "strike_return",
        "strike_price",
        "target_strike_price",
        "premium_estimate",
        "quote_quality_status",
        "policy_status",
        "policy_reason",
        "close_price",
        "entry_anchor_price",
        "close_return_from_entry",
        "feature_status",
        "label_status",
    ]
    out = merged.loc[:, keep_cols].copy()
    out["intraday_min_return_from_entry"] = float("nan")
    out["intraday_max_return_from_entry"] = float("nan")
    out["adaptive_exit_premium"] = float("nan")
    return _expand_risk_tiers(out, risk_tiers=risk_tiers)


def _expand_risk_tiers(frame: pd.DataFrame, *, risk_tiers: tuple[float, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    parts = []
    for tier in risk_tiers:
        part = frame.copy()
        part["risk_tier"] = float(tier)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _apply_fill_model_to_premium(*, entry_premium: object, spread: object, fill_model: FillModel) -> float:
    premium = _as_float(entry_premium)
    if premium is None or premium <= 0:
        return float("nan")
    spread_value = _as_float(spread)
    if spread_value is None or spread_value < 0.0:
        spread_value = 0.0
    if fill_model == FillModel.BID:
        return premium
    if fill_model == FillModel.MID:
        return premium + (spread_value / 2.0)
    return premium + spread_value


def _build_zero_dte_probability_rows(
    scored_rows: pd.DataFrame,
    *,
    assumptions_hash: str,
) -> list[ZeroDteProbabilityRow]:
    if scored_rows is None or scored_rows.empty:
        return []
    out: list[ZeroDteProbabilityRow] = []
    for _, raw in scored_rows.iterrows():
        risk_tier = _as_float(raw.get("risk_tier"))
        strike_return = _as_float(raw.get("strike_return"))
        probability = _as_float(raw.get("breach_probability"))
        if risk_tier is None or strike_return is None or probability is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        out.append(
            ZeroDteProbabilityRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                risk_tier=risk_tier,
                strike_return=strike_return,
                breach_probability=probability,
                breach_probability_ci_low=_as_float(raw.get("breach_probability_ci_low")),
                breach_probability_ci_high=_as_float(raw.get("breach_probability_ci_high")),
                sample_size=int(_as_float(raw.get("sample_size")) or 0),
                model_version=str(raw.get("model_version") or "unknown_model"),
                assumptions_hash=assumptions_hash,
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("policy_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_strike_ladder_rows(
    recommendations: pd.DataFrame,
    *,
    fill_model: FillModel,
    top_k_per_tier: int,
) -> list[ZeroDteStrikeLadderRow]:
    if recommendations is None or recommendations.empty:
        return []
    out: list[ZeroDteStrikeLadderRow] = []
    ordered = recommendations.sort_values(
        by=["decision_ts", "risk_tier", "ladder_rank"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    for _, raw in ordered.iterrows():
        rank = int(_as_float(raw.get("ladder_rank")) or 1)
        if rank > top_k_per_tier:
            continue
        risk_tier = _as_float(raw.get("risk_tier"))
        strike_price = _as_float(raw.get("strike_price"))
        strike_return = _as_float(raw.get("strike_return"))
        breach_probability = _as_float(raw.get("breach_probability"))
        if risk_tier is None or strike_price is None or strike_return is None or breach_probability is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        out.append(
            ZeroDteStrikeLadderRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                risk_tier=risk_tier,
                ladder_rank=rank,
                strike_price=strike_price,
                strike_return=strike_return,
                breach_probability=breach_probability,
                premium_estimate=_as_float(raw.get("premium_estimate")),
                fill_model=fill_model,
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("policy_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_simulation_rows(
    trade_rows: pd.DataFrame,
    *,
    fill_model: FillModel,
) -> list[ZeroDteSimulationRow]:
    if trade_rows is None or trade_rows.empty:
        return []
    out: list[ZeroDteSimulationRow] = []
    filled = trade_rows.loc[
        trade_rows.get("status", pd.Series(dtype=object)).astype(str).str.lower().eq("filled")
    ].copy()
    for _, raw in filled.iterrows():
        risk_tier = _as_float(raw.get("risk_tier"))
        if risk_tier is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        mode = str(raw.get("exit_mode") or "").strip().lower()
        if mode not in {"hold_to_close", "adaptive_exit"}:
            continue
        out.append(
            ZeroDteSimulationRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                contract_symbol=_as_clean_text(raw.get("contract_symbol")),
                risk_tier=risk_tier,
                exit_mode=mode,
                fill_model=fill_model,
                entry_premium=_as_float(raw.get("entry_premium_gross")),
                exit_premium=_as_float(raw.get("exit_premium_gross")),
                pnl_per_contract=_as_float(raw.get("pnl_per_contract")),
                max_loss_proxy=_as_float(raw.get("max_loss_proxy_per_contract")),
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("skip_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_anchor(row: pd.Series) -> DecisionAnchorMetadata | None:
    session = pd.to_datetime(row.get("session_date"), errors="coerce")
    decision_ts = pd.to_datetime(row.get("decision_ts"), errors="coerce", utc=True)
    decision_bar_ts = pd.to_datetime(
        row.get("decision_bar_completed_ts") or row.get("decision_ts"),
        errors="coerce",
        utc=True,
    )
    close_ts = pd.to_datetime(row.get("close_label_ts"), errors="coerce", utc=True)
    if pd.isna(session) or pd.isna(decision_ts) or pd.isna(decision_bar_ts) or pd.isna(close_ts):
        return None
    mode_text = str(row.get("decision_mode") or DecisionMode.FIXED_TIME.value).strip().lower()
    mode = DecisionMode(mode_text) if mode_text in {item.value for item in DecisionMode} else DecisionMode.FIXED_TIME
    entry_anchor = pd.to_datetime(row.get("entry_anchor_ts"), errors="coerce", utc=True)
    return DecisionAnchorMetadata(
        session_date=session.date(),
        decision_ts=decision_ts.to_pydatetime(),
        decision_bar_completed_ts=decision_bar_ts.to_pydatetime(),
        close_label_ts=close_ts.to_pydatetime(),
        decision_mode=mode,
        entry_anchor_ts=entry_anchor.to_pydatetime() if pd.notna(entry_anchor) else None,
    )


def _fit_and_serialize_active_model(
    candidates: pd.DataFrame,
    *,
    symbol: str,
    strike_grid: tuple[float, ...],
    assumptions: ZeroDteStudyAssumptions,
    assumptions_hash: str,
    fallback_trained_through: date,
) -> dict[str, Any] | None:
    if candidates is None or candidates.empty:
        return None
    training = candidates.copy()
    training = training.loc[
        training["feature_status"].astype(str).str.lower().eq("ok")
        & training["label_status"].astype(str).str.lower().eq("ok")
    ].copy()
    training = training.loc[pd.to_numeric(training["close_return_from_entry"], errors="coerce").notna()].copy()
    training = training.sort_values(
        by=["session_date", "decision_ts", "entry_anchor_ts"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop_duplicates(subset=["session_date", "decision_ts", "entry_anchor_ts"], keep="first")
    if training.empty:
        return None
    model = fit_zero_dte_tail_model(training, strike_returns=strike_grid)
    trained_through = _resolve_as_of_date(training, default=fallback_trained_through)
    return _serialize_tail_model_payload(
        model=model,
        symbol=symbol,
        model_version=f"active_{trained_through.isoformat()}",
        trained_through_session=trained_through,
        assumptions=assumptions,
        assumptions_hash=assumptions_hash,
    )


def _serialize_tail_model_payload(
    *,
    model: ZeroDTETailModel,
    symbol: str,
    model_version: str,
    trained_through_session: date,
    assumptions: ZeroDteStudyAssumptions,
    assumptions_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": utc_now().isoformat(),
        "symbol": symbol,
        "model_version": model_version,
        "trained_through_session": trained_through_session.isoformat(),
        "assumptions_hash": assumptions_hash,
        "assumptions": clean_nan(assumptions.model_dump(mode="json")),
        "tail_model": {
            "config": asdict(model.config),
            "strike_returns": list(model.strike_returns),
            "training_sample_size": int(model.training_sample_size),
            "global_stats": _frame_records(model.global_stats),
            "parent_stats": _frame_records(model.parent_stats),
            "bucket_stats": _frame_records(model.bucket_stats),
        },
    }


def _deserialize_tail_model(payload: dict[str, Any]) -> ZeroDTETailModel:
    config = ZeroDTETailModelConfig(**(payload.get("config") or {}))
    strike_returns = tuple(float(v) for v in payload.get("strike_returns") or _ZERO_DTE_DEFAULT_STRIKE_GRID)
    return ZeroDTETailModel(
        config=config,
        strike_returns=strike_returns,
        training_sample_size=int(payload.get("training_sample_size", 0)),
        global_stats=pd.DataFrame(payload.get("global_stats") or []),
        parent_stats=pd.DataFrame(payload.get("parent_stats") or []),
        bucket_stats=pd.DataFrame(payload.get("bucket_stats") or []),
    )


def _build_forward_snapshot_rows(
    *,
    recommendations: pd.DataFrame,
    scored_rows: pd.DataFrame,
    symbol: str,
    model_version: str,
    assumptions_hash: str,
    trained_through_session: str,
) -> list[dict[str, Any]]:
    if recommendations is None or recommendations.empty:
        return []
    lookup = scored_rows.loc[
        :,
        [
            "session_date",
            "decision_ts",
            "entry_anchor_ts",
            "risk_tier",
            "strike_return",
            "close_return_from_entry",
            "close_price",
            "close_label_ts",
        ],
    ].copy()
    merged = recommendations.merge(
        lookup,
        on=["session_date", "decision_ts", "entry_anchor_ts", "risk_tier", "strike_return"],
        how="left",
        sort=True,
    )
    out: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        session = pd.to_datetime(row.get("session_date"), errors="coerce")
        decision_ts = pd.to_datetime(row.get("decision_ts"), errors="coerce", utc=True)
        risk_tier = _as_float(row.get("risk_tier"))
        if pd.isna(session) or pd.isna(decision_ts) or risk_tier is None:
            continue
        realized = _as_float(row.get("close_return_from_entry"))
        out.append(
            clean_nan(
                {
                    "symbol": symbol,
                    "session_date": session.date().isoformat(),
                    "decision_ts": decision_ts.isoformat(),
                    "entry_anchor_ts": _timestamp_to_iso(row.get("entry_anchor_ts")),
                    "close_label_ts": _timestamp_to_iso(row.get("close_label_ts")),
                    "risk_tier": risk_tier,
                    "ladder_rank": int(_as_float(row.get("ladder_rank")) or 1),
                    "strike_return": _as_float(row.get("strike_return")),
                    "strike_price": _as_float(row.get("strike_price")),
                    "breach_probability": _as_float(row.get("breach_probability")),
                    "breach_probability_ci_low": _as_float(row.get("breach_probability_ci_low")),
                    "breach_probability_ci_high": _as_float(row.get("breach_probability_ci_high")),
                    "premium_estimate": _as_float(row.get("premium_estimate")),
                    "quote_quality_status": _as_clean_text(row.get("quote_quality_status"))
                    or QuoteQualityStatus.UNKNOWN.value,
                    "policy_status": _as_clean_text(row.get("policy_status")) or "ok",
                    "policy_reason": _as_clean_text(row.get("policy_reason")),
                    "model_version": model_version,
                    "assumptions_hash": assumptions_hash,
                    "trained_through_session": trained_through_session,
                    "realized_close_return_from_entry": realized,
                    "realized_close_price": _as_float(row.get("close_price")),
                    "reconciliation_status": "finalized" if realized is not None else "pending_close",
                    "generated_at": utc_now().isoformat(),
                }
            )
        )
    return out


def _upsert_forward_snapshot_records(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> int:
    existing = _read_jsonl_records(path)
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in existing:
        key = _row_key(row, key_fields=key_fields)
        if key is not None:
            by_key[key] = row
    for row in rows:
        key = _row_key(row, key_fields=key_fields)
        if key is not None:
            by_key[key] = clean_nan(row)

    ordered = [by_key[key] for key in sorted(by_key.keys())]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(clean_nan(row), sort_keys=True) + "\n")
    return len(ordered)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


def _row_key(row: dict[str, Any], *, key_fields: tuple[str, ...]) -> tuple[str, ...] | None:
    values: list[str] = []
    for field in key_fields:
        text = _as_clean_text(row.get(field))
        if text is None:
            return None
        values.append(text)
    return tuple(values)


def _save_zero_dte_study_artifact(out: Path, artifact: ZeroDtePutStudyArtifact, *, symbol: str) -> Path:
    base = out / "zero_dte_put_study" / symbol
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{artifact.as_of.isoformat()}.json"
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path


def _save_zero_dte_active_model(out: Path, *, symbol: str, payload: dict[str, Any]) -> Path:
    path = _default_zero_dte_active_model_path(out, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_nan(payload), indent=2, sort_keys=True), encoding="utf-8")
    return path


def _default_zero_dte_active_model_path(out: Path, symbol: str) -> Path:
    return out / "zero_dte_put_study" / symbol / "active_model.json"


def _default_zero_dte_forward_snapshot_path(out: Path, symbol: str) -> Path:
    return out / "zero_dte_put_study" / symbol / "forward_snapshots.jsonl"


def _resolve_as_of_date(frame: pd.DataFrame, *, default: date) -> date:
    if frame is None or frame.empty or "session_date" not in frame.columns:
        return default
    parsed = pd.to_datetime(frame["session_date"], errors="coerce").dropna()
    if parsed.empty:
        return default
    return parsed.max().date()


def _render_zero_dte_study_console(console: Console, result: _ZeroDTEStudyResult) -> None:
    artifact = result.artifact
    console.print(
        f"[bold]{artifact.assumptions.proxy_underlying_symbol}[/bold] 0DTE put study as-of {artifact.as_of}"
    )
    console.print(
        "Rows: "
        f"probabilities={len(artifact.probability_rows)}, "
        f"strike_ladders={len(artifact.strike_ladder_rows)}, "
        f"simulations={len(artifact.simulation_rows)}"
    )
    console.print("Preflight: " + ("passed" if result.preflight_passed else "failed (soft)"))
    if result.preflight_messages:
        console.print("Preflight diagnostics: " + "; ".join(result.preflight_messages))
    if artifact.warnings:
        console.print("Warnings: " + "; ".join(artifact.warnings))
    console.print("Proxy notice: " + artifact.disclaimer.spy_proxy_caveat)
    console.print(artifact.disclaimer.not_financial_advice)


def _render_zero_dte_forward_console(
    console: Console,
    *,
    payload: dict[str, Any],
    persisted_rows: int,
    snapshot_path: Path,
) -> None:
    console.print(
        f"[bold]{payload.get('symbol', DEFAULT_PROXY_UNDERLYING)}[/bold] forward snapshot session "
        f"{payload.get('session_date', '-')}"
    )
    console.print(
        "Rows: "
        f"total={payload.get('rows', 0)}, "
        f"pending_close={payload.get('pending_close_rows', 0)}, "
        f"finalized={payload.get('finalized_rows', 0)}"
    )
    console.print(f"Persisted unique rows: {persisted_rows}")
    console.print(f"Snapshot file: {snapshot_path}")
    disclaimer = payload.get("disclaimer") if isinstance(payload.get("disclaimer"), dict) else {}
    proxy = _as_clean_text(disclaimer.get("spy_proxy_caveat"))
    nfa = _as_clean_text(disclaimer.get("not_financial_advice"))
    if proxy:
        console.print("Proxy notice: " + proxy)
    if nfa:
        console.print(nfa)


def _parse_decision_mode(raw: str) -> DecisionMode:
    text = str(raw or "").strip().lower()
    if text not in {item.value for item in DecisionMode}:
        raise ValueError("--decision-mode must be fixed_time or rolling")
    return DecisionMode(text)


def _parse_fill_model(raw: str) -> FillModel:
    text = str(raw or "").strip().lower()
    if text not in {item.value for item in FillModel}:
        raise ValueError("--fill-model must be bid, mid, or ask")
    return FillModel(text)


def _parse_time_csv(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(raw or "").split(",") if item.strip())
    if not values:
        raise ValueError("At least one decision time is required.")
    for value in values:
        try:
            time.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"Invalid decision time '{value}' (expected HH:MM).") from exc
    return values


def _parse_positive_probability_csv(raw: str) -> tuple[float, ...]:
    values = _parse_float_csv(raw)
    if not values:
        raise ValueError("At least one risk tier is required.")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("Risk tiers must be in (0, 1).")
    return tuple(sorted(set(values)))


def _parse_strike_return_csv(raw: str) -> tuple[float, ...]:
    values = _parse_float_csv(raw)
    if not values:
        raise ValueError("At least one strike return is required.")
    if any(value >= 0.0 for value in values):
        raise ValueError("Strike grid values must be negative returns.")
    return tuple(sorted(set(values)))


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in [item.strip() for item in str(raw or "").split(",") if item.strip()]:
        try:
            out.append(float(part))
        except ValueError as exc:
            raise ValueError(f"Invalid numeric value '{part}'.") from exc
    return tuple(out)


def _coerce_quote_quality_status(raw: object) -> QuoteQualityStatus:
    text = str(raw or "").strip().lower()
    mapping = {item.value: item for item in QuoteQualityStatus}
    return mapping.get(text, QuoteQualityStatus.UNKNOWN)


def _coerce_skip_reason(raw: object) -> SkipReason | None:
    text = str(raw or "").strip().lower()
    if not text:
        return None
    mapping = {item.value: item for item in SkipReason}
    return mapping.get(text)


def _hash_zero_dte_assumptions(assumptions: ZeroDteStudyAssumptions) -> str:
    payload = json.dumps(clean_nan(assumptions.model_dump(mode="json")), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_clean_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _first_text(*values: object) -> str | None:
    for value in values:
        text = _as_clean_text(value)
        if text:
            return text
    return None


def _timestamp_to_iso(value: object) -> str | None:
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).isoformat()


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [clean_nan(row) for row in frame.to_dict(orient="records")]

