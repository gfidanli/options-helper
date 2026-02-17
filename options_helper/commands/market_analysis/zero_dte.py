from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from options_helper.commands.market_analysis.core_helpers import _normalize_output_format, _normalize_symbol
from options_helper.commands.market_analysis.zero_dte_output import (
    _default_zero_dte_forward_snapshot_path,
    _render_zero_dte_forward_console,
    _render_zero_dte_study_console,
    _save_zero_dte_active_model,
    _save_zero_dte_study_artifact,
    _upsert_forward_snapshot_records,
)
from options_helper.commands.market_analysis.zero_dte_parsing import (
    _parse_decision_mode,
    _parse_fill_model,
    _parse_positive_probability_csv,
    _parse_strike_return_csv,
    _parse_time_csv,
)
from options_helper.commands.market_analysis.zero_dte_types import _ZERO_DTE_FORWARD_KEY_FIELDS
from options_helper.data.zero_dte_dataset import DEFAULT_PROXY_UNDERLYING
from options_helper.schemas.common import clean_nan
from options_helper.schemas.zero_dte_put_study import DecisionMode, FillModel


def _package_workflows() -> tuple[Any, Any]:
    market_analysis_pkg = importlib.import_module("options_helper.commands.market_analysis")
    return (
        market_analysis_pkg._run_zero_dte_put_study_workflow,
        market_analysis_pkg._run_zero_dte_forward_snapshot_workflow,
    )


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
        run_study_workflow, _ = _package_workflows()
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
        _, run_forward_workflow = _package_workflows()
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


def register(app: typer.Typer) -> None:
    app.command("zero-dte-put-study")(zero_dte_put_study)
    app.command("zero-dte-put-forward-snapshot")(zero_dte_put_forward_snapshot)


__all__ = ["register", "zero_dte_put_study", "zero_dte_put_forward_snapshot", "_package_workflows"]
