from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from click.core import Context as ClickContext
from rich.console import Console
import typer

import options_helper.cli_deps as cli_deps
from .strategy_model_helpers_legacy import (
    _build_strategy_filter_config,
    _build_strategy_signal_kwargs,
    _build_target_ladder,
    _merge_strategy_modeling_profile_values,
    _normalize_output_timezone,
    _option_was_set_on_command_line,
    _parse_allowed_volatility_regimes,
    _parse_iso_date,
    _resolve_strategy_symbols,
    _split_csv_option,
)
from .strategy_model_runtime_output import (
    emit_artifact_paths,
    emit_directional_summary,
    emit_filter_summary,
)
from .strategy_model_runtime_profile_values import build_cli_profile_values
from .strategy_model_runtime_progress import (
    StageTracker,
    emit_stage_timings,
    instrument_service,
    run_strategy_model,
)

_VALID_STRATEGIES = {
    "sfp",
    "msb",
    "orb",
    "ma_crossover",
    "trend_following",
    "fib_retracement",
}


@dataclass(frozen=True)
class _ValidatedInputs:
    normalized_cli_strategy: str
    normalized_max_hold_timeframe: str
    normalized_output_timezone: str
    parsed_cli_start_date: Any
    parsed_cli_end_date: Any
    parsed_cli_allowed_volatility_regimes: tuple[str, ...]


@dataclass(frozen=True)
class _ProfileState:
    effective_profile: Any
    normalized_strategy: str
    signal_kwargs: dict[str, object]
    filter_config: Any
    target_ladder: tuple[Any, ...]


def _validate_core_inputs(*, ctx: typer.Context, params: dict[str, Any]) -> tuple[str, str, str]:
    strategy = str(params["strategy"]).strip().lower()
    if strategy not in _VALID_STRATEGIES:
        raise typer.BadParameter(
            "--strategy must be one of: sfp, msb, orb, ma_crossover, trend_following, fib_retracement"
        )
    if float(params["starting_capital"]) <= 0.0:
        raise typer.BadParameter("--starting-capital must be > 0")
    risk_per_trade_pct = float(params["risk_per_trade_pct"])
    if risk_per_trade_pct <= 0.0 or risk_per_trade_pct > 100.0:
        raise typer.BadParameter("--risk-per-trade-pct must be > 0 and <= 100")
    if str(params["gap_fill_policy"]) != "fill_at_open":
        raise typer.BadParameter("--gap-fill-policy currently supports only fill_at_open")

    max_hold_bars = params["max_hold_bars"]
    if max_hold_bars is not None and int(max_hold_bars) < 1:
        raise typer.BadParameter("--max-hold-bars must be >= 1")
    if bool(params["disable_max_hold"]) and _option_was_set_on_command_line(ctx, "max_hold_bars"):
        raise typer.BadParameter("--disable-max-hold cannot be combined with --max-hold-bars")

    if int(params["segment_min_trades"]) < 1:
        raise typer.BadParameter("--segment-min-trades must be >= 1")
    if int(params["segment_limit"]) < 1:
        raise typer.BadParameter("--segment-limit must be >= 1")

    universe_limit = params["universe_limit"]
    if universe_limit is not None and int(universe_limit) < 1:
        raise typer.BadParameter("--universe-limit must be >= 1")

    from options_helper.schemas.strategy_modeling_policy import normalize_max_hold_timeframe

    try:
        normalized_max_hold_timeframe = normalize_max_hold_timeframe(str(params["max_hold_timeframe"]))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    normalized_output_timezone = _normalize_output_timezone(str(params["output_timezone"]))
    return strategy, normalized_max_hold_timeframe, normalized_output_timezone


def _validate_pre_profile_inputs(*, params: dict[str, Any], strategy: str) -> tuple[Any, Any, tuple[str, ...]]:
    parsed_start = _parse_iso_date(params["start_date"], option_name="--start-date")
    parsed_end = _parse_iso_date(params["end_date"], option_name="--end-date")
    if parsed_start and parsed_end and parsed_start > parsed_end:
        raise typer.BadParameter("--start-date must be <= --end-date")

    _build_strategy_signal_kwargs(
        strategy=strategy,
        fib_retracement_pct=float(params["fib_retracement_pct"]),
        ma_fast_window=int(params["ma_fast_window"]),
        ma_slow_window=int(params["ma_slow_window"]),
        ma_trend_window=int(params["ma_trend_window"]),
        ma_fast_type=str(params["ma_fast_type"]),
        ma_slow_type=str(params["ma_slow_type"]),
        ma_trend_type=str(params["ma_trend_type"]),
        trend_slope_lookback_bars=int(params["trend_slope_lookback_bars"]),
        atr_window=int(params["atr_window"]),
        atr_stop_multiple=float(params["atr_stop_multiple"]),
    )
    _build_strategy_filter_config(
        allow_shorts=bool(params["allow_shorts"]),
        enable_orb_confirmation=bool(params["enable_orb_confirmation"]),
        orb_range_minutes=int(params["orb_range_minutes"]),
        orb_confirmation_cutoff_et=str(params["orb_confirmation_cutoff_et"]),
        orb_stop_policy=str(params["orb_stop_policy"]),
        enable_atr_stop_floor=bool(params["enable_atr_stop_floor"]),
        atr_stop_floor_multiple=float(params["atr_stop_floor_multiple"]),
        enable_rsi_extremes=bool(params["enable_rsi_extremes"]),
        enable_ema9_regime=bool(params["enable_ema9_regime"]),
        ema9_slope_lookback_bars=int(params["ema9_slope_lookback_bars"]),
        enable_volatility_regime=bool(params["enable_volatility_regime"]),
        allowed_volatility_regimes=str(params["allowed_volatility_regimes"]),
    )
    allowed_regimes = _parse_allowed_volatility_regimes(
        str(params["allowed_volatility_regimes"]),
        option_name="--allowed-volatility-regimes",
    )
    return parsed_start, parsed_end, allowed_regimes


def _validate_cli_inputs(*, ctx: typer.Context, params: dict[str, Any]) -> _ValidatedInputs:
    strategy, normalized_max_hold_timeframe, normalized_output_timezone = _validate_core_inputs(
        ctx=ctx,
        params=params,
    )
    parsed_start, parsed_end, allowed_regimes = _validate_pre_profile_inputs(
        params=params,
        strategy=strategy,
    )
    return _ValidatedInputs(
        normalized_cli_strategy=strategy,
        normalized_max_hold_timeframe=normalized_max_hold_timeframe,
        normalized_output_timezone=normalized_output_timezone,
        parsed_cli_start_date=parsed_start,
        parsed_cli_end_date=parsed_end,
        parsed_cli_allowed_volatility_regimes=allowed_regimes,
    )


def _load_profile(*, params: dict[str, Any]) -> Any:
    if params["profile"] is None:
        return None

    from options_helper.data.strategy_modeling_profiles import (
        StrategyModelingProfileStoreError,
        load_strategy_modeling_profile,
    )

    try:
        return load_strategy_modeling_profile(Path(params["profile_path"]), str(params["profile"]))
    except StrategyModelingProfileStoreError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _build_profile_state(
    *,
    params: dict[str, Any],
    effective_profile: Any,
) -> _ProfileState:
    normalized_strategy = effective_profile.strategy
    signal_kwargs = _build_strategy_signal_kwargs(
        strategy=normalized_strategy,
        fib_retracement_pct=float(effective_profile.fib_retracement_pct),
        ma_fast_window=int(effective_profile.ma_fast_window),
        ma_slow_window=int(effective_profile.ma_slow_window),
        ma_trend_window=int(effective_profile.ma_trend_window),
        ma_fast_type=effective_profile.ma_fast_type,
        ma_slow_type=effective_profile.ma_slow_type,
        ma_trend_type=effective_profile.ma_trend_type,
        trend_slope_lookback_bars=int(effective_profile.trend_slope_lookback_bars),
        atr_window=int(effective_profile.atr_window),
        atr_stop_multiple=float(effective_profile.atr_stop_multiple),
    )
    filter_config = _build_strategy_filter_config(
        allow_shorts=effective_profile.allow_shorts,
        enable_orb_confirmation=effective_profile.enable_orb_confirmation,
        orb_range_minutes=int(effective_profile.orb_range_minutes),
        orb_confirmation_cutoff_et=effective_profile.orb_confirmation_cutoff_et,
        orb_stop_policy=effective_profile.orb_stop_policy,
        enable_atr_stop_floor=effective_profile.enable_atr_stop_floor,
        atr_stop_floor_multiple=float(effective_profile.atr_stop_floor_multiple),
        enable_rsi_extremes=effective_profile.enable_rsi_extremes,
        enable_ema9_regime=effective_profile.enable_ema9_regime,
        ema9_slope_lookback_bars=int(effective_profile.ema9_slope_lookback_bars),
        enable_volatility_regime=effective_profile.enable_volatility_regime,
        allowed_volatility_regimes=",".join(effective_profile.allowed_volatility_regimes),
    )
    target_ladder = _build_target_ladder(
        min_tenths=int(effective_profile.r_ladder_min_tenths),
        max_tenths=int(effective_profile.r_ladder_max_tenths),
        step_tenths=int(effective_profile.r_ladder_step_tenths),
    )
    return _ProfileState(
        effective_profile=effective_profile,
        normalized_strategy=normalized_strategy,
        signal_kwargs=signal_kwargs,
        filter_config=filter_config,
        target_ladder=target_ladder,
    )


def _maybe_save_profile(*, params: dict[str, Any], effective_profile: Any, console: Console) -> None:
    if params["save_profile"] is None:
        return

    from options_helper.data.strategy_modeling_profiles import (
        StrategyModelingProfileStoreError,
        save_strategy_modeling_profile,
    )

    try:
        save_strategy_modeling_profile(
            Path(params["profile_path"]),
            str(params["save_profile"]),
            effective_profile,
            overwrite=bool(params["overwrite_profile"]),
        )
    except StrategyModelingProfileStoreError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"Saved strategy-modeling profile '{params['save_profile']}' -> {params['profile_path']}")


def _resolve_profile_state(
    *,
    ctx: typer.Context,
    params: dict[str, Any],
    validation: _ValidatedInputs,
    console: Console,
) -> _ProfileState:
    from pydantic import ValidationError

    loaded_profile = _load_profile(params=params)
    cli_profile_values = build_cli_profile_values(params=params, validation=validation)
    try:
        effective_profile = _merge_strategy_modeling_profile_values(
            ctx=ctx,
            loaded_profile=loaded_profile,
            cli_values=cli_profile_values,
        )
    except ValidationError as exc:
        detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
        raise typer.BadParameter(f"Invalid strategy-modeling profile values: {detail}") from exc

    if bool(params["disable_max_hold"]):
        effective_profile = effective_profile.model_copy(update={"max_hold_bars": None})
    if bool(params["disable_stop_moves"]):
        effective_profile = effective_profile.model_copy(update={"stop_move_rules": ()})

    _maybe_save_profile(params=params, effective_profile=effective_profile, console=console)
    return _build_profile_state(params=params, effective_profile=effective_profile)


def _build_request(
    *,
    params: dict[str, Any],
    service: Any,
    profile_state: _ProfileState,
    validation: _ValidatedInputs,
) -> tuple[Any, tuple[str, ...]]:
    from options_helper.analysis.strategy_modeling import StrategyModelingRequest

    excluded_symbols = set(_split_csv_option(params["exclude_symbols"]))
    resolved_symbols = _resolve_strategy_symbols(
        service=service,
        requested_symbols=list(profile_state.effective_profile.symbols),
        excluded_symbols=excluded_symbols,
        universe_limit=params["universe_limit"],
    )
    if not resolved_symbols:
        raise typer.BadParameter("No symbols remain after applying include/exclude/universe filters.")

    base_request = StrategyModelingRequest(
        strategy=profile_state.normalized_strategy,
        symbols=resolved_symbols,
        start_date=profile_state.effective_profile.start_date,
        end_date=profile_state.effective_profile.end_date,
        intraday_timeframe=profile_state.effective_profile.intraday_timeframe,
        target_ladder=profile_state.target_ladder,
        starting_capital=float(profile_state.effective_profile.starting_capital),
        max_hold_bars=profile_state.effective_profile.max_hold_bars,
        filter_config=profile_state.filter_config,
        signal_kwargs=profile_state.signal_kwargs,
        policy={
            "require_intraday_bars": True,
            "sizing_rule": "risk_pct_of_equity",
            "risk_per_trade_pct": float(profile_state.effective_profile.risk_per_trade_pct),
            "gap_fill_policy": profile_state.effective_profile.gap_fill_policy,
            "max_hold_bars": profile_state.effective_profile.max_hold_bars,
            "max_hold_timeframe": profile_state.effective_profile.max_hold_timeframe,
            "one_open_per_symbol": bool(profile_state.effective_profile.one_open_per_symbol),
            "stop_move_rules": profile_state.effective_profile.stop_move_rules,
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
    )
    request = SimpleNamespace(
        **vars(base_request),
        intraday_source=profile_state.effective_profile.intraday_source,
        gap_fill_policy=profile_state.effective_profile.gap_fill_policy,
        max_hold_timeframe=profile_state.effective_profile.max_hold_timeframe,
        intra_bar_tie_break_rule="stop_first",
        signal_confirmation_lag_bars=params["signal_confirmation_lag_bars"],
        output_timezone=validation.normalized_output_timezone,
        segment_dimensions=tuple(_split_csv_option(params["segment_dimensions"], uppercase=False)),
        segment_values=tuple(_split_csv_option(params["segment_values"], uppercase=False)),
        segment_min_trades=int(params["segment_min_trades"]),
        segment_limit=int(params["segment_limit"]),
    )
    return request, resolved_symbols


def run_strategy_model_command(*, params: dict[str, Any]) -> None:
    from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts

    ctx = params.get("ctx")
    if not isinstance(ctx, ClickContext):
        raise typer.BadParameter("Strategy-model command context is unavailable.")

    console = Console(width=200)
    validation = _validate_cli_inputs(ctx=ctx, params=params)
    profile_state = _resolve_profile_state(
        ctx=ctx,
        params=params,
        validation=validation,
        console=console,
    )

    tracker = StageTracker(console=console, show_progress=bool(params["show_progress"]))
    service = instrument_service(service=cli_deps.build_strategy_modeling_service(), tracker=tracker)
    request, resolved_symbols = _build_request(
        params=params,
        service=service,
        profile_state=profile_state,
        validation=validation,
    )
    run_result, _run_elapsed = run_strategy_model(
        service=service,
        request=request,
        strategy=profile_state.normalized_strategy,
        symbol_count=len(resolved_symbols),
        intraday_timeframe=profile_state.effective_profile.intraday_timeframe,
        tracker=tracker,
    )

    paths = write_strategy_modeling_artifacts(
        out_dir=Path(params["out"]),
        strategy=profile_state.normalized_strategy,
        request=request,
        run_result=run_result,
        write_json=bool(params["write_json"]),
        write_csv=bool(params["write_csv"]),
        write_md=bool(params["write_md"]),
    )
    segment_count = len(tuple(getattr(run_result, "segment_records", ()) or ()))
    segments_shown = min(segment_count, int(params["segment_limit"]))
    trade_count = len(tuple(getattr(run_result, "trade_simulations", ()) or ()))
    console.print(
        f"strategy={profile_state.normalized_strategy} symbols={len(resolved_symbols)} "
        f"trades={trade_count} segments_shown={segments_shown}"
    )

    emit_filter_summary(console=console, run_result=run_result)
    emit_directional_summary(console=console, run_result=run_result)
    emit_artifact_paths(
        console=console,
        write_json=bool(params["write_json"]),
        write_csv=bool(params["write_csv"]),
        write_md=bool(params["write_md"]),
        paths=paths,
    )
    emit_stage_timings(tracker=tracker)
