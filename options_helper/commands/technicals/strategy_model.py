from __future__ import annotations

from pathlib import Path
import time
from types import SimpleNamespace

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.commands.technicals.strategy_model_helpers_legacy import (
    _build_strategy_filter_config,
    _build_strategy_signal_kwargs,
    _build_target_ladder,
    _coerce_int,
    _extract_directional_headline,
    _intraday_coverage_block_message,
    _mapping_view,
    _merge_strategy_modeling_profile_values,
    _normalize_output_timezone,
    _option_was_set_on_command_line,
    _parse_allowed_volatility_regimes,
    _parse_iso_date,
    _resolve_strategy_symbols,
    _split_csv_option,
)


def technicals_strategy_model(
    ctx: typer.Context,
    strategy: str = typer.Option(
        "sfp",
        "--strategy",
        help="Strategy to model: sfp, msb, orb, ma_crossover, trend_following, or fib_retracement.",
    ),
    profile: str | None = typer.Option(
        None,
        "--profile",
        help="Load a named strategy-modeling profile from --profile-path.",
    ),
    save_profile: str | None = typer.Option(
        None,
        "--save-profile",
        help="Save effective strategy-modeling inputs under this profile name.",
    ),
    overwrite_profile: bool = typer.Option(
        False,
        "--overwrite-profile/--no-overwrite-profile",
        help="Allow --save-profile to overwrite an existing profile name.",
    ),
    profile_path: Path = typer.Option(
        Path("config/strategy_modeling_profiles.json"),
        "--profile-path",
        help="Strategy-modeling profile store path (JSON).",
    ),
    allow_shorts: bool = typer.Option(
        True,
        "--allow-shorts/--no-allow-shorts",
        help="Allow short-direction events to pass entry filters.",
    ),
    enable_orb_confirmation: bool = typer.Option(
        False,
        "--enable-orb-confirmation/--no-enable-orb-confirmation",
        help="Require ORB breakout confirmation by cutoff for SFP/MSB events.",
    ),
    orb_range_minutes: int = typer.Option(
        15,
        "--orb-range-minutes",
        help="Opening-range window in minutes for ORB confirmation.",
    ),
    orb_confirmation_cutoff_et: str = typer.Option(
        "10:30",
        "--orb-confirmation-cutoff-et",
        help="Cutoff for ORB confirmation in ET (HH:MM, 24-hour).",
    ),
    orb_stop_policy: str = typer.Option(
        "base",
        "--orb-stop-policy",
        help="ORB stop policy: base, orb_range, or tighten.",
    ),
    enable_atr_stop_floor: bool = typer.Option(
        False,
        "--enable-atr-stop-floor/--no-enable-atr-stop-floor",
        help="Apply ATR stop floor gate to entries.",
    ),
    atr_stop_floor_multiple: float = typer.Option(
        0.5,
        "--atr-stop-floor-multiple",
        help="ATR multiple required for ATR stop floor gate.",
    ),
    enable_rsi_extremes: bool = typer.Option(
        False,
        "--enable-rsi-extremes/--no-enable-rsi-extremes",
        help="Require RSI extreme context for entries.",
    ),
    enable_ema9_regime: bool = typer.Option(
        False,
        "--enable-ema9-regime/--no-enable-ema9-regime",
        help="Require EMA9 regime alignment for entries.",
    ),
    ema9_slope_lookback_bars: int = typer.Option(
        3,
        "--ema9-slope-lookback-bars",
        help="Lookback bars for EMA9 slope regime computation.",
    ),
    enable_volatility_regime: bool = typer.Option(
        False,
        "--enable-volatility-regime/--no-enable-volatility-regime",
        help="Enable volatility-regime allowlist filtering.",
    ),
    allowed_volatility_regimes: str = typer.Option(
        "low,normal,high",
        "--allowed-volatility-regimes",
        help="Comma-separated volatility regimes to allow: low,normal,high.",
    ),
    ma_fast_window: int = typer.Option(
        20,
        "--ma-fast-window",
        help="Fast MA window for ma_crossover/trend_following signal generation.",
    ),
    ma_slow_window: int = typer.Option(
        50,
        "--ma-slow-window",
        help="Slow MA window for ma_crossover signal generation.",
    ),
    ma_trend_window: int = typer.Option(
        200,
        "--ma-trend-window",
        help="Trend MA window for trend_following signal generation.",
    ),
    ma_fast_type: str = typer.Option(
        "sma",
        "--ma-fast-type",
        help="Fast MA type for strategy signal generation: sma or ema.",
    ),
    ma_slow_type: str = typer.Option(
        "sma",
        "--ma-slow-type",
        help="Slow MA type for strategy signal generation: sma or ema.",
    ),
    ma_trend_type: str = typer.Option(
        "sma",
        "--ma-trend-type",
        help="Trend MA type for strategy signal generation: sma or ema.",
    ),
    trend_slope_lookback_bars: int = typer.Option(
        3,
        "--trend-slope-lookback-bars",
        help="Lookback bars used for trend MA slope checks in trend_following.",
    ),
    atr_window: int = typer.Option(
        14,
        "--atr-window",
        help="ATR window used by ma_crossover/trend_following signal stops.",
    ),
    atr_stop_multiple: float = typer.Option(
        2.0,
        "--atr-stop-multiple",
        help="ATR stop multiple used by ma_crossover/trend_following signal stops.",
    ),
    fib_retracement_pct: float = typer.Option(
        61.8,
        "--fib-retracement-pct",
        help="Fib retracement percent (e.g., 61.8; accepts 0.618 as ratio).",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        help="Comma-separated symbols (if omitted, uses universe loader).",
    ),
    exclude_symbols: str | None = typer.Option(
        None,
        "--exclude-symbols",
        help="Comma-separated symbols to exclude from modeled universe.",
    ),
    universe_limit: int | None = typer.Option(
        None,
        "--universe-limit",
        help="Optional max number of resolved universe symbols.",
    ),
    start_date: str | None = typer.Option(
        None,
        "--start-date",
        help="Optional ISO start date (YYYY-MM-DD).",
    ),
    end_date: str | None = typer.Option(
        None,
        "--end-date",
        help="Optional ISO end date (YYYY-MM-DD).",
    ),
    intraday_timeframe: str = typer.Option(
        "5Min",
        "--intraday-timeframe",
        help="Required intraday bar timeframe for simulation.",
    ),
    intraday_source: str = typer.Option(
        "alpaca",
        "--intraday-source",
        help="Intraday source label for service request metadata.",
    ),
    r_ladder_min_tenths: int = typer.Option(
        10,
        "--r-ladder-min-tenths",
        help="Minimum target R in tenths (10 => 1.0R).",
    ),
    r_ladder_max_tenths: int = typer.Option(
        20,
        "--r-ladder-max-tenths",
        help="Maximum target R in tenths (20 => 2.0R).",
    ),
    r_ladder_step_tenths: int = typer.Option(
        1,
        "--r-ladder-step-tenths",
        help="Step size for target ladder in tenths.",
    ),
    starting_capital: float = typer.Option(
        100000.0,
        "--starting-capital",
        help="Starting portfolio capital for strategy modeling.",
    ),
    risk_per_trade_pct: float = typer.Option(
        1.0,
        "--risk-per-trade-pct",
        help="Per-trade risk percent used with risk_pct_of_equity sizing.",
    ),
    stop_move_rules: list[str] = typer.Option(
        [],
        "--stop-move",
        help=(
            "Repeatable stop adjustment rule expressed as 'trigger_r:stop_r' in R multiples "
            "(evaluated on bar close, applied starting next bar). Example: --stop-move 1.0:0.0 "
            "to move to breakeven after a +1.0R close."
        ),
    ),
    disable_stop_moves: bool = typer.Option(
        False,
        "--disable-stop-moves",
        help="Disable stop-move rules even if a loaded profile includes them.",
    ),
    gap_fill_policy: str = typer.Option(
        "fill_at_open",
        "--gap-fill-policy",
        help="Gap fill policy for stop/target-through-open fills.",
    ),
    max_hold_bars: int | None = typer.Option(
        None,
        "--max-hold-bars",
        help="Optional max hold duration in bars for simulated trades.",
    ),
    max_hold_timeframe: str = typer.Option(
        "entry",
        "--max-hold-timeframe",
        help="Timeframe unit for max hold bars (entry, 10Min, 1H, 1D, 1W).",
    ),
    disable_max_hold: bool = typer.Option(
        False,
        "--disable-max-hold",
        help="Disable max-hold time stop even if profile values set one.",
    ),
    one_open_per_symbol: bool = typer.Option(
        True,
        "--one-open-per-symbol/--no-one-open-per-symbol",
        help="Allow at most one open trade per symbol at a time.",
    ),
    signal_confirmation_lag_bars: int | None = typer.Option(
        None,
        "--signal-confirmation-lag-bars",
        help="Optional right-side confirmation lag metadata for close-confirmed signals.",
    ),
    segment_dimensions: str | None = typer.Option(
        None,
        "--segment-dimensions",
        help="Comma-separated segment dimensions to display first.",
    ),
    segment_values: str | None = typer.Option(
        None,
        "--segment-values",
        help="Comma-separated segment values to prioritize in output.",
    ),
    segment_min_trades: int = typer.Option(
        1,
        "--segment-min-trades",
        help="Minimum trades required for segment display filters.",
    ),
    segment_limit: int = typer.Option(
        10,
        "--segment-limit",
        help="Maximum segments to summarize in CLI output.",
    ),
    output_timezone: str = typer.Option(
        "America/Chicago",
        "--output-timezone",
        help="Timezone used for serialized report timestamps (IANA name, e.g. America/Chicago).",
    ),
    out: Path = typer.Option(
        Path("data/reports/technicals/strategy_modeling"),
        "--out",
        help="Output root for strategy-modeling artifacts.",
    ),
    write_json: bool = typer.Option(True, "--write-json/--no-write-json", help="Write summary JSON."),
    write_csv: bool = typer.Option(True, "--write-csv/--no-write-csv", help="Write CSV artifacts."),
    write_md: bool = typer.Option(True, "--write-md/--no-write-md", help="Write markdown summary."),
    show_progress: bool = typer.Option(
        True,
        "--show-progress/--no-show-progress",
        help="Print stage status updates while strategy modeling runs.",
    ),
) -> None:
    """Run strategy-modeling service and write JSON/CSV/Markdown artifacts."""
    from pydantic import ValidationError

    from options_helper.analysis.strategy_modeling import StrategyModelingRequest
    from options_helper.data.strategy_modeling_artifacts import write_strategy_modeling_artifacts
    from options_helper.data.strategy_modeling_profiles import (
        StrategyModelingProfileStoreError,
        load_strategy_modeling_profile,
        save_strategy_modeling_profile,
    )
    from options_helper.schemas.strategy_modeling_policy import normalize_max_hold_timeframe

    console = Console(width=200)

    normalized_cli_strategy = strategy.strip().lower()
    if normalized_cli_strategy not in {
        "sfp",
        "msb",
        "orb",
        "ma_crossover",
        "trend_following",
        "fib_retracement",
    }:
        raise typer.BadParameter(
            "--strategy must be one of: sfp, msb, orb, ma_crossover, trend_following, fib_retracement"
        )
    if starting_capital <= 0.0:
        raise typer.BadParameter("--starting-capital must be > 0")
    if risk_per_trade_pct <= 0.0 or risk_per_trade_pct > 100.0:
        raise typer.BadParameter("--risk-per-trade-pct must be > 0 and <= 100")
    if gap_fill_policy != "fill_at_open":
        raise typer.BadParameter("--gap-fill-policy currently supports only fill_at_open")
    if max_hold_bars is not None and int(max_hold_bars) < 1:
        raise typer.BadParameter("--max-hold-bars must be >= 1")
    try:
        normalized_max_hold_timeframe = normalize_max_hold_timeframe(max_hold_timeframe)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if disable_max_hold and _option_was_set_on_command_line(ctx, "max_hold_bars"):
        raise typer.BadParameter("--disable-max-hold cannot be combined with --max-hold-bars")
    if segment_min_trades < 1:
        raise typer.BadParameter("--segment-min-trades must be >= 1")
    if segment_limit < 1:
        raise typer.BadParameter("--segment-limit must be >= 1")
    if universe_limit is not None and universe_limit < 1:
        raise typer.BadParameter("--universe-limit must be >= 1")
    normalized_output_timezone = _normalize_output_timezone(output_timezone)

    parsed_cli_start_date = _parse_iso_date(start_date, option_name="--start-date")
    parsed_cli_end_date = _parse_iso_date(end_date, option_name="--end-date")
    if parsed_cli_start_date and parsed_cli_end_date and parsed_cli_start_date > parsed_cli_end_date:
        raise typer.BadParameter("--start-date must be <= --end-date")
    # Preserve option-specific error messages for explicit CLI validation before profile merging.
    _build_strategy_signal_kwargs(
        strategy=normalized_cli_strategy,
        fib_retracement_pct=float(fib_retracement_pct),
        ma_fast_window=int(ma_fast_window),
        ma_slow_window=int(ma_slow_window),
        ma_trend_window=int(ma_trend_window),
        ma_fast_type=ma_fast_type,
        ma_slow_type=ma_slow_type,
        ma_trend_type=ma_trend_type,
        trend_slope_lookback_bars=int(trend_slope_lookback_bars),
        atr_window=int(atr_window),
        atr_stop_multiple=float(atr_stop_multiple),
    )
    _build_strategy_filter_config(
        allow_shorts=allow_shorts,
        enable_orb_confirmation=enable_orb_confirmation,
        orb_range_minutes=int(orb_range_minutes),
        orb_confirmation_cutoff_et=orb_confirmation_cutoff_et,
        orb_stop_policy=orb_stop_policy,
        enable_atr_stop_floor=enable_atr_stop_floor,
        atr_stop_floor_multiple=float(atr_stop_floor_multiple),
        enable_rsi_extremes=enable_rsi_extremes,
        enable_ema9_regime=enable_ema9_regime,
        ema9_slope_lookback_bars=int(ema9_slope_lookback_bars),
        enable_volatility_regime=enable_volatility_regime,
        allowed_volatility_regimes=allowed_volatility_regimes,
    )
    parsed_cli_allowed_volatility_regimes = _parse_allowed_volatility_regimes(
        allowed_volatility_regimes,
        option_name="--allowed-volatility-regimes",
    )

    loaded_profile = None
    if profile is not None:
        try:
            loaded_profile = load_strategy_modeling_profile(profile_path, profile)
        except StrategyModelingProfileStoreError as exc:
            raise typer.BadParameter(str(exc)) from exc

    cli_profile_values: dict[str, object] = {
        "strategy": normalized_cli_strategy,
        "symbols": tuple(_split_csv_option(symbols)),
        "start_date": parsed_cli_start_date,
        "end_date": parsed_cli_end_date,
        "intraday_timeframe": str(intraday_timeframe),
        "intraday_source": str(intraday_source),
        "starting_capital": float(starting_capital),
        "risk_per_trade_pct": float(risk_per_trade_pct),
        "fib_retracement_pct": float(fib_retracement_pct),
        "gap_fill_policy": str(gap_fill_policy),
        "max_hold_bars": int(max_hold_bars) if max_hold_bars is not None else None,
        "max_hold_timeframe": normalized_max_hold_timeframe,
        "one_open_per_symbol": bool(one_open_per_symbol),
        "stop_move_rules": tuple(stop_move_rules),
        "r_ladder_min_tenths": int(r_ladder_min_tenths),
        "r_ladder_max_tenths": int(r_ladder_max_tenths),
        "r_ladder_step_tenths": int(r_ladder_step_tenths),
        "allow_shorts": bool(allow_shorts),
        "enable_orb_confirmation": bool(enable_orb_confirmation),
        "orb_range_minutes": int(orb_range_minutes),
        "orb_confirmation_cutoff_et": str(orb_confirmation_cutoff_et),
        "orb_stop_policy": str(orb_stop_policy),
        "enable_atr_stop_floor": bool(enable_atr_stop_floor),
        "atr_stop_floor_multiple": float(atr_stop_floor_multiple),
        "enable_rsi_extremes": bool(enable_rsi_extremes),
        "enable_ema9_regime": bool(enable_ema9_regime),
        "ema9_slope_lookback_bars": int(ema9_slope_lookback_bars),
        "enable_volatility_regime": bool(enable_volatility_regime),
        "allowed_volatility_regimes": parsed_cli_allowed_volatility_regimes,
        "ma_fast_window": int(ma_fast_window),
        "ma_slow_window": int(ma_slow_window),
        "ma_trend_window": int(ma_trend_window),
        "ma_fast_type": str(ma_fast_type),
        "ma_slow_type": str(ma_slow_type),
        "ma_trend_type": str(ma_trend_type),
        "trend_slope_lookback_bars": int(trend_slope_lookback_bars),
        "atr_window": int(atr_window),
        "atr_stop_multiple": float(atr_stop_multiple),
    }
    try:
        effective_profile = _merge_strategy_modeling_profile_values(
            ctx=ctx,
            loaded_profile=loaded_profile,
            cli_values=cli_profile_values,
        )
    except ValidationError as exc:
        detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
        raise typer.BadParameter(f"Invalid strategy-modeling profile values: {detail}") from exc
    if disable_max_hold:
        effective_profile = effective_profile.model_copy(update={"max_hold_bars": None})
    if disable_stop_moves:
        effective_profile = effective_profile.model_copy(update={"stop_move_rules": ()})

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

    if save_profile is not None:
        try:
            save_strategy_modeling_profile(
                profile_path,
                save_profile,
                effective_profile,
                overwrite=overwrite_profile,
            )
        except StrategyModelingProfileStoreError as exc:
            raise typer.BadParameter(str(exc)) from exc
        console.print(f"Saved strategy-modeling profile '{save_profile}' -> {profile_path}")

    service = cli_deps.build_strategy_modeling_service()
    stage_timings: dict[str, float] = {}

    def _format_seconds(value: float) -> str:
        return f"{value:.2f}s"

    def _accumulate_timing(name: str, elapsed: float) -> None:
        stage_timings[name] = stage_timings.get(name, 0.0) + elapsed

    def _safe_detail(detail_builder, output, args, kwargs) -> str:  # noqa: ANN001
        if detail_builder is None:
            return ""
        try:
            detail = detail_builder(output, args, kwargs)
        except Exception:  # noqa: BLE001
            return ""
        if not detail:
            return ""
        return f" | {detail}"

    def _timed_stage(name: str, fn, detail_builder=None):  # noqa: ANN001
        def _wrapped(*args, **kwargs):  # noqa: ANN002,ANN003
            if show_progress:
                console.print(f"[cyan]... {name}[/cyan]")
            started = time.perf_counter()
            output = fn(*args, **kwargs)
            elapsed = time.perf_counter() - started
            _accumulate_timing(name, elapsed)
            if show_progress:
                detail = _safe_detail(detail_builder, output, args, kwargs)
                console.print(f"[green]OK  {name} ({_format_seconds(elapsed)}){detail}[/green]")
            return output

        return _wrapped

    service_stage_attrs = (
        "list_universe_loader",
        "daily_loader",
        "required_sessions_builder",
        "intraday_loader",
        "feature_computer",
        "signal_builder",
        "trade_simulator",
        "portfolio_builder",
        "metrics_computer",
        "segmentation_aggregator",
    )
    if show_progress and all(hasattr(service, attr) for attr in service_stage_attrs):
        from options_helper.analysis.strategy_modeling import StrategyModelingService

        service = StrategyModelingService(
            list_universe_loader=_timed_stage(
                "Loading universe",
                service.list_universe_loader,
                lambda out, _a, _k: f"symbols={len(tuple(getattr(out, 'symbols', ()) or ())) or 0}",
            ),
            daily_loader=_timed_stage(
                "Loading daily candles",
                service.daily_loader,
                lambda out, _a, _k: (
                    f"loaded={len(getattr(out, 'candles_by_symbol', {}) or {})} "
                    f"skipped={len(tuple(getattr(out, 'skipped_symbols', ()) or ())) or 0} "
                    f"missing={len(tuple(getattr(out, 'missing_symbols', ()) or ())) or 0}"
                ),
            ),
            required_sessions_builder=_timed_stage(
                "Building required sessions",
                service.required_sessions_builder,
                lambda out, _a, _k: (
                    f"symbols={len(out or {})} "
                    f"sessions={sum(len(tuple(v or ())) for v in (out or {}).values())}"
                ),
            ),
            intraday_loader=_timed_stage(
                "Loading intraday bars",
                service.intraday_loader,
                lambda out, _a, _k: (
                    f"symbols={len(getattr(out, 'bars_by_symbol', {}) or {})} "
                    f"blocked={len(tuple(getattr(getattr(out, 'preflight', None), 'blocked_symbols', ()) or ())) or 0}"
                ),
            ),
            feature_computer=_timed_stage("Computing features", service.feature_computer),
            signal_builder=_timed_stage(
                "Building signals",
                service.signal_builder,
                lambda out, _a, kwargs: (
                    f"symbol={str(kwargs.get('symbol', '')).upper()} events={len(tuple(out or ())) or 0}"
                ),
            ),
            trade_simulator=_timed_stage(
                "Simulating trades",
                service.trade_simulator,
                lambda out, args, kwargs: (
                    f"events={len(tuple(args[0] or ())) if args else 0} "
                    f"targets={len(tuple(kwargs.get('target_ladder') or ())) or 0} "
                    f"trades={len(tuple(out or ())) or 0}"
                ),
            ),
            portfolio_builder=_timed_stage(
                "Building portfolio ledger",
                service.portfolio_builder,
                lambda out, _a, _k: (
                    f"accepted={len(tuple(getattr(out, 'accepted_trade_ids', ()) or ())) or 0} "
                    f"skipped={len(tuple(getattr(out, 'skipped_trade_ids', ()) or ())) or 0}"
                ),
            ),
            metrics_computer=_timed_stage("Computing metrics", service.metrics_computer),
            segmentation_aggregator=_timed_stage(
                "Building segmentation",
                service.segmentation_aggregator,
                lambda out, _a, _k: f"segments={len(tuple(getattr(out, 'segments', ()) or ())) or 0}",
            ),
        )

    resolved_symbols = _resolve_strategy_symbols(
        service=service,
        requested_symbols=list(effective_profile.symbols),
        excluded_symbols=set(_split_csv_option(exclude_symbols)),
        universe_limit=universe_limit,
    )
    if not resolved_symbols:
        raise typer.BadParameter("No symbols remain after applying include/exclude/universe filters.")

    base_request = StrategyModelingRequest(
        strategy=normalized_strategy,
        symbols=resolved_symbols,
        start_date=effective_profile.start_date,
        end_date=effective_profile.end_date,
        intraday_timeframe=effective_profile.intraday_timeframe,
        target_ladder=target_ladder,
        starting_capital=float(effective_profile.starting_capital),
        max_hold_bars=effective_profile.max_hold_bars,
        filter_config=filter_config,
        signal_kwargs=signal_kwargs,
        policy={
            "require_intraday_bars": True,
            "sizing_rule": "risk_pct_of_equity",
            "risk_per_trade_pct": float(effective_profile.risk_per_trade_pct),
            "gap_fill_policy": effective_profile.gap_fill_policy,
            "max_hold_bars": effective_profile.max_hold_bars,
            "max_hold_timeframe": effective_profile.max_hold_timeframe,
            "one_open_per_symbol": bool(effective_profile.one_open_per_symbol),
            "stop_move_rules": effective_profile.stop_move_rules,
            "entry_ts_anchor_policy": "first_tradable_bar_open_after_signal_confirmed_ts",
        },
    )

    request = SimpleNamespace(
        **vars(base_request),
        intraday_source=effective_profile.intraday_source,
        gap_fill_policy=effective_profile.gap_fill_policy,
        max_hold_timeframe=effective_profile.max_hold_timeframe,
        intra_bar_tie_break_rule="stop_first",
        signal_confirmation_lag_bars=signal_confirmation_lag_bars,
        output_timezone=normalized_output_timezone,
        segment_dimensions=tuple(_split_csv_option(segment_dimensions, uppercase=False)),
        segment_values=tuple(_split_csv_option(segment_values, uppercase=False)),
        segment_min_trades=int(segment_min_trades),
        segment_limit=int(segment_limit),
    )

    if show_progress:
        console.print(
            (
                f"[cyan]Starting strategy-model run: strategy={normalized_strategy} "
                f"symbols={len(resolved_symbols)} timeframe={effective_profile.intraday_timeframe}[/cyan]"
            )
        )
    run_started = time.perf_counter()
    run_result = service.run(request)
    run_elapsed = time.perf_counter() - run_started
    if show_progress:
        console.print(f"[green]Run complete in {_format_seconds(run_elapsed)}[/green]")
    block_message = _intraday_coverage_block_message(getattr(run_result, "intraday_preflight", None))
    if block_message is not None:
        raise typer.BadParameter(block_message)

    paths = write_strategy_modeling_artifacts(
        out_dir=out,
        strategy=normalized_strategy,
        request=request,
        run_result=run_result,
        write_json=write_json,
        write_csv=write_csv,
        write_md=write_md,
    )

    segment_count = len(tuple(getattr(run_result, "segment_records", ()) or ()))
    segments_shown = min(segment_count, int(segment_limit))
    trade_count = len(tuple(getattr(run_result, "trade_simulations", ()) or ()))
    console.print(
        (
            f"strategy={normalized_strategy} symbols={len(resolved_symbols)} "
            f"trades={trade_count} segments_shown={segments_shown}"
        )
    )

    filter_summary = _mapping_view(getattr(run_result, "filter_summary", None))
    if filter_summary:
        base_events = _coerce_int(filter_summary.get("base_event_count"))
        kept_events = _coerce_int(filter_summary.get("kept_event_count"))
        rejected_events = _coerce_int(filter_summary.get("rejected_event_count"))
        if base_events is not None and kept_events is not None and rejected_events is not None:
            console.print(f"filters base={base_events} kept={kept_events} rejected={rejected_events}")

        reject_counts = _mapping_view(filter_summary.get("reject_counts"))
        reject_parts: list[str] = []
        for reason, count in reject_counts.items():
            parsed = _coerce_int(count)
            if parsed is None or parsed <= 0:
                continue
            reject_parts.append(f"{reason}={parsed}")
        if reject_parts:
            console.print(f"filter_rejects {', '.join(reject_parts)}")

    directional_metrics = _mapping_view(getattr(run_result, "directional_metrics", None))
    if directional_metrics:
        directional_parts: list[str] = []
        for label in ("combined", "long_only", "short_only"):
            trade_count_value, total_return_value = _extract_directional_headline(
                directional_metrics.get(label)
            )
            if trade_count_value is None and total_return_value is None:
                continue

            part = label
            if trade_count_value is not None:
                part = f"{part} trades={trade_count_value}"
            if total_return_value is not None:
                part = f"{part} return={total_return_value:.2f}%"
            directional_parts.append(part)
        if directional_parts:
            console.print(f"directional {' | '.join(directional_parts)}")

    if write_json:
        console.print(f"Wrote summary JSON: {paths.summary_json}")
    if write_csv:
        console.print(f"Wrote trades CSV: {paths.trade_log_csv}")
        console.print(f"Wrote R ladder CSV: {paths.r_ladder_csv}")
        console.print(f"Wrote segments CSV: {paths.segments_csv}")
    if write_md:
        console.print(f"Wrote summary Markdown: {paths.summary_md}")
        console.print(f"Wrote LLM analysis prompt: {paths.llm_analysis_prompt_md}")
    if show_progress and stage_timings:
        console.print("Stage timings:")
        for name in sorted(stage_timings, key=stage_timings.get, reverse=True):
            console.print(f"  - {name}: {_format_seconds(stage_timings[name])}")


def register(app: typer.Typer) -> None:
    app.command("strategy-model")(technicals_strategy_model)


__all__ = ["register", "technicals_strategy_model"]
