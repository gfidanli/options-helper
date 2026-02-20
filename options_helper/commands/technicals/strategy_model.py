from __future__ import annotations

from pathlib import Path

import typer

from .strategy_model_runtime import run_strategy_model_command

_STRATEGY_OPT = typer.Option(
        "sfp",
        "--strategy",
        help="Strategy to model: sfp, msb, orb, ma_crossover, trend_following, or fib_retracement.",
    )

_PROFILE_OPT = typer.Option(
        None,
        "--profile",
        help="Load a named strategy-modeling profile from --profile-path.",
    )

_SAVE_PROFILE_OPT = typer.Option(
        None,
        "--save-profile",
        help="Save effective strategy-modeling inputs under this profile name.",
    )

_OVERWRITE_PROFILE_OPT = typer.Option(
        False,
        "--overwrite-profile/--no-overwrite-profile",
        help="Allow --save-profile to overwrite an existing profile name.",
    )

_PROFILE_PATH_OPT = typer.Option(
        Path("config/strategy_modeling_profiles.json"),
        "--profile-path",
        help="Strategy-modeling profile store path (JSON).",
    )

_ALLOW_SHORTS_OPT = typer.Option(
        True,
        "--allow-shorts/--no-allow-shorts",
        help="Allow short-direction events to pass entry filters.",
    )

_ENABLE_ORB_CONFIRMATION_OPT = typer.Option(
        False,
        "--enable-orb-confirmation/--no-enable-orb-confirmation",
        help="Require ORB breakout confirmation by cutoff for SFP/MSB events.",
    )

_ORB_RANGE_MINUTES_OPT = typer.Option(
        15,
        "--orb-range-minutes",
        help="Opening-range window in minutes for ORB confirmation.",
    )

_ORB_CONFIRMATION_CUTOFF_ET_OPT = typer.Option(
        "10:30",
        "--orb-confirmation-cutoff-et",
        help="Cutoff for ORB confirmation in ET (HH:MM, 24-hour).",
    )

_ORB_STOP_POLICY_OPT = typer.Option(
        "base",
        "--orb-stop-policy",
        help="ORB stop policy: base, orb_range, or tighten.",
    )

_ENABLE_ATR_STOP_FLOOR_OPT = typer.Option(
        False,
        "--enable-atr-stop-floor/--no-enable-atr-stop-floor",
        help="Apply ATR stop floor gate to entries.",
    )

_ATR_STOP_FLOOR_MULTIPLE_OPT = typer.Option(
        0.5,
        "--atr-stop-floor-multiple",
        help="ATR multiple required for ATR stop floor gate.",
    )

_ENABLE_RSI_EXTREMES_OPT = typer.Option(
        False,
        "--enable-rsi-extremes/--no-enable-rsi-extremes",
        help="Require RSI extreme context for entries.",
    )

_ENABLE_EMA9_REGIME_OPT = typer.Option(
        False,
        "--enable-ema9-regime/--no-enable-ema9-regime",
        help="Require EMA9 regime alignment for entries.",
    )

_EMA9_SLOPE_LOOKBACK_BARS_OPT = typer.Option(
        3,
        "--ema9-slope-lookback-bars",
        help="Lookback bars for EMA9 slope regime computation.",
    )

_ENABLE_VOLATILITY_REGIME_OPT = typer.Option(
        False,
        "--enable-volatility-regime/--no-enable-volatility-regime",
        help="Enable volatility-regime allowlist filtering.",
    )

_ALLOWED_VOLATILITY_REGIMES_OPT = typer.Option(
        "low,normal,high",
        "--allowed-volatility-regimes",
        help="Comma-separated volatility regimes to allow: low,normal,high.",
    )

_MA_FAST_WINDOW_OPT = typer.Option(
        20,
        "--ma-fast-window",
        help="Fast MA window for ma_crossover/trend_following signal generation.",
    )

_MA_SLOW_WINDOW_OPT = typer.Option(
        50,
        "--ma-slow-window",
        help="Slow MA window for ma_crossover signal generation.",
    )

_MA_TREND_WINDOW_OPT = typer.Option(
        200,
        "--ma-trend-window",
        help="Trend MA window for trend_following signal generation.",
    )

_MA_FAST_TYPE_OPT = typer.Option(
        "sma",
        "--ma-fast-type",
        help="Fast MA type for strategy signal generation: sma or ema.",
    )

_MA_SLOW_TYPE_OPT = typer.Option(
        "sma",
        "--ma-slow-type",
        help="Slow MA type for strategy signal generation: sma or ema.",
    )

_MA_TREND_TYPE_OPT = typer.Option(
        "sma",
        "--ma-trend-type",
        help="Trend MA type for strategy signal generation: sma or ema.",
    )

_TREND_SLOPE_LOOKBACK_BARS_OPT = typer.Option(
        3,
        "--trend-slope-lookback-bars",
        help="Lookback bars used for trend MA slope checks in trend_following.",
    )

_ATR_WINDOW_OPT = typer.Option(
        14,
        "--atr-window",
        help="ATR window used by ma_crossover/trend_following signal stops.",
    )

_ATR_STOP_MULTIPLE_OPT = typer.Option(
        2.0,
        "--atr-stop-multiple",
        help="ATR stop multiple used by ma_crossover/trend_following signal stops.",
    )

_FIB_RETRACEMENT_PCT_OPT = typer.Option(
        61.8,
        "--fib-retracement-pct",
        help="Fib retracement percent (e.g., 61.8; accepts 0.618 as ratio).",
    )

_SYMBOLS_OPT = typer.Option(
        None,
        "--symbols",
        help="Comma-separated symbols (if omitted, uses universe loader).",
    )

_EXCLUDE_SYMBOLS_OPT = typer.Option(
        None,
        "--exclude-symbols",
        help="Comma-separated symbols to exclude from modeled universe.",
    )

_UNIVERSE_LIMIT_OPT = typer.Option(
        None,
        "--universe-limit",
        help="Optional max number of resolved universe symbols.",
    )

_START_DATE_OPT = typer.Option(
        None,
        "--start-date",
        help="Optional ISO start date (YYYY-MM-DD).",
    )

_END_DATE_OPT = typer.Option(
        None,
        "--end-date",
        help="Optional ISO end date (YYYY-MM-DD).",
    )

_INTRADAY_TIMEFRAME_OPT = typer.Option(
        "5Min",
        "--intraday-timeframe",
        help="Required intraday bar timeframe for simulation.",
    )

_INTRADAY_SOURCE_OPT = typer.Option(
        "alpaca",
        "--intraday-source",
        help="Intraday source label for service request metadata.",
    )

_R_LADDER_MIN_TENTHS_OPT = typer.Option(
        10,
        "--r-ladder-min-tenths",
        help="Minimum target R in tenths (10 => 1.0R).",
    )

_R_LADDER_MAX_TENTHS_OPT = typer.Option(
        20,
        "--r-ladder-max-tenths",
        help="Maximum target R in tenths (20 => 2.0R).",
    )

_R_LADDER_STEP_TENTHS_OPT = typer.Option(
        1,
        "--r-ladder-step-tenths",
        help="Step size for target ladder in tenths.",
    )

_STARTING_CAPITAL_OPT = typer.Option(
        100000.0,
        "--starting-capital",
        help="Starting portfolio capital for strategy modeling.",
    )

_RISK_PER_TRADE_PCT_OPT = typer.Option(
        1.0,
        "--risk-per-trade-pct",
        help="Per-trade risk percent used with risk_pct_of_equity sizing.",
    )

_STOP_MOVE_RULES_OPT = typer.Option(
        [],
        "--stop-move",
        help=(
            "Repeatable stop adjustment rule expressed as 'trigger_r:stop_r' in R multiples "
            "(evaluated on bar close, applied starting next bar). Example: --stop-move 1.0:0.0 "
            "to move to breakeven after a +1.0R close."
        ),
    )

_DISABLE_STOP_MOVES_OPT = typer.Option(
        False,
        "--disable-stop-moves",
        help="Disable stop-move rules even if a loaded profile includes them.",
    )

_GAP_FILL_POLICY_OPT = typer.Option(
        "fill_at_open",
        "--gap-fill-policy",
        help="Gap fill policy for stop/target-through-open fills.",
    )

_MAX_HOLD_BARS_OPT = typer.Option(
        None,
        "--max-hold-bars",
        help="Optional max hold duration in bars for simulated trades.",
    )

_MAX_HOLD_TIMEFRAME_OPT = typer.Option(
        "entry",
        "--max-hold-timeframe",
        help="Timeframe unit for max hold bars (entry, 10Min, 1H, 1D, 1W).",
    )

_DISABLE_MAX_HOLD_OPT = typer.Option(
        False,
        "--disable-max-hold",
        help="Disable max-hold time stop even if profile values set one.",
    )

_ONE_OPEN_PER_SYMBOL_OPT = typer.Option(
        True,
        "--one-open-per-symbol/--no-one-open-per-symbol",
        help="Allow at most one open trade per symbol at a time.",
    )

_SIGNAL_CONFIRMATION_LAG_BARS_OPT = typer.Option(
        None,
        "--signal-confirmation-lag-bars",
        help="Optional right-side confirmation lag metadata for close-confirmed signals.",
    )

_SEGMENT_DIMENSIONS_OPT = typer.Option(
        None,
        "--segment-dimensions",
        help="Comma-separated segment dimensions to display first.",
    )

_SEGMENT_VALUES_OPT = typer.Option(
        None,
        "--segment-values",
        help="Comma-separated segment values to prioritize in output.",
    )

_SEGMENT_MIN_TRADES_OPT = typer.Option(
        1,
        "--segment-min-trades",
        help="Minimum trades required for segment display filters.",
    )

_SEGMENT_LIMIT_OPT = typer.Option(
        10,
        "--segment-limit",
        help="Maximum segments to summarize in CLI output.",
    )

_OUTPUT_TIMEZONE_OPT = typer.Option(
        "America/Chicago",
        "--output-timezone",
        help="Timezone used for serialized report timestamps (IANA name, e.g. America/Chicago).",
    )

_OUT_OPT = typer.Option(
        Path("data/reports/technicals/strategy_modeling"),
        "--out",
        help="Output root for strategy-modeling artifacts.",
    )

_WRITE_JSON_OPT = typer.Option(True, "--write-json/--no-write-json", help="Write summary JSON.")

_WRITE_CSV_OPT = typer.Option(True, "--write-csv/--no-write-csv", help="Write CSV artifacts.")

_WRITE_MD_OPT = typer.Option(True, "--write-md/--no-write-md", help="Write markdown summary.")

_SHOW_PROGRESS_OPT = typer.Option(
        True,
        "--show-progress/--no-show-progress",
        help="Print stage status updates while strategy modeling runs.",
    )


def technicals_strategy_model(
    ctx: typer.Context,
    strategy: str = _STRATEGY_OPT,
    profile: str | None = _PROFILE_OPT,
    save_profile: str | None = _SAVE_PROFILE_OPT,
    overwrite_profile: bool = _OVERWRITE_PROFILE_OPT,
    profile_path: Path = _PROFILE_PATH_OPT,
    allow_shorts: bool = _ALLOW_SHORTS_OPT,
    enable_orb_confirmation: bool = _ENABLE_ORB_CONFIRMATION_OPT,
    orb_range_minutes: int = _ORB_RANGE_MINUTES_OPT,
    orb_confirmation_cutoff_et: str = _ORB_CONFIRMATION_CUTOFF_ET_OPT,
    orb_stop_policy: str = _ORB_STOP_POLICY_OPT,
    enable_atr_stop_floor: bool = _ENABLE_ATR_STOP_FLOOR_OPT,
    atr_stop_floor_multiple: float = _ATR_STOP_FLOOR_MULTIPLE_OPT,
    enable_rsi_extremes: bool = _ENABLE_RSI_EXTREMES_OPT,
    enable_ema9_regime: bool = _ENABLE_EMA9_REGIME_OPT,
    ema9_slope_lookback_bars: int = _EMA9_SLOPE_LOOKBACK_BARS_OPT,
    enable_volatility_regime: bool = _ENABLE_VOLATILITY_REGIME_OPT,
    allowed_volatility_regimes: str = _ALLOWED_VOLATILITY_REGIMES_OPT,
    ma_fast_window: int = _MA_FAST_WINDOW_OPT,
    ma_slow_window: int = _MA_SLOW_WINDOW_OPT,
    ma_trend_window: int = _MA_TREND_WINDOW_OPT,
    ma_fast_type: str = _MA_FAST_TYPE_OPT,
    ma_slow_type: str = _MA_SLOW_TYPE_OPT,
    ma_trend_type: str = _MA_TREND_TYPE_OPT,
    trend_slope_lookback_bars: int = _TREND_SLOPE_LOOKBACK_BARS_OPT,
    atr_window: int = _ATR_WINDOW_OPT,
    atr_stop_multiple: float = _ATR_STOP_MULTIPLE_OPT,
    fib_retracement_pct: float = _FIB_RETRACEMENT_PCT_OPT,
    symbols: str | None = _SYMBOLS_OPT,
    exclude_symbols: str | None = _EXCLUDE_SYMBOLS_OPT,
    universe_limit: int | None = _UNIVERSE_LIMIT_OPT,
    start_date: str | None = _START_DATE_OPT,
    end_date: str | None = _END_DATE_OPT,
    intraday_timeframe: str = _INTRADAY_TIMEFRAME_OPT,
    intraday_source: str = _INTRADAY_SOURCE_OPT,
    r_ladder_min_tenths: int = _R_LADDER_MIN_TENTHS_OPT,
    r_ladder_max_tenths: int = _R_LADDER_MAX_TENTHS_OPT,
    r_ladder_step_tenths: int = _R_LADDER_STEP_TENTHS_OPT,
    starting_capital: float = _STARTING_CAPITAL_OPT,
    risk_per_trade_pct: float = _RISK_PER_TRADE_PCT_OPT,
    stop_move_rules: list[str] = _STOP_MOVE_RULES_OPT,
    disable_stop_moves: bool = _DISABLE_STOP_MOVES_OPT,
    gap_fill_policy: str = _GAP_FILL_POLICY_OPT,
    max_hold_bars: int | None = _MAX_HOLD_BARS_OPT,
    max_hold_timeframe: str = _MAX_HOLD_TIMEFRAME_OPT,
    disable_max_hold: bool = _DISABLE_MAX_HOLD_OPT,
    one_open_per_symbol: bool = _ONE_OPEN_PER_SYMBOL_OPT,
    signal_confirmation_lag_bars: int | None = _SIGNAL_CONFIRMATION_LAG_BARS_OPT,
    segment_dimensions: str | None = _SEGMENT_DIMENSIONS_OPT,
    segment_values: str | None = _SEGMENT_VALUES_OPT,
    segment_min_trades: int = _SEGMENT_MIN_TRADES_OPT,
    segment_limit: int = _SEGMENT_LIMIT_OPT,
    output_timezone: str = _OUTPUT_TIMEZONE_OPT,
    out: Path = _OUT_OPT,
    write_json: bool = _WRITE_JSON_OPT,
    write_csv: bool = _WRITE_CSV_OPT,
    write_md: bool = _WRITE_MD_OPT,
    show_progress: bool = _SHOW_PROGRESS_OPT,
) -> None:
    """Run strategy-modeling service and write JSON/CSV/Markdown artifacts."""
    run_strategy_model_command(params=dict(locals()))


def register(app: typer.Typer) -> None:
    app.command("strategy-model")(technicals_strategy_model)


__all__ = ["register", "technicals_strategy_model"]
