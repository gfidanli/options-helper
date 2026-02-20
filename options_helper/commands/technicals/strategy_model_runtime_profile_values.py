from __future__ import annotations

from typing import Any

from .strategy_model_helpers_legacy import _split_csv_option


def build_cli_profile_values(*, params: dict[str, Any], validation: Any) -> dict[str, object]:
    return {
        "strategy": validation.normalized_cli_strategy,
        "symbols": tuple(_split_csv_option(params["symbols"])),
        "start_date": validation.parsed_cli_start_date,
        "end_date": validation.parsed_cli_end_date,
        "intraday_timeframe": str(params["intraday_timeframe"]),
        "intraday_source": str(params["intraday_source"]),
        "starting_capital": float(params["starting_capital"]),
        "risk_per_trade_pct": float(params["risk_per_trade_pct"]),
        "fib_retracement_pct": float(params["fib_retracement_pct"]),
        "gap_fill_policy": str(params["gap_fill_policy"]),
        "max_hold_bars": int(params["max_hold_bars"]) if params["max_hold_bars"] is not None else None,
        "max_hold_timeframe": validation.normalized_max_hold_timeframe,
        "one_open_per_symbol": bool(params["one_open_per_symbol"]),
        "stop_move_rules": tuple(params["stop_move_rules"]),
        "r_ladder_min_tenths": int(params["r_ladder_min_tenths"]),
        "r_ladder_max_tenths": int(params["r_ladder_max_tenths"]),
        "r_ladder_step_tenths": int(params["r_ladder_step_tenths"]),
        "allow_shorts": bool(params["allow_shorts"]),
        "enable_orb_confirmation": bool(params["enable_orb_confirmation"]),
        "orb_range_minutes": int(params["orb_range_minutes"]),
        "orb_confirmation_cutoff_et": str(params["orb_confirmation_cutoff_et"]),
        "orb_stop_policy": str(params["orb_stop_policy"]),
        "enable_atr_stop_floor": bool(params["enable_atr_stop_floor"]),
        "atr_stop_floor_multiple": float(params["atr_stop_floor_multiple"]),
        "enable_rsi_extremes": bool(params["enable_rsi_extremes"]),
        "enable_ema9_regime": bool(params["enable_ema9_regime"]),
        "ema9_slope_lookback_bars": int(params["ema9_slope_lookback_bars"]),
        "enable_volatility_regime": bool(params["enable_volatility_regime"]),
        "allowed_volatility_regimes": validation.parsed_cli_allowed_volatility_regimes,
        "ma_fast_window": int(params["ma_fast_window"]),
        "ma_slow_window": int(params["ma_slow_window"]),
        "ma_trend_window": int(params["ma_trend_window"]),
        "ma_fast_type": str(params["ma_fast_type"]),
        "ma_slow_type": str(params["ma_slow_type"]),
        "ma_trend_type": str(params["ma_trend_type"]),
        "trend_slope_lookback_bars": int(params["trend_slope_lookback_bars"]),
        "atr_window": int(params["atr_window"]),
        "atr_stop_multiple": float(params["atr_stop_multiple"]),
    }


__all__ = ["build_cli_profile_values"]
