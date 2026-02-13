from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import typer

if TYPE_CHECKING:
    from options_helper.analysis.strategy_simulator import StrategyRTarget


def _parse_iso_date(value: str | None, *, option_name: str) -> date | None:
    if value is None:
        return None
    token = value.strip()
    if not token:
        return None
    try:
        return date.fromisoformat(token)
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be in YYYY-MM-DD format.") from exc


def _split_csv_option(value: str | None, *, uppercase: bool = True) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if uppercase:
        return [item.upper() for item in items]
    return items


def _normalize_hhmm_et_cutoff(value: str, *, option_name: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise typer.BadParameter(f"{option_name} must be HH:MM in 24-hour ET time.")
    try:
        datetime.strptime(token, "%H:%M")
    except ValueError as exc:
        raise typer.BadParameter(f"{option_name} must be HH:MM in 24-hour ET time.") from exc
    return token


def _normalize_orb_stop_policy(value: str, *, option_name: str) -> str:
    token = str(value or "").strip().lower()
    allowed = ("base", "orb_range", "tighten")
    if token not in allowed:
        raise typer.BadParameter(f"{option_name} must be one of: {', '.join(allowed)}")
    return token


def _normalize_ma_type(value: str, *, option_name: str) -> str:
    token = str(value or "").strip().lower()
    allowed = ("sma", "ema")
    if token not in allowed:
        raise typer.BadParameter(f"{option_name} must be one of: {', '.join(allowed)}")
    return token


def _parse_allowed_volatility_regimes(value: str, *, option_name: str) -> tuple[str, ...]:
    allowed = ("low", "normal", "high")
    parsed = tuple(item.strip().lower() for item in str(value or "").split(",") if item.strip())
    if not parsed:
        raise typer.BadParameter(
            f"{option_name} must include at least one regime from: {', '.join(allowed)}"
        )
    if len(set(parsed)) != len(parsed):
        raise typer.BadParameter(f"{option_name} must not include duplicate regimes.")
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise typer.BadParameter(
            f"{option_name} contains invalid regime(s): {', '.join(invalid)}. "
            f"Allowed: {', '.join(allowed)}"
        )
    return parsed


def _build_strategy_signal_kwargs(
    *,
    strategy: str,
    fib_retracement_pct: float,
    ma_fast_window: int,
    ma_slow_window: int,
    ma_trend_window: int,
    ma_fast_type: str,
    ma_slow_type: str,
    ma_trend_type: str,
    trend_slope_lookback_bars: int,
    atr_window: int,
    atr_stop_multiple: float,
) -> dict[str, object]:
    if ma_fast_window < 1:
        raise typer.BadParameter("--ma-fast-window must be >= 1")
    if ma_slow_window < 1:
        raise typer.BadParameter("--ma-slow-window must be >= 1")
    if ma_trend_window < 1:
        raise typer.BadParameter("--ma-trend-window must be >= 1")
    if trend_slope_lookback_bars < 1:
        raise typer.BadParameter("--trend-slope-lookback-bars must be >= 1")
    if atr_window < 1:
        raise typer.BadParameter("--atr-window must be >= 1")
    if atr_stop_multiple <= 0.0:
        raise typer.BadParameter("--atr-stop-multiple must be > 0")

    normalized_fast_type = _normalize_ma_type(ma_fast_type, option_name="--ma-fast-type")
    normalized_slow_type = _normalize_ma_type(ma_slow_type, option_name="--ma-slow-type")
    normalized_trend_type = _normalize_ma_type(ma_trend_type, option_name="--ma-trend-type")

    if strategy == "ma_crossover":
        if ma_fast_window >= ma_slow_window:
            raise typer.BadParameter("--ma-fast-window must be < --ma-slow-window for ma_crossover.")
        return {
            "fast_window": int(ma_fast_window),
            "slow_window": int(ma_slow_window),
            "fast_type": normalized_fast_type,
            "slow_type": normalized_slow_type,
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    if strategy == "trend_following":
        return {
            "trend_window": int(ma_trend_window),
            "trend_type": normalized_trend_type,
            "fast_window": int(ma_fast_window),
            "fast_type": normalized_fast_type,
            "slope_lookback_bars": int(trend_slope_lookback_bars),
            "atr_window": int(atr_window),
            "atr_stop_multiple": float(atr_stop_multiple),
        }
    if strategy == "fib_retracement":
        from options_helper.analysis.fib_retracement import normalize_fib_retracement_pct

        try:
            normalized_fib_retracement_pct = normalize_fib_retracement_pct(fib_retracement_pct)
        except ValueError as exc:
            raise typer.BadParameter(f"--fib-retracement-pct {exc}") from exc
        return {"fib_retracement_pct": float(normalized_fib_retracement_pct)}
    return {}


def _build_strategy_filter_config(
    *,
    allow_shorts: bool,
    enable_orb_confirmation: bool,
    orb_range_minutes: int,
    orb_confirmation_cutoff_et: str,
    orb_stop_policy: str,
    enable_atr_stop_floor: bool,
    atr_stop_floor_multiple: float,
    enable_rsi_extremes: bool,
    enable_ema9_regime: bool,
    ema9_slope_lookback_bars: int,
    enable_volatility_regime: bool,
    allowed_volatility_regimes: str,
):
    from pydantic import ValidationError

    from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig

    if orb_range_minutes < 1 or orb_range_minutes > 120:
        raise typer.BadParameter("--orb-range-minutes must be between 1 and 120.")
    if atr_stop_floor_multiple <= 0.0:
        raise typer.BadParameter("--atr-stop-floor-multiple must be > 0.")
    if ema9_slope_lookback_bars < 1:
        raise typer.BadParameter("--ema9-slope-lookback-bars must be >= 1.")

    cutoff = _normalize_hhmm_et_cutoff(
        orb_confirmation_cutoff_et,
        option_name="--orb-confirmation-cutoff-et",
    )
    normalized_stop_policy = _normalize_orb_stop_policy(
        orb_stop_policy,
        option_name="--orb-stop-policy",
    )
    parsed_regimes = _parse_allowed_volatility_regimes(
        allowed_volatility_regimes,
        option_name="--allowed-volatility-regimes",
    )

    payload = {
        "allow_shorts": bool(allow_shorts),
        "enable_orb_confirmation": bool(enable_orb_confirmation),
        "orb_range_minutes": int(orb_range_minutes),
        "orb_confirmation_cutoff_et": cutoff,
        "orb_stop_policy": normalized_stop_policy,
        "enable_atr_stop_floor": bool(enable_atr_stop_floor),
        "atr_stop_floor_multiple": float(atr_stop_floor_multiple),
        "enable_rsi_extremes": bool(enable_rsi_extremes),
        "enable_ema9_regime": bool(enable_ema9_regime),
        "ema9_slope_lookback_bars": int(ema9_slope_lookback_bars),
        "enable_volatility_regime": bool(enable_volatility_regime),
        "allowed_volatility_regimes": parsed_regimes,
    }
    try:
        return StrategyEntryFilterConfig.model_validate(payload)
    except ValidationError as exc:
        detail = "; ".join(error.get("msg", "invalid value") for error in exc.errors())
        raise typer.BadParameter(f"Invalid strategy filter configuration: {detail}") from exc


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _mapping_view(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, SimpleNamespace):
        return vars(value)
    if hasattr(value, "__dict__"):
        try:
            return dict(vars(value))
        except TypeError:
            return {}
    return {}


def _extract_directional_headline(bucket: object) -> tuple[int | None, float | None]:
    payload = _mapping_view(bucket)
    if not payload:
        return None, None

    trade_count = _coerce_int(payload.get("trade_count"))
    if trade_count is None:
        trade_count = _coerce_int(payload.get("closed_trade_count"))

    total_return = _coerce_float(payload.get("total_return_pct"))
    if total_return is None:
        total_return = _coerce_float(_mapping_view(payload.get("portfolio_metrics")).get("total_return_pct"))
    return trade_count, total_return


def _build_target_ladder(
    *,
    min_tenths: int,
    max_tenths: int,
    step_tenths: int,
) -> tuple[StrategyRTarget, ...]:
    from options_helper.analysis.strategy_simulator import build_r_target_ladder

    if min_tenths < 1:
        raise typer.BadParameter("--r-ladder-min-tenths must be >= 1")
    if max_tenths < min_tenths:
        raise typer.BadParameter("--r-ladder-max-tenths must be >= --r-ladder-min-tenths")
    if step_tenths < 1:
        raise typer.BadParameter("--r-ladder-step-tenths must be >= 1")

    return build_r_target_ladder(
        min_target_tenths=min_tenths,
        max_target_tenths=max_tenths,
        step_tenths=step_tenths,
    )


_OUTPUT_TIMEZONE_ALIASES: dict[str, str] = {
    "CST": "America/Chicago",
    "CDT": "America/Chicago",
}


def _normalize_output_timezone(value: str, *, option_name: str = "--output-timezone") -> str:
    token = str(value or "").strip()
    if not token:
        raise typer.BadParameter(f"{option_name} must be a non-empty IANA timezone.")

    canonical = _OUTPUT_TIMEZONE_ALIASES.get(token.upper(), token)
    try:
        ZoneInfo(canonical)
    except ZoneInfoNotFoundError as exc:
        raise typer.BadParameter(
            f"{option_name}={value!r} is invalid. Use an IANA timezone (for example America/Chicago)."
        ) from exc
    return canonical


def _resolve_strategy_symbols(
    *,
    service: object,
    requested_symbols: list[str],
    excluded_symbols: set[str],
    universe_limit: int | None,
) -> tuple[str, ...]:
    candidates: list[str] = []
    if requested_symbols:
        candidates = requested_symbols
    else:
        loader = getattr(service, "list_universe_loader", None)
        if not callable(loader):
            raise typer.BadParameter(
                "No --symbols provided and strategy-modeling service does not expose universe loader."
            )
        universe = loader(database_path=None)
        symbols = getattr(universe, "symbols", ()) or ()
        candidates = [str(symbol).upper() for symbol in symbols if str(symbol).strip()]

    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in candidates:
        token = str(symbol).strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        deduped.append(token)

    filtered = [symbol for symbol in deduped if symbol not in excluded_symbols]
    if universe_limit is not None:
        filtered = filtered[: int(universe_limit)]

    return tuple(filtered)


def _intraday_coverage_block_message(preflight: object | None) -> str | None:
    if preflight is None:
        return None

    blocked_symbols = tuple(getattr(preflight, "blocked_symbols", ()) or ())
    if not blocked_symbols:
        return None

    coverage_by_symbol = getattr(preflight, "coverage_by_symbol", {}) or {}
    details: list[str] = []
    for symbol in blocked_symbols:
        coverage = coverage_by_symbol.get(symbol)
        required_days = getattr(coverage, "required_days", ()) or ()
        missing_days = getattr(coverage, "missing_days", ()) or ()
        required = len(tuple(required_days))
        missing = len(tuple(missing_days))
        details.append(f"{symbol}({missing}/{required} missing)")

    detail_text = ", ".join(details) if details else ", ".join(blocked_symbols)
    message = f"Missing required intraday coverage for requested scope: {detail_text}"
    notes = tuple(getattr(preflight, "notes", ()) or ())
    if notes:
        message = f"{message} | notes: {'; '.join(str(note) for note in notes)}"
    return message


def _option_was_set_on_command_line(ctx: typer.Context, option_name: str) -> bool:
    from click.core import ParameterSource

    source = ctx.get_parameter_source(option_name)
    return source == ParameterSource.COMMANDLINE


def _merge_strategy_modeling_profile_values(
    *,
    ctx: typer.Context,
    loaded_profile,
    cli_values: dict[str, object],
):
    from options_helper.schemas.strategy_modeling_profile import StrategyModelingProfile

    payload = loaded_profile.model_dump() if loaded_profile is not None else {}
    for field_name, field_value in cli_values.items():
        if loaded_profile is None or _option_was_set_on_command_line(ctx, field_name):
            payload[field_name] = field_value
        else:
            payload.setdefault(field_name, field_value)
    return StrategyModelingProfile.model_validate(payload)


__all__ = [
    "_parse_iso_date",
    "_split_csv_option",
    "_parse_allowed_volatility_regimes",
    "_build_strategy_signal_kwargs",
    "_build_strategy_filter_config",
    "_merge_strategy_modeling_profile_values",
    "_option_was_set_on_command_line",
    "_build_target_ladder",
    "_resolve_strategy_symbols",
    "_intraday_coverage_block_message",
    "_normalize_output_timezone",
    "_mapping_view",
    "_coerce_int",
    "_extract_directional_headline",
]
