from __future__ import annotations

from datetime import time

from options_helper.schemas.zero_dte_put_study import DecisionMode, FillModel


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


__all__ = [
    "_parse_decision_mode",
    "_parse_fill_model",
    "_parse_time_csv",
    "_parse_positive_probability_csv",
    "_parse_strike_return_csv",
    "_parse_float_csv",
]
