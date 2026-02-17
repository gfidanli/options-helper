from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TrendDirection = Literal["up", "down", "flat"]
DivergenceDirection = Literal["bullish", "bearish"]

DEFAULT_CONFLUENCE_CFG: dict[str, Any] = {
    "weights": {
        "weekly_trend": 25.0,
        "extension": 10.0,
        "flow": 20.0,
        "rsi_divergence": 10.0,
        "iv_regime": 5.0,
    },
    "extension": {"tail_low": 5.0, "tail_high": 95.0},
    "flow": {"min_abs_notional": 1_000_000.0},
    "rsi_divergence": {"favor_factor": 0.5},
    "iv_regime": {"low": 0.8, "high": 1.2},
}


@dataclass(frozen=True)
class ConfluenceInputs:
    weekly_trend: TrendDirection | None = None
    extension_percentile: float | None = None
    rsi_divergence: DivergenceDirection | None = None
    flow_delta_oi_notional: float | None = None
    iv_rv_20d: float | None = None


class ConfluenceComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    weight: float
    value: float | str | None
    score: float | None
    reason: str


class ConfluenceScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    total: float
    coverage: float
    components: list[ConfluenceComponent]
    warnings: list[str] = Field(default_factory=list)


def score_confluence(inputs: ConfluenceInputs, cfg: dict | None = None) -> ConfluenceScore:
    cfg = _merge_cfg(cfg)
    weights = cfg.get("weights", {})
    warnings: list[str] = []
    components: list[ConfluenceComponent] = []

    w_total = _total_weight(weights)
    w_weekly = _weight(weights, "weekly_trend")
    w_extension = _weight(weights, "extension")
    w_flow = _weight(weights, "flow")
    w_rsi = _weight(weights, "rsi_divergence")
    w_iv = _weight(weights, "iv_regime")

    trend = _normalize_trend(inputs.weekly_trend)
    components.append(_score_weekly_trend(trend, w_weekly))
    components.append(_score_extension(inputs.extension_percentile, trend, w_extension, cfg, warnings))
    components.append(_score_flow(inputs.flow_delta_oi_notional, trend, w_flow, cfg, warnings))
    components.append(_score_rsi_divergence(inputs.rsi_divergence, trend, w_rsi, cfg))
    components.append(_score_iv_regime(inputs.iv_rv_20d, w_iv, cfg, warnings))

    scored_components = [c for c in components if c.score is not None and c.weight > 0]
    w_available = sum(c.weight for c in scored_components)

    if w_total > 0 and w_available > 0:
        raw = sum(c.score for c in scored_components) / w_total
        total = 50.0 + 50.0 * _clamp(raw, -1.0, 1.0)
    else:
        total = 50.0

    coverage = (w_available / w_total) if w_total > 0 else 0.0
    if w_total == 0:
        warnings.append("no_weights_configured")
    elif w_available == 0:
        warnings.append("no_components_scored")
    elif coverage < 1.0:
        warnings.append("partial_coverage")

    return ConfluenceScore(
        total=_clamp(total, 0.0, 100.0),
        coverage=_clamp(coverage, 0.0, 1.0),
        components=components,
        warnings=warnings,
    )


def _merge_cfg(cfg: dict | None) -> dict[str, Any]:
    base = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_CONFLUENCE_CFG.items()}
    if not cfg:
        return base
    for key, val in cfg.items():
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            merged = base.get(key, {}).copy()
            merged.update(val)
            base[key] = merged
        else:
            base[key] = val
    return base


def _weight(weights: dict, name: str) -> float:
    try:
        w = float(weights.get(name, 0.0))
    except Exception:  # noqa: BLE001
        return 0.0
    return w if w > 0 else 0.0


def _total_weight(weights: dict) -> float:
    total = 0.0
    for name in ("weekly_trend", "extension", "flow", "rsi_divergence", "iv_regime"):
        total += _weight(weights, name)
    return total


def _normalize_trend(value: object) -> TrendDirection | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            value = getattr(value, "value")
        except Exception:  # noqa: BLE001
            value = value
    v = str(value).strip().lower()
    if v in {"up", "bull", "bullish"}:
        return "up"
    if v in {"down", "bear", "bearish"}:
        return "down"
    if v in {"flat", "neutral"}:
        return "flat"
    return None


def _normalize_divergence(value: object) -> DivergenceDirection | None:
    if value is None:
        return None
    if hasattr(value, "value"):
        try:
            value = getattr(value, "value")
        except Exception:  # noqa: BLE001
            value = value
    v = str(value).strip().lower()
    if v in {"bull", "bullish"}:
        return "bullish"
    if v in {"bear", "bearish"}:
        return "bearish"
    return None


def _score_weekly_trend(trend: TrendDirection | None, weight: float) -> ConfluenceComponent:
    if weight <= 0:
        return ConfluenceComponent(
            name="weekly_trend",
            weight=0.0,
            value=trend,
            score=None,
            reason="Weekly trend component disabled.",
        )
    if trend is None:
        return ConfluenceComponent(
            name="weekly_trend",
            weight=weight,
            value=None,
            score=None,
            reason="Weekly trend unavailable.",
        )
    if trend == "up":
        return ConfluenceComponent(
            name="weekly_trend",
            weight=weight,
            value="up",
            score=weight,
            reason="Weekly trend up.",
        )
    if trend == "down":
        return ConfluenceComponent(
            name="weekly_trend",
            weight=weight,
            value="down",
            score=-weight,
            reason="Weekly trend down.",
        )
    return ConfluenceComponent(
        name="weekly_trend",
        weight=weight,
        value="flat",
        score=0.0,
        reason="Weekly trend flat/neutral.",
    )


def _score_extension(
    percentile: float | None,
    trend: TrendDirection | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ConfluenceComponent:
    if weight <= 0:
        return ConfluenceComponent(
            name="extension",
            weight=0.0,
            value=percentile,
            score=None,
            reason="Extension component disabled.",
        )
    if percentile is None:
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=None,
            score=None,
            reason="Extension percentile unavailable.",
        )
    try:
        pct = float(percentile)
    except Exception:  # noqa: BLE001
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=None,
            score=None,
            reason="Extension percentile invalid.",
        )
    if pct < 0 or pct > 100:
        warnings.append("extension_percentile_out_of_range")
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=pct,
            score=None,
            reason="Extension percentile out of range.",
        )
    tail_low = float((cfg.get("extension") or {}).get("tail_low", 5.0))
    tail_high = float((cfg.get("extension") or {}).get("tail_high", 95.0))
    if tail_low >= tail_high:
        warnings.append("extension_thresholds_invalid")
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=pct,
            score=None,
            reason="Extension thresholds invalid.",
        )
    if trend is None:
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=pct,
            score=None,
            reason="Extension alignment unavailable (missing trend).",
        )
    if trend == "flat":
        return ConfluenceComponent(
            name="extension",
            weight=weight,
            value=pct,
            score=0.0,
            reason="Trend flat; extension treated as neutral.",
        )
    tail_component = _score_extension_tail(pct=pct, trend=trend, weight=weight, tail_low=tail_low, tail_high=tail_high)
    if tail_component is not None:
        return tail_component
    return ConfluenceComponent(
        name="extension",
        weight=weight,
        value=pct,
        score=0.0,
        reason="Extension percentile neutral.",
    )


def _score_extension_tail(
    *,
    pct: float,
    trend: TrendDirection,
    weight: float,
    tail_low: float,
    tail_high: float,
) -> ConfluenceComponent | None:
    if pct <= tail_low:
        aligned = trend == "up"
        score = weight if aligned else -weight
        reason = "Low extension tail aligned with trend." if aligned else "Low extension tail against trend."
        return ConfluenceComponent(name="extension", weight=weight, value=pct, score=score, reason=reason)
    if pct >= tail_high:
        aligned = trend == "down"
        score = weight if aligned else -weight
        reason = "High extension tail aligned with trend." if aligned else "High extension tail against trend."
        return ConfluenceComponent(name="extension", weight=weight, value=pct, score=score, reason=reason)
    return None


def _score_flow(
    delta_oi_notional: float | None,
    trend: TrendDirection | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ConfluenceComponent:
    if weight <= 0:
        return ConfluenceComponent(
            name="flow_alignment",
            weight=0.0,
            value=delta_oi_notional,
            score=None,
            reason="Flow component disabled.",
        )
    if delta_oi_notional is None:
        return ConfluenceComponent(
            name="flow_alignment",
            weight=weight,
            value=None,
            score=None,
            reason="Flow alignment unavailable.",
        )
    if trend is None:
        return ConfluenceComponent(
            name="flow_alignment",
            weight=weight,
            value=delta_oi_notional,
            score=None,
            reason="Flow alignment unavailable (missing trend).",
        )
    try:
        val = float(delta_oi_notional)
    except Exception:  # noqa: BLE001
        return ConfluenceComponent(
            name="flow_alignment",
            weight=weight,
            value=None,
            score=None,
            reason="Flow notional invalid.",
        )

    min_abs = float((cfg.get("flow") or {}).get("min_abs_notional", 0.0))
    if min_abs < 0:
        warnings.append("flow_min_abs_invalid")
        min_abs = 0.0

    if abs(val) < min_abs:
        return ConfluenceComponent(
            name="flow_alignment",
            weight=weight,
            value=val,
            score=0.0,
            reason="Flow notional small/neutral.",
        )
    if trend == "flat":
        return ConfluenceComponent(
            name="flow_alignment",
            weight=weight,
            value=val,
            score=0.0,
            reason="Trend flat; flow treated as neutral.",
        )

    flow_dir = "up" if val > 0 else "down"
    aligned = flow_dir == trend
    score = weight if aligned else -weight
    reason = "Flow aligned with trend." if aligned else "Flow conflicts with trend."
    return ConfluenceComponent(name="flow_alignment", weight=weight, value=val, score=score, reason=reason)


def _score_rsi_divergence(
    divergence: DivergenceDirection | None,
    trend: TrendDirection | None,
    weight: float,
    cfg: dict,
) -> ConfluenceComponent:
    if weight <= 0:
        return ConfluenceComponent(
            name="rsi_divergence",
            weight=0.0,
            value=divergence,
            score=None,
            reason="RSI divergence component disabled.",
        )
    div = _normalize_divergence(divergence)
    if div is None:
        return ConfluenceComponent(
            name="rsi_divergence",
            weight=weight,
            value=None,
            score=None,
            reason="RSI divergence unavailable.",
        )
    if trend is None:
        return ConfluenceComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=None,
            reason="RSI divergence alignment unavailable (missing trend).",
        )

    favor_factor = float((cfg.get("rsi_divergence") or {}).get("favor_factor", 0.5))
    favor_factor = _clamp(favor_factor, 0.0, 1.0)
    against_factor = -1.0

    aligned = (div == "bullish" and trend == "up") or (div == "bearish" and trend == "down")
    if aligned:
        return ConfluenceComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=weight * favor_factor,
            reason="RSI divergence in favor of trend.",
        )
    if trend in {"up", "down"}:
        return ConfluenceComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=weight * against_factor,
            reason="RSI divergence against trend.",
        )
    return ConfluenceComponent(
        name="rsi_divergence",
        weight=weight,
        value=div,
        score=0.0,
        reason="RSI divergence with flat trend.",
    )


def _score_iv_regime(
    iv_rv: float | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ConfluenceComponent:
    if weight <= 0:
        return ConfluenceComponent(
            name="iv_regime",
            weight=0.0,
            value=iv_rv,
            score=None,
            reason="IV regime component disabled.",
        )
    if iv_rv is None:
        return ConfluenceComponent(
            name="iv_regime",
            weight=weight,
            value=None,
            score=None,
            reason="IV/RV unavailable.",
        )
    try:
        ratio = float(iv_rv)
    except Exception:  # noqa: BLE001
        return ConfluenceComponent(
            name="iv_regime",
            weight=weight,
            value=None,
            score=None,
            reason="IV/RV invalid.",
        )

    iv_cfg = cfg.get("iv_regime") or {}
    low = float(iv_cfg.get("low", 0.8))
    high = float(iv_cfg.get("high", 1.2))
    if low >= high:
        warnings.append("iv_regime_thresholds_invalid")
        return ConfluenceComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=None,
            reason="IV/RV thresholds invalid.",
        )

    if ratio >= high:
        return ConfluenceComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=weight,
            reason="IV/RV high (premium rich).",
        )
    if ratio <= low:
        return ConfluenceComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=-weight,
            reason="IV/RV low (premium cheap).",
        )
    return ConfluenceComponent(
        name="iv_regime",
        weight=weight,
        value=ratio,
        score=0.0,
        reason="IV/RV neutral.",
    )


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, float(val)))
