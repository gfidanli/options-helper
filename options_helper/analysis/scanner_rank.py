from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TrendDirection = Literal["up", "down", "flat"]
DivergenceDirection = Literal["bullish", "bearish"]

DEFAULT_SCANNER_RANK_CFG: dict[str, Any] = {
    "weights": {
        "extension": 40.0,
        "weekly_trend": 15.0,
        "rsi_divergence": 10.0,
        "liquidity": 15.0,
        "iv_regime": 10.0,
        "flow": 10.0,
    },
    "extension": {"tail_low": 5.0, "tail_high": 95.0},
    "rsi_divergence": {"favor_factor": 0.5},
    "iv_regime": {"low": 0.8, "high": 1.2},
    "flow": {"min_abs_notional": 1_000_000.0},
    "top_reasons": 3,
}


@dataclass(frozen=True)
class ScannerRankInputs:
    extension_percentile: float | None = None
    weekly_trend: TrendDirection | None = None
    rsi_divergence: DivergenceDirection | None = None
    iv_rv_20d: float | None = None
    liquidity_score: float | None = None
    flow_delta_oi_notional: float | None = None


class ScannerRankComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    weight: float
    value: float | str | None
    score: float | None
    reason: str


class ScannerRankResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    coverage: float
    components: list[ScannerRankComponent]
    warnings: list[str] = Field(default_factory=list)
    top_reasons: list[str] = Field(default_factory=list)


def rank_scanner_candidate(inputs: ScannerRankInputs, cfg: dict | None = None) -> ScannerRankResult:
    cfg = _merge_cfg(cfg)
    weights = cfg.get("weights", {})
    warnings: list[str] = []
    components: list[ScannerRankComponent] = []

    w_total = _total_weight(weights)
    trend = _normalize_trend(inputs.weekly_trend)

    components.append(_score_extension(inputs.extension_percentile, _weight(weights, "extension"), cfg, warnings))
    components.append(_score_weekly_trend(trend, _weight(weights, "weekly_trend")))
    components.append(_score_rsi_divergence(inputs.rsi_divergence, trend, _weight(weights, "rsi_divergence"), cfg))
    components.append(_score_liquidity(inputs.liquidity_score, _weight(weights, "liquidity"), warnings))
    components.append(_score_iv_regime(inputs.iv_rv_20d, _weight(weights, "iv_regime"), cfg, warnings))
    components.append(
        _score_flow(inputs.flow_delta_oi_notional, trend, _weight(weights, "flow"), cfg, warnings)
    )

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

    top_reasons = _extract_top_reasons(components, limit=int(cfg.get("top_reasons", 3)))

    return ScannerRankResult(
        score=_clamp(total, 0.0, 100.0),
        coverage=_clamp(coverage, 0.0, 1.0),
        components=components,
        warnings=warnings,
        top_reasons=top_reasons,
    )


def _merge_cfg(cfg: dict | None) -> dict[str, Any]:
    base = {k: (v.copy() if isinstance(v, dict) else v) for k, v in DEFAULT_SCANNER_RANK_CFG.items()}
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
    for name in ("extension", "weekly_trend", "rsi_divergence", "liquidity", "iv_regime", "flow"):
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


def _score_extension(
    percentile: float | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="extension_tail",
            weight=0.0,
            value=percentile,
            score=None,
            reason="Extension component disabled.",
        )
    if percentile is None:
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=None,
            score=None,
            reason="Extension percentile unavailable.",
        )
    try:
        pct = float(percentile)
    except Exception:  # noqa: BLE001
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=None,
            score=None,
            reason="Extension percentile invalid.",
        )
    if pct < 0 or pct > 100:
        warnings.append("extension_percentile_out_of_range")
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=pct,
            score=None,
            reason="Extension percentile out of range.",
        )

    ext_cfg = cfg.get("extension") or {}
    tail_low = float(ext_cfg.get("tail_low", 5.0))
    tail_high = float(ext_cfg.get("tail_high", 95.0))
    if tail_low >= tail_high:
        warnings.append("extension_thresholds_invalid")
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=pct,
            score=None,
            reason="Extension thresholds invalid.",
        )

    if pct <= tail_low:
        denom = tail_low if tail_low > 0 else 1.0
        severity = _clamp((tail_low - pct) / denom, 0.0, 1.0)
        score = weight * severity
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=pct,
            score=score,
            reason=f"Extension tail low (pct={pct:.1f}).",
        )
    if pct >= tail_high:
        denom = (100.0 - tail_high) if tail_high < 100.0 else 1.0
        severity = _clamp((pct - tail_high) / denom, 0.0, 1.0)
        score = weight * severity
        return ScannerRankComponent(
            name="extension_tail",
            weight=weight,
            value=pct,
            score=score,
            reason=f"Extension tail high (pct={pct:.1f}).",
        )

    return ScannerRankComponent(
        name="extension_tail",
        weight=weight,
        value=pct,
        score=0.0,
        reason="Extension percentile neutral.",
    )


def _score_weekly_trend(trend: TrendDirection | None, weight: float) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="weekly_trend",
            weight=0.0,
            value=trend,
            score=None,
            reason="Weekly trend component disabled.",
        )
    if trend is None:
        return ScannerRankComponent(
            name="weekly_trend",
            weight=weight,
            value=None,
            score=None,
            reason="Weekly trend unavailable.",
        )
    if trend == "flat":
        return ScannerRankComponent(
            name="weekly_trend",
            weight=weight,
            value="flat",
            score=0.0,
            reason="Weekly trend flat/neutral.",
        )
    return ScannerRankComponent(
        name="weekly_trend",
        weight=weight,
        value=trend,
        score=weight,
        reason=f"Weekly trend {trend}.",
    )


def _score_rsi_divergence(
    divergence: DivergenceDirection | None,
    trend: TrendDirection | None,
    weight: float,
    cfg: dict,
) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="rsi_divergence",
            weight=0.0,
            value=divergence,
            score=None,
            reason="RSI divergence component disabled.",
        )
    div = _normalize_divergence(divergence)
    if div is None:
        return ScannerRankComponent(
            name="rsi_divergence",
            weight=weight,
            value=None,
            score=None,
            reason="RSI divergence unavailable.",
        )
    if trend is None:
        return ScannerRankComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=None,
            reason="RSI divergence unavailable (missing trend).",
        )
    if trend == "flat":
        return ScannerRankComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=0.0,
            reason="RSI divergence with flat trend.",
        )

    favor_factor = float((cfg.get("rsi_divergence") or {}).get("favor_factor", 0.5))
    favor_factor = _clamp(favor_factor, 0.0, 1.0)
    aligned = (div == "bullish" and trend == "up") or (div == "bearish" and trend == "down")
    if aligned:
        return ScannerRankComponent(
            name="rsi_divergence",
            weight=weight,
            value=div,
            score=weight * favor_factor,
            reason="RSI divergence aligns with trend.",
        )
    return ScannerRankComponent(
        name="rsi_divergence",
        weight=weight,
        value=div,
        score=-weight,
        reason="RSI divergence against trend.",
    )


def _score_liquidity(
    liquidity_score: float | None,
    weight: float,
    warnings: list[str],
) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="liquidity",
            weight=0.0,
            value=liquidity_score,
            score=None,
            reason="Liquidity component disabled.",
        )
    if liquidity_score is None:
        return ScannerRankComponent(
            name="liquidity",
            weight=weight,
            value=None,
            score=None,
            reason="Liquidity score unavailable.",
        )
    try:
        score_val = float(liquidity_score)
    except Exception:  # noqa: BLE001
        return ScannerRankComponent(
            name="liquidity",
            weight=weight,
            value=None,
            score=None,
            reason="Liquidity score invalid.",
        )
    if score_val < 0 or score_val > 1:
        warnings.append("liquidity_score_out_of_range")
    score_val = _clamp(score_val, 0.0, 1.0)
    return ScannerRankComponent(
        name="liquidity",
        weight=weight,
        value=score_val,
        score=weight * score_val,
        reason=f"Liquidity score {score_val:.2f}.",
    )


def _score_iv_regime(
    iv_rv: float | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="iv_regime",
            weight=0.0,
            value=iv_rv,
            score=None,
            reason="IV regime component disabled.",
        )
    if iv_rv is None:
        return ScannerRankComponent(
            name="iv_regime",
            weight=weight,
            value=None,
            score=None,
            reason="IV/RV unavailable.",
        )
    try:
        ratio = float(iv_rv)
    except Exception:  # noqa: BLE001
        return ScannerRankComponent(
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
        return ScannerRankComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=None,
            reason="IV/RV thresholds invalid.",
        )

    if ratio >= high:
        return ScannerRankComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=weight,
            reason="IV/RV high (premium rich).",
        )
    if ratio <= low:
        return ScannerRankComponent(
            name="iv_regime",
            weight=weight,
            value=ratio,
            score=-weight,
            reason="IV/RV low (premium cheap).",
        )
    return ScannerRankComponent(
        name="iv_regime",
        weight=weight,
        value=ratio,
        score=0.0,
        reason="IV/RV neutral.",
    )


def _score_flow(
    delta_oi_notional: float | None,
    trend: TrendDirection | None,
    weight: float,
    cfg: dict,
    warnings: list[str],
) -> ScannerRankComponent:
    if weight <= 0:
        return ScannerRankComponent(
            name="flow_alignment",
            weight=0.0,
            value=delta_oi_notional,
            score=None,
            reason="Flow component disabled.",
        )
    if delta_oi_notional is None:
        return ScannerRankComponent(
            name="flow_alignment",
            weight=weight,
            value=None,
            score=None,
            reason="Flow alignment unavailable.",
        )
    if trend is None:
        return ScannerRankComponent(
            name="flow_alignment",
            weight=weight,
            value=delta_oi_notional,
            score=None,
            reason="Flow alignment unavailable (missing trend).",
        )
    try:
        val = float(delta_oi_notional)
    except Exception:  # noqa: BLE001
        return ScannerRankComponent(
            name="flow_alignment",
            weight=weight,
            value=None,
            score=None,
            reason="Flow notional invalid.",
        )

    min_abs = float((cfg.get("flow") or {}).get("min_abs_notional", 0.0))
    if min_abs < 0:
        warnings.append("flow_min_abs_invalid")
        return ScannerRankComponent(
            name="flow_alignment",
            weight=weight,
            value=val,
            score=None,
            reason="Flow thresholds invalid.",
        )

    if abs(val) < min_abs:
        return ScannerRankComponent(
            name="flow_alignment",
            weight=weight,
            value=val,
            score=0.0,
            reason="Flow below notional threshold.",
        )
    flow_dir = "up" if val > 0 else "down"
    aligned = flow_dir == trend
    score = weight if aligned else -weight
    reason = "Flow aligns with trend." if aligned else "Flow against trend."
    return ScannerRankComponent(
        name="flow_alignment",
        weight=weight,
        value=val,
        score=score,
        reason=reason,
    )


def _extract_top_reasons(components: list[ScannerRankComponent], *, limit: int) -> list[str]:
    if limit <= 0:
        return []
    scored = [c for c in components if c.score is not None and c.weight > 0 and abs(c.score) > 0]
    ranked = sorted(scored, key=lambda c: abs(float(c.score or 0.0)), reverse=True)
    reasons: list[str] = []
    for comp in ranked:
        reason = comp.reason
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= limit:
            break
    return reasons


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(high, float(val)))
