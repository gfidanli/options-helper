from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MarketRegimeTag = Literal["trend_up", "trend_down", "choppy", "mixed", "insufficient_data"]
SymbolRegimeTag = Literal["trend_up", "trend_down", "sideways", "choppy", "mixed", "insufficient_data"]
TradeDirection = Literal["long", "short"]
RegimeTactic = Literal["breakout", "undercut_reclaim", "avoid"]
SupportModel = Literal["ema", "static"]

RegimeBucket = Literal[
    "trend_supportive",
    "trend_headwind",
    "range_bound",
    "mixed",
    "insufficient_data",
]
DecisionId = Literal[
    "trend_breakout",
    "range_undercut_reclaim",
    "mixed_undercut_reclaim",
    "avoid_headwind",
    "avoid_uncertain",
    "avoid_insufficient_data",
]


@dataclass(frozen=True)
class RegimeTacticRecommendation:
    tactic: RegimeTactic
    support_model: SupportModel
    rationale: list[str]


@dataclass(frozen=True)
class _DecisionSpec:
    tactic: RegimeTactic
    support_model: SupportModel
    rationale: tuple[str, ...]


# Accepted aliases for stable, case-insensitive input normalization.
MARKET_REGIME_ALIASES: dict[str, MarketRegimeTag] = {
    "trend_up": "trend_up",
    "uptrend": "trend_up",
    "trend": "trend_up",
    "trend_down": "trend_down",
    "downtrend": "trend_down",
    "choppy": "choppy",
    "sideways": "choppy",
    "mixed": "mixed",
    "insufficient_data": "insufficient_data",
}
SYMBOL_REGIME_ALIASES: dict[str, SymbolRegimeTag] = {
    "trend_up": "trend_up",
    "uptrend": "trend_up",
    "trend": "trend_up",
    "trend_down": "trend_down",
    "downtrend": "trend_down",
    "sideways": "sideways",
    "choppy": "choppy",
    "mixed": "mixed",
    "insufficient_data": "insufficient_data",
}

# Direction is used to orient trend tags into "supportive" vs "headwind" buckets.
DIRECTIONAL_TREND_MAP: dict[TradeDirection, tuple[MarketRegimeTag, MarketRegimeTag]] = {
    "long": ("trend_up", "trend_down"),
    "short": ("trend_down", "trend_up"),
}

# Decision specs are intentionally small and deterministic.
DECISION_SPECS: dict[DecisionId, _DecisionSpec] = {
    "trend_breakout": _DecisionSpec(
        tactic="breakout",
        support_model="ema",
        rationale=(
            "Market and symbol trends align with trade direction.",
            "EMA support tracks continuation structure in trending tape.",
        ),
    ),
    "range_undercut_reclaim": _DecisionSpec(
        tactic="undercut_reclaim",
        support_model="static",
        rationale=(
            "Range-bound conditions favor reclaim entries over breakout chasing.",
            "Static support levels keep risk placement explicit in chop.",
        ),
    ),
    "mixed_undercut_reclaim": _DecisionSpec(
        tactic="undercut_reclaim",
        support_model="static",
        rationale=(
            "Regime disagreement lowers continuation confidence.",
            "Use reclaim setups around clear static levels and tighter invalidation.",
        ),
    ),
    "avoid_headwind": _DecisionSpec(
        tactic="avoid",
        support_model="static",
        rationale=(
            "At least one regime trend conflicts with trade direction.",
            "Skip until market and symbol context stop fighting the setup.",
        ),
    ),
    "avoid_uncertain": _DecisionSpec(
        tactic="avoid",
        support_model="static",
        rationale=(
            "Mixed/unclear regimes reduce signal clarity.",
            "Stand aside until a cleaner regime combination appears.",
        ),
    ),
    "avoid_insufficient_data": _DecisionSpec(
        tactic="avoid",
        support_model="static",
        rationale=(
            "Insufficient data prevents reliable regime classification.",
            "Wait for enough history before selecting an entry tactic.",
        ),
    ),
}

# Direction-agnostic decision matrix on oriented buckets.
# Rows/cols are (market_bucket, symbol_bucket).
REGIME_TACTIC_DECISION_TABLE: dict[tuple[RegimeBucket, RegimeBucket], DecisionId] = {
    ("trend_supportive", "trend_supportive"): "trend_breakout",
    ("trend_supportive", "range_bound"): "mixed_undercut_reclaim",
    ("trend_supportive", "mixed"): "mixed_undercut_reclaim",
    ("trend_supportive", "trend_headwind"): "avoid_headwind",
    ("trend_supportive", "insufficient_data"): "avoid_insufficient_data",
    ("range_bound", "trend_supportive"): "mixed_undercut_reclaim",
    ("range_bound", "range_bound"): "range_undercut_reclaim",
    ("range_bound", "mixed"): "range_undercut_reclaim",
    ("range_bound", "trend_headwind"): "avoid_headwind",
    ("range_bound", "insufficient_data"): "avoid_insufficient_data",
    ("mixed", "trend_supportive"): "mixed_undercut_reclaim",
    ("mixed", "range_bound"): "range_undercut_reclaim",
    ("mixed", "mixed"): "avoid_uncertain",
    ("mixed", "trend_headwind"): "avoid_headwind",
    ("mixed", "insufficient_data"): "avoid_insufficient_data",
    ("trend_headwind", "trend_supportive"): "avoid_headwind",
    ("trend_headwind", "range_bound"): "avoid_headwind",
    ("trend_headwind", "mixed"): "avoid_headwind",
    ("trend_headwind", "trend_headwind"): "avoid_headwind",
    ("trend_headwind", "insufficient_data"): "avoid_insufficient_data",
    ("insufficient_data", "trend_supportive"): "avoid_insufficient_data",
    ("insufficient_data", "range_bound"): "avoid_insufficient_data",
    ("insufficient_data", "mixed"): "avoid_insufficient_data",
    ("insufficient_data", "trend_headwind"): "avoid_insufficient_data",
    ("insufficient_data", "insufficient_data"): "avoid_insufficient_data",
}


def map_regime_to_tactic(
    market_regime: MarketRegimeTag | str,
    symbol_regime: SymbolRegimeTag | str,
    *,
    direction: TradeDirection = "long",
) -> RegimeTacticRecommendation:
    normalized_direction = _normalize_direction(direction)
    normalized_market = _normalize_market_regime(market_regime)
    normalized_symbol = _normalize_symbol_regime(symbol_regime)
    market_bucket = _bucketize_market(normalized_market, direction=normalized_direction)
    symbol_bucket = _bucketize_symbol(normalized_symbol, direction=normalized_direction)
    decision_id = REGIME_TACTIC_DECISION_TABLE[(market_bucket, symbol_bucket)]
    spec = DECISION_SPECS[decision_id]
    return RegimeTacticRecommendation(
        tactic=spec.tactic,
        support_model=spec.support_model,
        rationale=list(spec.rationale),
    )


def _normalize_direction(value: TradeDirection | str) -> TradeDirection:
    token = str(value or "").strip().lower()
    if token == "long":
        return "long"
    if token == "short":
        return "short"
    raise ValueError("direction must be one of: long, short")


def _normalize_market_regime(value: MarketRegimeTag | str) -> MarketRegimeTag:
    token = str(value or "").strip().lower()
    normalized = MARKET_REGIME_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(set(MARKET_REGIME_ALIASES.values())))
        raise ValueError(f"Unknown market_regime '{value}'. Expected one of: {allowed}")
    return normalized


def _normalize_symbol_regime(value: SymbolRegimeTag | str) -> SymbolRegimeTag:
    token = str(value or "").strip().lower()
    normalized = SYMBOL_REGIME_ALIASES.get(token)
    if normalized is None:
        allowed = ", ".join(sorted(set(SYMBOL_REGIME_ALIASES.values())))
        raise ValueError(f"Unknown symbol_regime '{value}'. Expected one of: {allowed}")
    return normalized


def _bucketize_market(regime: MarketRegimeTag, *, direction: TradeDirection) -> RegimeBucket:
    supportive_tag, headwind_tag = DIRECTIONAL_TREND_MAP[direction]
    if regime == supportive_tag:
        return "trend_supportive"
    if regime == headwind_tag:
        return "trend_headwind"
    if regime == "choppy":
        return "range_bound"
    if regime == "mixed":
        return "mixed"
    return "insufficient_data"


def _bucketize_symbol(regime: SymbolRegimeTag, *, direction: TradeDirection) -> RegimeBucket:
    supportive_tag, headwind_tag = DIRECTIONAL_TREND_MAP[direction]
    if regime == supportive_tag:
        return "trend_supportive"
    if regime == headwind_tag:
        return "trend_headwind"
    if regime in {"sideways", "choppy"}:
        return "range_bound"
    if regime == "mixed":
        return "mixed"
    return "insufficient_data"


__all__ = [
    "MarketRegimeTag",
    "SymbolRegimeTag",
    "TradeDirection",
    "RegimeTactic",
    "SupportModel",
    "RegimeTacticRecommendation",
    "map_regime_to_tactic",
]
