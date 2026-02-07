from __future__ import annotations

from typing import Any, Mapping

from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig


def parse_strategy_modeling_policy_config(
    overrides: Mapping[str, Any] | None = None,
) -> StrategyModelingPolicyConfig:
    """Parse policy overrides with strict validation and repository defaults."""

    payload = dict(overrides or {})
    return StrategyModelingPolicyConfig.model_validate(payload)


__all__ = [
    "StrategyModelingPolicyConfig",
    "parse_strategy_modeling_policy_config",
]

