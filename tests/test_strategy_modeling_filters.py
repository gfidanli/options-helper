from __future__ import annotations

from dataclasses import fields

import pytest
from pydantic import ValidationError

from options_helper.analysis.strategy_modeling import StrategyModelingRequest, StrategyModelingRunResult
from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig


def _dataclass_field_names(model: type) -> set[str]:
    return {item.name for item in fields(model)}


def test_strategy_entry_filter_config_defaults_keep_filters_off() -> None:
    cfg = StrategyEntryFilterConfig()
    assert cfg.allow_shorts is True
    assert cfg.enable_orb_confirmation is False
    assert cfg.enable_atr_stop_floor is False
    assert cfg.enable_rsi_extremes is False
    assert cfg.enable_ema9_regime is False
    assert cfg.enable_volatility_regime is False
    assert cfg.orb_stop_policy == "base"


def test_strategy_entry_filter_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        StrategyEntryFilterConfig.model_validate({"unknown_filter": True})


@pytest.mark.parametrize(
    "payload",
    (
        {"orb_stop_policy": "invalid"},
        {"orb_range_minutes": 0},
        {"orb_confirmation_cutoff_et": "24:00"},
        {"atr_stop_floor_multiple": 0.0},
        {"ema9_slope_lookback_bars": 0},
        {"allowed_volatility_regimes": []},
        {"allowed_volatility_regimes": ["low", "low"]},
        {"allowed_volatility_regimes": ["extreme"]},
    ),
)
def test_strategy_entry_filter_config_enforces_domain_constraints(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        StrategyEntryFilterConfig.model_validate(payload)


def test_strategy_modeling_contracts_include_filter_and_directional_fields() -> None:
    request_fields = _dataclass_field_names(StrategyModelingRequest)
    result_fields = _dataclass_field_names(StrategyModelingRunResult)

    assert "filter_config" in request_fields
    assert "filter_metadata" in result_fields
    assert "filter_summary" in result_fields
    assert "directional_metrics" in result_fields
