from __future__ import annotations

import pytest
from pydantic import ValidationError

from options_helper.analysis.strategy_modeling_policy import (
    StrategyModelingPolicyConfig,
    parse_strategy_modeling_policy_config,
)


def test_strategy_modeling_policy_defaults() -> None:
    cfg = StrategyModelingPolicyConfig()
    assert cfg.require_intraday_bars is True
    assert cfg.max_hold_bars is None
    assert cfg.max_hold_timeframe == "entry"
    assert cfg.sizing_rule == "risk_pct_of_equity"
    assert cfg.risk_per_trade_pct == 1.0
    assert cfg.one_open_per_symbol is True
    assert cfg.gap_fill_policy == "fill_at_open"
    assert cfg.entry_ts_anchor_policy == "first_tradable_bar_open_after_signal_confirmed_ts"
    assert cfg.price_adjustment_policy == "adjusted_ohlc"
    assert cfg.stop_trail_rules == []


def test_parse_strategy_modeling_policy_config_applies_overrides() -> None:
    cfg = parse_strategy_modeling_policy_config(
        {
            "require_intraday_bars": False,
            "max_hold_bars": "30",
            "max_hold_timeframe": "10min",
            "risk_per_trade_pct": "2.5",
            "one_open_per_symbol": "false",
        }
    )
    assert cfg.require_intraday_bars is False
    assert cfg.max_hold_bars == 30
    assert cfg.max_hold_timeframe == "10Min"
    assert cfg.risk_per_trade_pct == 2.5
    assert cfg.one_open_per_symbol is False
    assert cfg.entry_ts_anchor_policy == "first_tradable_bar_open_after_signal_confirmed_ts"


def test_parse_strategy_modeling_policy_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config({"unknown_policy": True})


def test_parse_strategy_modeling_policy_config_rejects_invalid_risk() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config({"risk_per_trade_pct": 0.0})


def test_parse_strategy_modeling_policy_config_rejects_invalid_max_hold_bars() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config({"max_hold_bars": 0})


def test_parse_strategy_modeling_policy_config_rejects_invalid_max_hold_timeframe() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config({"max_hold_timeframe": "quarter"})


def test_parse_strategy_modeling_policy_config_sorts_stop_trail_rules_by_start_r() -> None:
    cfg = parse_strategy_modeling_policy_config(
        {
            "stop_trail_rules": [
                {"start_r": 2.0, "ema_span": 9},
                {"start_r": 0.5, "ema_span": 21, "buffer_atr_multiple": 0.25},
                {"start_r": 1.0, "ema_span": 50},
            ]
        }
    )
    assert [rule.start_r for rule in cfg.stop_trail_rules] == [0.5, 1.0, 2.0]
    assert [rule.ema_span for rule in cfg.stop_trail_rules] == [21, 50, 9]
    assert cfg.stop_trail_rules[0].buffer_atr_multiple == 0.25


def test_parse_strategy_modeling_policy_config_rejects_invalid_stop_trail_rules() -> None:
    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config(
            {"stop_trail_rules": [{"start_r": -0.1, "ema_span": 21}]}
        )

    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config(
            {"stop_trail_rules": [{"start_r": 0.5, "ema_span": 13}]}
        )

    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config(
            {
                "stop_trail_rules": [
                    {"start_r": 0.5, "ema_span": 21, "buffer_atr_multiple": -0.1}
                ]
            }
        )

    with pytest.raises(ValidationError):
        parse_strategy_modeling_policy_config(
            {
                "stop_trail_rules": [
                    {"start_r": 0.5, "ema_span": 21},
                    {"start_r": 0.5, "ema_span": 9},
                ]
            }
        )
