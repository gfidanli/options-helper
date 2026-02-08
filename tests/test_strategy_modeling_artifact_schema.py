from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from options_helper.analysis.strategy_modeling_artifact import (
    parse_strategy_modeling_artifact,
    serialize_strategy_modeling_artifact,
)
from options_helper.schemas.strategy_modeling_artifact import (
    STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION,
    StrategyModelingArtifact,
)


def _ts(day: int, hour: int) -> datetime:
    return datetime(2026, 1, day, hour, 0, tzinfo=timezone.utc)


def _signal_row() -> dict[str, object]:
    return {
        "event_id": "evt-1",
        "strategy": "sfp",
        "symbol": "SPY",
        "timeframe": "1d",
        "direction": "long",
        "signal_ts": _ts(2, 21),
        "signal_confirmed_ts": _ts(2, 21),
        "entry_ts": _ts(3, 14),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "signal_open": 100.0,
        "signal_high": 102.0,
        "signal_low": 99.0,
        "signal_close": 101.0,
        "stop_price": 99.0,
        "notes": [],
    }


def _trade_row() -> dict[str, object]:
    return {
        "trade_id": "tr-1",
        "event_id": "evt-1",
        "strategy": "sfp",
        "symbol": "SPY",
        "direction": "long",
        "signal_ts": _ts(2, 21),
        "signal_confirmed_ts": _ts(2, 21),
        "entry_ts": _ts(3, 14),
        "entry_price_source": "first_tradable_bar_open_after_signal_confirmed_ts",
        "entry_price": 100.0,
        "stop_price": 99.0,
        "target_price": 101.0,
        "exit_ts": _ts(3, 15),
        "exit_price": 101.0,
        "status": "closed",
        "exit_reason": "target_hit",
        "reject_code": None,
        "initial_risk": 1.0,
        "realized_r": 1.0,
        "mae_r": -0.2,
        "mfe_r": 1.0,
        "holding_bars": 1,
        "gap_fill_applied": False,
    }


def _portfolio_metrics() -> dict[str, object]:
    return {
        "starting_capital": 10_000.0,
        "ending_capital": 10_100.0,
        "total_return_pct": 0.01,
        "sharpe_ratio": 1.0,
        "trade_count": 1,
        "win_rate": 1.0,
        "loss_rate": 0.0,
    }


def _artifact_payload() -> dict[str, object]:
    return {
        "schema_version": STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION,
        "generated_at": _ts(6, 18),
        "run_id": "run-001",
        "strategy": "sfp",
        "symbols": ["SPY"],
        "from_date": "2026-01-02",
        "to_date": "2026-01-06",
        "policy": {},
        "portfolio_metrics": _portfolio_metrics(),
        "target_hit_rates": [
            {
                "target_label": "1.0R",
                "target_r": 1.0,
                "trade_count": 1,
                "hit_count": 1,
                "hit_rate": 1.0,
            }
        ],
        "segment_records": [
            {
                "segment_dimension": "symbol",
                "segment_value": "SPY",
                "trade_count": 1,
                "win_rate": 1.0,
                "avg_realized_r": 1.0,
            }
        ],
        "equity_curve": [
            {
                "ts": _ts(3, 21),
                "equity": 10_100.0,
                "cash": 10_100.0,
                "drawdown_pct": 0.0,
                "open_trade_count": 0,
                "closed_trade_count": 1,
            }
        ],
        "trade_simulations": [_trade_row()],
        "signal_events": [_signal_row()],
        "notes": ["ok"],
    }


@pytest.mark.parametrize(
    "missing_key",
    (
        "run_id",
        "strategy",
        "symbols",
        "policy",
        "portfolio_metrics",
        "target_hit_rates",
        "segment_records",
        "equity_curve",
        "trade_simulations",
        "signal_events",
    ),
)
def test_strategy_modeling_artifact_requires_contract_sections(missing_key: str) -> None:
    payload = _artifact_payload()
    payload.pop(missing_key)

    with pytest.raises(ValidationError):
        StrategyModelingArtifact.model_validate(payload)


def test_strategy_modeling_artifact_serialization_includes_schema_version() -> None:
    artifact = parse_strategy_modeling_artifact(_artifact_payload())
    serialized = serialize_strategy_modeling_artifact(artifact)

    assert serialized["schema_version"] == STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION


def test_parse_strategy_modeling_artifact_rejects_unknown_schema_version() -> None:
    payload = _artifact_payload()
    payload["schema_version"] = 99

    with pytest.raises(ValueError, match="Unsupported strategy modeling artifact schema version: 99"):
        parse_strategy_modeling_artifact(payload)


def test_parse_strategy_modeling_artifact_supports_legacy_unversioned_payload() -> None:
    legacy_payload = {
        "generated_at": _ts(6, 18),
        "run_id": "legacy-run",
        "strategy": "sfp",
        "universe": ["SPY"],
        "start_date": "2026-01-02",
        "end_date": "2026-01-06",
        "policy_overrides": {"risk_per_trade_pct": 2.5},
        "metrics": _portfolio_metrics(),
        "r_ladder": [
            {
                "target_label": "1.0R",
                "target_r": 1.0,
                "trade_count": 1,
                "hit_count": 1,
            }
        ],
        "segments": [
            {
                "segment_dimension": "symbol",
                "segment_value": "SPY",
                "trade_count": 1,
            }
        ],
        "equity": [
            {
                "ts": _ts(3, 21),
                "equity": 10_100.0,
            }
        ],
        "trades": [_trade_row()],
        "signals": [_signal_row()],
        "warnings": ["legacy-warning"],
    }

    artifact = parse_strategy_modeling_artifact(legacy_payload)

    assert artifact.schema_version == STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION
    assert artifact.symbols == ["SPY"]
    assert artifact.from_date is not None
    assert artifact.to_date is not None
    assert artifact.from_date.isoformat() == "2026-01-02"
    assert artifact.to_date.isoformat() == "2026-01-06"
    assert artifact.policy.risk_per_trade_pct == pytest.approx(2.5)
    assert artifact.target_hit_rates[0].target_label == "1.0R"
    assert artifact.trade_simulations[0].trade_id == "tr-1"
    assert artifact.signal_events[0].event_id == "evt-1"
    assert artifact.notes == ["legacy-warning"]
