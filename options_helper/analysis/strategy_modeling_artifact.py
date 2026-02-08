from __future__ import annotations

from typing import Any, Mapping

from options_helper.schemas.strategy_modeling_artifact import (
    STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION,
    StrategyModelingArtifact,
)


def parse_strategy_modeling_artifact(
    payload: Mapping[str, Any] | StrategyModelingArtifact,
) -> StrategyModelingArtifact:
    """Parse schema-versioned strategy-modeling artifacts with legacy compatibility."""

    if isinstance(payload, StrategyModelingArtifact):
        return payload

    raw_payload = dict(payload)
    schema_version = raw_payload.get("schema_version")

    if schema_version in {None, 0, "0"}:
        raw_payload = _normalize_legacy_payload(raw_payload)
        schema_version = raw_payload.get("schema_version")

    parsed_version = _parse_schema_version(schema_version)
    if parsed_version != STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported strategy modeling artifact schema version: "
            f"{parsed_version}. Expected {STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION}."
        )

    return StrategyModelingArtifact.model_validate(raw_payload)


def serialize_strategy_modeling_artifact(artifact: StrategyModelingArtifact) -> dict[str, Any]:
    return artifact.to_dict()


def _normalize_legacy_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    symbols = payload.get("symbols")
    if symbols is None:
        symbols = payload.get("universe")
    if symbols is None:
        symbols = []

    notes = payload.get("notes")
    if notes is None:
        notes = payload.get("warnings")
    if notes is None:
        notes = []

    normalized = {
        "schema_version": STRATEGY_MODELING_ARTIFACT_SCHEMA_VERSION,
        "generated_at": payload.get("generated_at"),
        "run_id": payload.get("run_id"),
        "strategy": payload.get("strategy"),
        "symbols": symbols,
        "from_date": payload.get("from_date") if "from_date" in payload else payload.get("start_date"),
        "to_date": payload.get("to_date") if "to_date" in payload else payload.get("end_date"),
        "policy": payload.get("policy") if "policy" in payload else payload.get("policy_overrides", {}),
        "portfolio_metrics": payload.get("portfolio_metrics")
        if "portfolio_metrics" in payload
        else payload.get("metrics"),
        "target_hit_rates": payload.get("target_hit_rates")
        if "target_hit_rates" in payload
        else payload.get("r_ladder", []),
        "segment_records": payload.get("segment_records")
        if "segment_records" in payload
        else payload.get("segments", []),
        "equity_curve": payload.get("equity_curve")
        if "equity_curve" in payload
        else payload.get("equity", []),
        "trade_simulations": payload.get("trade_simulations")
        if "trade_simulations" in payload
        else payload.get("trades", []),
        "signal_events": payload.get("signal_events")
        if "signal_events" in payload
        else payload.get("signals", []),
        "notes": notes,
        "disclaimer": payload.get("disclaimer")
        or "Not financial advice. For informational/educational use only.",
    }
    return normalized


def _parse_schema_version(raw_value: Any) -> int:
    try:
        return int(raw_value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "Strategy modeling artifact schema_version must be an integer."
        ) from exc


__all__ = [
    "parse_strategy_modeling_artifact",
    "serialize_strategy_modeling_artifact",
]
