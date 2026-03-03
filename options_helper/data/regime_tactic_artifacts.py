from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGIME_TACTIC_SCHEMA_VERSION = 1
REGIME_TACTIC_DISCLAIMER = "Informational output only; not financial advice."


def build_regime_tactic_artifact_payload(
    *,
    asof_date: str,
    symbol: str,
    market_symbol: str,
    symbol_regime: str,
    symbol_diagnostics: dict[str, Any],
    market_regime: str,
    market_diagnostics: dict[str, Any],
    direction: str,
    tactic: str,
    support_model: str,
    rationale: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": REGIME_TACTIC_SCHEMA_VERSION,
        "asof_date": str(asof_date),
        "symbol": str(symbol).upper(),
        "market_symbol": str(market_symbol).upper(),
        "regimes": {
            "symbol": {"tag": str(symbol_regime), "diagnostics": dict(symbol_diagnostics)},
            "market": {"tag": str(market_regime), "diagnostics": dict(market_diagnostics)},
        },
        "recommendation": {
            "direction": str(direction),
            "tactic": str(tactic),
            "support_model": str(support_model),
            "rationale": [str(line) for line in rationale],
        },
        "disclaimer": REGIME_TACTIC_DISCLAIMER,
    }


def write_regime_tactic_artifact_json(payload: dict[str, Any], *, out: Path) -> Path:
    symbol_label = str(payload.get("symbol", "UNKNOWN")).upper() or "UNKNOWN"
    asof_label = str(payload.get("asof_date", "unknown")) or "unknown"
    run_dir = out / symbol_label
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"{asof_label}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


__all__ = [
    "REGIME_TACTIC_DISCLAIMER",
    "REGIME_TACTIC_SCHEMA_VERSION",
    "build_regime_tactic_artifact_payload",
    "write_regime_tactic_artifact_json",
]
