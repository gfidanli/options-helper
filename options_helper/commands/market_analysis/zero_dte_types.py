from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from options_helper.schemas.zero_dte_put_study import ZeroDtePutStudyArtifact


_ZERO_DTE_DEFAULT_STRIKE_GRID: tuple[float, ...] = (-0.03, -0.02, -0.015, -0.01, -0.005)
_ZERO_DTE_FORWARD_KEY_FIELDS: tuple[str, ...] = (
    "symbol",
    "session_date",
    "decision_ts",
    "risk_tier",
    "model_version",
    "assumptions_hash",
)


@dataclass(frozen=True)
class _ZeroDTEStudyResult:
    artifact: ZeroDtePutStudyArtifact
    active_model: dict[str, Any] | None
    preflight_passed: bool
    preflight_messages: list[str]


@dataclass(frozen=True)
class _ZeroDTEForwardResult:
    payload: dict[str, Any]
    rows: list[dict[str, Any]]


__all__ = [
    "_ZERO_DTE_DEFAULT_STRIKE_GRID",
    "_ZERO_DTE_FORWARD_KEY_FIELDS",
    "_ZeroDTEStudyResult",
    "_ZeroDTEForwardResult",
]
