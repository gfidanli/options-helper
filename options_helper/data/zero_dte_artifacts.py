from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from options_helper.schemas.zero_dte_put_study import (
    ModelPromotionAction,
    ModelRegistryStatus,
    ZeroDteBacktestSummaryArtifact,
    ZeroDteBacktestSummaryRow,
    ZeroDteCalibrationArtifact,
    ZeroDteCalibrationTableRow,
    ZeroDteForwardProbabilityCurveRow,
    ZeroDteForwardSnapshotArtifact,
    ZeroDteForwardSnapshotRow,
    ZeroDteForwardStrikeLadderRow,
    ZeroDteForwardUpsertKey,
    ZeroDteModelCompatibilityMetadata,
    ZeroDteModelPromotionRecord,
    ZeroDteModelRegistryArtifact,
    ZeroDteModelRegistryEntry,
    ZeroDteModelSnapshotArtifact,
    ZeroDteProbabilityCurveArtifact,
    ZeroDteStrikeLadderArtifact,
    ZeroDteStudyAssumptions,
    ZeroDteTradeLedgerArtifact,
    ZeroDteTradeLedgerRow,
)


_DEFAULT_REGISTRY_AS_OF = date(1970, 1, 1)
_TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ZeroDteArtifactsError(RuntimeError):
    """Base error for 0DTE artifact persistence."""


class ModelStateMissingError(ZeroDteArtifactsError):
    """Raised when model state files are missing."""


class ModelStateStaleError(ZeroDteArtifactsError):
    """Raised when a model state is stale for requested scoring."""


class ModelStateRegistryError(ZeroDteArtifactsError):
    """Raised when registry state is invalid or incomplete."""


class ModelStateCompatibilityError(ZeroDteArtifactsError):
    """Raised when model compatibility requirements are not met."""


@dataclass(frozen=True)
class ZeroDteArtifactPaths:
    root_dir: Path
    model_snapshots_dir: Path
    model_registry_path: Path
    probability_curves_path: Path
    strike_ladders_path: Path
    calibration_tables_path: Path
    backtest_summaries_path: Path
    forward_snapshots_path: Path
    trade_ledgers_path: Path



def build_zero_dte_artifact_paths(root_dir: Path) -> ZeroDteArtifactPaths:
    root = Path(root_dir)
    return ZeroDteArtifactPaths(
        root_dir=root,
        model_snapshots_dir=root / "model_states",
        model_registry_path=root / "model_registry.json",
        probability_curves_path=root / "probability_curves.json",
        strike_ladders_path=root / "strike_ladders.json",
        calibration_tables_path=root / "calibration_tables.json",
        backtest_summaries_path=root / "backtest_summaries.json",
        forward_snapshots_path=root / "forward_snapshots.json",
        trade_ledgers_path=root / "trade_ledgers.json",
    )



def _safe_token(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("token values must not be empty")
    return _TOKEN_RE.sub("_", raw)



def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).isoformat()
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")



def canonicalize_payload(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )



def compute_assumptions_hash(assumptions: ZeroDteStudyAssumptions | Mapping[str, Any]) -> str:
    if isinstance(assumptions, ZeroDteStudyAssumptions):
        payload: Mapping[str, Any] = assumptions.to_dict()
    else:
        payload = assumptions
    return hashlib.sha256(canonicalize_payload(payload).encode("utf-8")).hexdigest()



def compute_snapshot_hash(
    *,
    model_version: str,
    trained_through_session: date,
    assumptions_hash: str,
    model_payload: Mapping[str, Any],
) -> str:
    payload = {
        "model_version": model_version,
        "trained_through_session": trained_through_session.isoformat(),
        "assumptions_hash": assumptions_hash,
        "model_payload": model_payload,
    }
    return hashlib.sha256(canonicalize_payload(payload).encode("utf-8")).hexdigest()



def build_model_snapshot(
    *,
    model_version: str,
    trained_through_session: date,
    assumptions_hash: str,
    model_payload: Mapping[str, Any],
    compatibility: ZeroDteModelCompatibilityMetadata | None = None,
    notes: Sequence[str] | None = None,
) -> ZeroDteModelSnapshotArtifact:
    snapshot_hash = compute_snapshot_hash(
        model_version=model_version,
        trained_through_session=trained_through_session,
        assumptions_hash=assumptions_hash,
        model_payload=model_payload,
    )
    return ZeroDteModelSnapshotArtifact(
        model_version=model_version,
        trained_through_session=trained_through_session,
        assumptions_hash=assumptions_hash,
        snapshot_hash=snapshot_hash,
        model_payload=dict(model_payload),
        compatibility=(
            compatibility
            if compatibility is not None
            else ZeroDteModelCompatibilityMetadata()
        ),
        notes=list(notes or []),
    )



def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()



def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))



def _to_utc_iso(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat()



def _compatibility_matches(
    actual: ZeroDteModelCompatibilityMetadata,
    required: ZeroDteModelCompatibilityMetadata,
) -> bool:
    return (
        actual.artifact_schema_version >= required.artifact_schema_version
        and actual.feature_contract_version == required.feature_contract_version
        and actual.policy_contract_version == required.policy_contract_version
    )



def forward_upsert_key(row: ZeroDteForwardUpsertKey) -> tuple[str, str, str, str, str, str]:
    return (
        row.symbol.upper(),
        row.session_date.isoformat(),
        _to_utc_iso(row.decision_ts),
        f"{row.risk_tier:.10f}",
        row.model_version,
        row.assumptions_hash,
    )



TModel = TypeVar("TModel")


def _merge_rows(
    existing: Sequence[TModel],
    incoming: Sequence[TModel],
    key_fn: Callable[[TModel], tuple[Any, ...]],
) -> list[TModel]:
    merged: dict[tuple[Any, ...], TModel] = {}
    for row in existing:
        merged[key_fn(row)] = row
    for row in incoming:
        merged[key_fn(row)] = row
    return [merged[key] for key in sorted(merged)]


@dataclass
class ZeroDteArtifactStore:
    root_dir: Path

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.paths = build_zero_dte_artifact_paths(self.root_dir)

    def model_snapshot_path(self, model_version: str) -> Path:
        return self.paths.model_snapshots_dir / f"{_safe_token(model_version)}.json"

    def save_model_snapshot(self, snapshot: ZeroDteModelSnapshotArtifact) -> Path:
        path = self.model_snapshot_path(snapshot.model_version)
        _atomic_write_json(path, snapshot.to_dict())
        return path

    def load_model_snapshot(self, model_version: str) -> ZeroDteModelSnapshotArtifact:
        return self._load_model_snapshot_path(self.model_snapshot_path(model_version))

    def _load_model_snapshot_path(self, path: Path) -> ZeroDteModelSnapshotArtifact:
        if not path.exists():
            raise ModelStateMissingError(f"Model snapshot not found: {path}")
        return ZeroDteModelSnapshotArtifact.model_validate(_read_json(path))

    def load_model_registry(self) -> ZeroDteModelRegistryArtifact:
        if not self.paths.model_registry_path.exists():
            return ZeroDteModelRegistryArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteModelRegistryArtifact.model_validate(_read_json(self.paths.model_registry_path))

    def save_model_registry(self, registry: ZeroDteModelRegistryArtifact) -> None:
        _atomic_write_json(self.paths.model_registry_path, registry.to_dict())

    def register_model_snapshot(
        self,
        snapshot: ZeroDteModelSnapshotArtifact,
        *,
        status: ModelRegistryStatus = ModelRegistryStatus.INACTIVE,
    ) -> ZeroDteModelRegistryArtifact:
        snapshot_path = self.save_model_snapshot(snapshot)
        registry = self.load_model_registry()
        prior_entries = {entry.model_version: entry for entry in registry.entries}
        prior_entry = prior_entries.get(snapshot.model_version)
        status_to_use = prior_entry.status if prior_entry is not None else status
        promoted_at = prior_entry.promoted_at if prior_entry is not None else None
        path_string = str(snapshot_path.relative_to(self.root_dir))
        prior_created_at = prior_entry.created_at if prior_entry is not None else None
        prior_entries[snapshot.model_version] = ZeroDteModelRegistryEntry(
            model_version=snapshot.model_version,
            trained_through_session=snapshot.trained_through_session,
            assumptions_hash=snapshot.assumptions_hash,
            snapshot_hash=snapshot.snapshot_hash,
            snapshot_path=path_string,
            compatibility=snapshot.compatibility,
            status=status_to_use,
            promoted_at=promoted_at,
            created_at=prior_created_at or datetime.now(timezone.utc),
        )
        registry.entries = [prior_entries[key] for key in sorted(prior_entries)]
        if registry.active_model_version and registry.active_model_version not in prior_entries:
            registry.active_model_version = None
        if (
            registry.previous_active_model_version
            and registry.previous_active_model_version not in prior_entries
        ):
            registry.previous_active_model_version = None
        self.save_model_registry(registry)
        return registry

    def promote_model(
        self,
        model_version: str,
        *,
        as_of: date,
        reason: str | None = None,
        promoted_at: datetime | None = None,
    ) -> ZeroDteModelRegistryArtifact:
        registry = self.load_model_registry()
        entries = {entry.model_version: entry for entry in registry.entries}
        target = entries.get(model_version)
        if target is None:
            raise ModelStateRegistryError(f"Cannot promote unknown model_version={model_version}")
        ts = promoted_at or datetime.now(timezone.utc)
        previous_active = registry.active_model_version
        for version, entry in entries.items():
            if version == model_version:
                entry.status = ModelRegistryStatus.ACTIVE
                entry.promoted_at = ts
            elif entry.status == ModelRegistryStatus.ACTIVE:
                entry.status = ModelRegistryStatus.INACTIVE
        if previous_active != model_version:
            registry.previous_active_model_version = previous_active
        registry.active_model_version = model_version
        registry.compatibility = target.compatibility
        registry.as_of = as_of
        registry.promotion_history.append(
            ZeroDteModelPromotionRecord(
                action=ModelPromotionAction.PROMOTE,
                from_model_version=previous_active,
                to_model_version=model_version,
                promoted_at=ts,
                reason=reason,
                compatibility=target.compatibility,
            )
        )
        registry.entries = [entries[key] for key in sorted(entries)]
        self.save_model_registry(registry)
        return registry

    def rollback_model(
        self,
        *,
        as_of: date,
        reason: str | None = None,
        promoted_at: datetime | None = None,
    ) -> ZeroDteModelRegistryArtifact:
        registry = self.load_model_registry()
        from_version = registry.active_model_version
        to_version = registry.previous_active_model_version
        if not from_version:
            raise ModelStateRegistryError("Cannot rollback: no active model is set")
        if not to_version:
            raise ModelStateRegistryError("Cannot rollback: no previous active model is available")

        entries = {entry.model_version: entry for entry in registry.entries}
        target = entries.get(to_version)
        if target is None:
            raise ModelStateRegistryError(
                f"Cannot rollback: previous_active_model_version={to_version} is missing"
            )

        ts = promoted_at or datetime.now(timezone.utc)
        for version, entry in entries.items():
            if version == to_version:
                entry.status = ModelRegistryStatus.ACTIVE
                entry.promoted_at = ts
            elif entry.status == ModelRegistryStatus.ACTIVE:
                entry.status = ModelRegistryStatus.INACTIVE

        registry.active_model_version = to_version
        registry.previous_active_model_version = from_version
        registry.compatibility = target.compatibility
        registry.as_of = as_of
        registry.promotion_history.append(
            ZeroDteModelPromotionRecord(
                action=ModelPromotionAction.ROLLBACK,
                from_model_version=from_version,
                to_model_version=to_version,
                promoted_at=ts,
                reason=reason,
                compatibility=target.compatibility,
            )
        )
        registry.entries = [entries[key] for key in sorted(entries)]
        self.save_model_registry(registry)
        return registry

    def resolve_active_model(
        self,
        *,
        scoring_session: date,
        min_trained_through_session: date | None = None,
        expected_assumptions_hash: str | None = None,
        required_compatibility: ZeroDteModelCompatibilityMetadata | None = None,
    ) -> ZeroDteModelSnapshotArtifact:
        registry = self.load_model_registry()
        active_version = registry.active_model_version
        if not active_version:
            raise ModelStateRegistryError("No active model version is set in the registry")

        entry = None
        for candidate in registry.entries:
            if candidate.model_version == active_version:
                entry = candidate
                break
        if entry is None:
            raise ModelStateRegistryError(
                f"Active model_version={active_version} not found in registry entries"
            )

        if entry.status != ModelRegistryStatus.ACTIVE:
            raise ModelStateRegistryError(
                f"Registry active model_version={active_version} is not marked active"
            )

        entry_compatibility = entry.compatibility
        registry_compatibility = registry.compatibility
        if required_compatibility is not None:
            if not _compatibility_matches(entry_compatibility, required_compatibility):
                raise ModelStateCompatibilityError(
                    f"Active model {active_version} is incompatible with required compatibility"
                )
            if not _compatibility_matches(registry_compatibility, required_compatibility):
                raise ModelStateCompatibilityError("Model registry compatibility requirements are unmet")

        snapshot_path = Path(entry.snapshot_path)
        if not snapshot_path.is_absolute():
            snapshot_path = self.root_dir / snapshot_path
        snapshot = self._load_model_snapshot_path(snapshot_path)

        if snapshot.model_version != entry.model_version:
            raise ModelStateCompatibilityError("Model snapshot version does not match registry entry")
        if snapshot.snapshot_hash != entry.snapshot_hash:
            raise ModelStateCompatibilityError("Model snapshot hash does not match registry entry")
        if snapshot.assumptions_hash != entry.assumptions_hash:
            raise ModelStateCompatibilityError("Model assumptions hash does not match registry entry")
        if expected_assumptions_hash and snapshot.assumptions_hash != expected_assumptions_hash:
            raise ModelStateCompatibilityError(
                "Model assumptions hash does not match expected assumptions hash"
            )

        if snapshot.trained_through_session >= scoring_session:
            raise ModelStateStaleError(
                "Active model trained_through_session must be before scoring_session"
            )
        if (
            min_trained_through_session is not None
            and snapshot.trained_through_session < min_trained_through_session
        ):
            raise ModelStateStaleError(
                "Active model trained_through_session is stale for requested scoring window"
            )
        return snapshot

    def upsert_probability_curves(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteForwardProbabilityCurveRow | Mapping[str, Any]],
    ) -> ZeroDteProbabilityCurveArtifact:
        existing = self._load_probability_artifact()
        incoming = [ZeroDteForwardProbabilityCurveRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.probability_rows, incoming, _probability_curve_key)
        artifact = ZeroDteProbabilityCurveArtifact(as_of=as_of, probability_rows=merged)
        _atomic_write_json(self.paths.probability_curves_path, artifact.to_dict())
        return artifact

    def upsert_strike_ladders(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteForwardStrikeLadderRow | Mapping[str, Any]],
    ) -> ZeroDteStrikeLadderArtifact:
        existing = self._load_strike_ladder_artifact()
        incoming = [ZeroDteForwardStrikeLadderRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.strike_ladder_rows, incoming, _strike_ladder_key)
        artifact = ZeroDteStrikeLadderArtifact(as_of=as_of, strike_ladder_rows=merged)
        _atomic_write_json(self.paths.strike_ladders_path, artifact.to_dict())
        return artifact

    def upsert_calibration_tables(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteCalibrationTableRow | Mapping[str, Any]],
    ) -> ZeroDteCalibrationArtifact:
        existing = self._load_calibration_artifact()
        incoming = [ZeroDteCalibrationTableRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.calibration_rows, incoming, _calibration_row_key)
        artifact = ZeroDteCalibrationArtifact(as_of=as_of, calibration_rows=merged)
        _atomic_write_json(self.paths.calibration_tables_path, artifact.to_dict())
        return artifact

    def upsert_backtest_summaries(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteBacktestSummaryRow | Mapping[str, Any]],
    ) -> ZeroDteBacktestSummaryArtifact:
        existing = self._load_backtest_summary_artifact()
        incoming = [ZeroDteBacktestSummaryRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.backtest_rows, incoming, _backtest_row_key)
        artifact = ZeroDteBacktestSummaryArtifact(as_of=as_of, backtest_rows=merged)
        _atomic_write_json(self.paths.backtest_summaries_path, artifact.to_dict())
        return artifact

    def upsert_forward_snapshots(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteForwardSnapshotRow | Mapping[str, Any]],
    ) -> ZeroDteForwardSnapshotArtifact:
        existing = self._load_forward_snapshot_artifact()
        incoming = [ZeroDteForwardSnapshotRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.snapshot_rows, incoming, forward_upsert_key)
        artifact = ZeroDteForwardSnapshotArtifact(as_of=as_of, snapshot_rows=merged)
        _atomic_write_json(self.paths.forward_snapshots_path, artifact.to_dict())
        return artifact

    def upsert_trade_ledgers(
        self,
        *,
        as_of: date,
        rows: Sequence[ZeroDteTradeLedgerRow | Mapping[str, Any]],
    ) -> ZeroDteTradeLedgerArtifact:
        existing = self._load_trade_ledger_artifact()
        incoming = [ZeroDteTradeLedgerRow.model_validate(row) for row in rows]
        merged = _merge_rows(existing.trade_rows, incoming, _trade_ledger_key)
        artifact = ZeroDteTradeLedgerArtifact(as_of=as_of, trade_rows=merged)
        _atomic_write_json(self.paths.trade_ledgers_path, artifact.to_dict())
        return artifact

    def _load_probability_artifact(self) -> ZeroDteProbabilityCurveArtifact:
        path = self.paths.probability_curves_path
        if not path.exists():
            return ZeroDteProbabilityCurveArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteProbabilityCurveArtifact.model_validate(_read_json(path))

    def _load_strike_ladder_artifact(self) -> ZeroDteStrikeLadderArtifact:
        path = self.paths.strike_ladders_path
        if not path.exists():
            return ZeroDteStrikeLadderArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteStrikeLadderArtifact.model_validate(_read_json(path))

    def _load_calibration_artifact(self) -> ZeroDteCalibrationArtifact:
        path = self.paths.calibration_tables_path
        if not path.exists():
            return ZeroDteCalibrationArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteCalibrationArtifact.model_validate(_read_json(path))

    def _load_backtest_summary_artifact(self) -> ZeroDteBacktestSummaryArtifact:
        path = self.paths.backtest_summaries_path
        if not path.exists():
            return ZeroDteBacktestSummaryArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteBacktestSummaryArtifact.model_validate(_read_json(path))

    def _load_forward_snapshot_artifact(self) -> ZeroDteForwardSnapshotArtifact:
        path = self.paths.forward_snapshots_path
        if not path.exists():
            return ZeroDteForwardSnapshotArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteForwardSnapshotArtifact.model_validate(_read_json(path))

    def _load_trade_ledger_artifact(self) -> ZeroDteTradeLedgerArtifact:
        path = self.paths.trade_ledgers_path
        if not path.exists():
            return ZeroDteTradeLedgerArtifact(as_of=_DEFAULT_REGISTRY_AS_OF)
        return ZeroDteTradeLedgerArtifact.model_validate(_read_json(path))



def _probability_curve_key(
    row: ZeroDteForwardProbabilityCurveRow,
) -> tuple[str, str, str, str, str, str, str]:
    return (*forward_upsert_key(row), f"{row.strike_return:.10f}")



def _strike_ladder_key(
    row: ZeroDteForwardStrikeLadderRow,
) -> tuple[str, str, str, str, str, str, str]:
    return (*forward_upsert_key(row), f"{row.ladder_rank:04d}")



def _calibration_row_key(row: ZeroDteCalibrationTableRow) -> tuple[str, str, str, str]:
    return (
        row.model_version,
        row.assumptions_hash,
        f"{row.risk_tier:.10f}",
        row.probability_bin,
    )



def _backtest_row_key(row: ZeroDteBacktestSummaryRow) -> tuple[str, str, str, str, str]:
    return (
        row.model_version,
        row.assumptions_hash,
        row.session_date.isoformat(),
        f"{row.risk_tier:.10f}",
        row.exit_mode.value,
    )



def _trade_ledger_key(
    row: ZeroDteTradeLedgerRow,
) -> tuple[str, str, str, str, str, str, str, str, str]:
    return (
        *forward_upsert_key(row),
        row.exit_mode.value,
        row.contract_symbol or "",
        _to_utc_iso(row.entry_ts),
    )


__all__ = [
    "ModelStateCompatibilityError",
    "ModelStateMissingError",
    "ModelStateRegistryError",
    "ModelStateStaleError",
    "ZeroDteArtifactPaths",
    "ZeroDteArtifactStore",
    "ZeroDteArtifactsError",
    "build_model_snapshot",
    "build_zero_dte_artifact_paths",
    "canonicalize_payload",
    "compute_assumptions_hash",
    "compute_snapshot_hash",
    "forward_upsert_key",
]
