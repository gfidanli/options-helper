from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from options_helper.schemas.strategy_modeling_profile import (
    STRATEGY_MODELING_PROFILE_SCHEMA_VERSION,
    StrategyModelingProfile,
    StrategyModelingProfileStore,
)


DEFAULT_STRATEGY_MODELING_PROFILE_PATH = Path("config/strategy_modeling_profiles.json")


class StrategyModelingProfileStoreError(ValueError):
    """Raised when strategy-modeling profile persistence fails validation."""


def list_strategy_modeling_profiles(path: Path | str) -> list[str]:
    store = _load_store(Path(path), allow_missing=True)
    return sorted(store.profiles)


def load_strategy_modeling_profile(path: Path | str, name: str) -> StrategyModelingProfile:
    profile_name = _normalize_profile_name(name)
    store = _load_store(Path(path), allow_missing=False)
    profile = store.profiles.get(profile_name)
    if profile is None:
        available = sorted(store.profiles)
        if available:
            available_str = ", ".join(available)
            raise StrategyModelingProfileStoreError(
                f"Profile {profile_name!r} was not found. Available profiles: {available_str}."
            )
        raise StrategyModelingProfileStoreError(
            f"Profile {profile_name!r} was not found. The profile store is empty."
        )
    return profile


def save_strategy_modeling_profile(
    path: Path | str,
    name: str,
    profile: StrategyModelingProfile | Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    profile_name = _normalize_profile_name(name)
    normalized_profile = _normalize_profile(profile)
    store_path = Path(path)
    store = _load_store(store_path, allow_missing=True)

    profiles = dict(store.profiles)
    if profile_name in profiles and not overwrite:
        raise StrategyModelingProfileStoreError(
            f"Profile {profile_name!r} already exists. Re-run with overwrite enabled to replace it."
        )

    profiles[profile_name] = normalized_profile
    updated_store = StrategyModelingProfileStore(
        schema_version=STRATEGY_MODELING_PROFILE_SCHEMA_VERSION,
        updated_at=datetime.now(timezone.utc),
        profiles=profiles,
    )
    _atomic_write_store(store_path, updated_store)


def _normalize_profile(
    profile: StrategyModelingProfile | Mapping[str, Any],
) -> StrategyModelingProfile:
    if isinstance(profile, StrategyModelingProfile):
        return profile
    try:
        return StrategyModelingProfile.model_validate(dict(profile))
    except ValidationError as exc:
        raise StrategyModelingProfileStoreError(
            f"Invalid strategy-modeling profile payload: {_format_validation_error(exc)}"
        ) from exc


def _normalize_profile_name(value: str) -> str:
    token = str(value or "").strip()
    if not token:
        raise StrategyModelingProfileStoreError("Profile name must be non-empty.")
    return token


def _load_store(path: Path, *, allow_missing: bool) -> StrategyModelingProfileStore:
    if not path.exists():
        if allow_missing:
            return StrategyModelingProfileStore.empty()
        raise StrategyModelingProfileStoreError(
            f"Profile store file does not exist: {path}. "
            "Save a profile first or pass a valid --profile-path."
        )

    raw_payload = _read_json_payload(path)
    version = raw_payload.get("schema_version")
    if version != STRATEGY_MODELING_PROFILE_SCHEMA_VERSION:
        raise StrategyModelingProfileStoreError(
            f"Unsupported profile store schema_version {version!r} in {path}. "
            f"Expected {STRATEGY_MODELING_PROFILE_SCHEMA_VERSION}."
        )

    try:
        return StrategyModelingProfileStore.model_validate(raw_payload)
    except ValidationError as exc:
        raise StrategyModelingProfileStoreError(
            f"Invalid profile store format in {path}: {_format_validation_error(exc)}"
        ) from exc


def _read_json_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StrategyModelingProfileStoreError(
            f"Profile store file {path} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})."
        ) from exc
    except OSError as exc:
        raise StrategyModelingProfileStoreError(f"Failed to read profile store file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise StrategyModelingProfileStoreError(
            f"Profile store file {path} must contain a JSON object at the top level."
        )
    return payload


def _atomic_write_store(path: Path, store: StrategyModelingProfileStore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    payload = json.dumps(store.to_json_payload(), indent=2, sort_keys=True)
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def _format_validation_error(exc: ValidationError) -> str:
    errors: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        msg = str(error.get("msg", "invalid value"))
        errors.append(f"{loc}: {msg}")
    return "; ".join(errors)


__all__ = [
    "DEFAULT_STRATEGY_MODELING_PROFILE_PATH",
    "StrategyModelingProfileStoreError",
    "list_strategy_modeling_profiles",
    "load_strategy_modeling_profile",
    "save_strategy_modeling_profile",
]
