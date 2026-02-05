from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_STORAGE_BACKEND: ContextVar[str] = ContextVar("options_helper_default_storage_backend", default="duckdb")
_DEFAULT_DUCKDB_PATH: ContextVar[Path] = ContextVar(
    "options_helper_default_duckdb_path",
    default=Path("data/warehouse/options.duckdb"),
)


@dataclass(frozen=True)
class StorageRuntimeConfig:
    backend: str
    duckdb_path: Path


def get_default_storage_backend() -> str:
    return _DEFAULT_STORAGE_BACKEND.get()


def set_default_storage_backend(name: str | None) -> Token[str]:
    cleaned = (name or "duckdb").strip().lower()
    if cleaned not in {"filesystem", "duckdb"}:
        cleaned = "duckdb"
    return _DEFAULT_STORAGE_BACKEND.set(cleaned)


def reset_default_storage_backend(token: Token[str]) -> None:
    _DEFAULT_STORAGE_BACKEND.reset(token)


def get_default_duckdb_path() -> Path:
    return _DEFAULT_DUCKDB_PATH.get()


def set_default_duckdb_path(path: Path | str | None) -> Token[Path]:
    val = Path(path) if path is not None else Path("data/warehouse/options.duckdb")
    return _DEFAULT_DUCKDB_PATH.set(val)


def reset_default_duckdb_path(token: Token[Path]) -> None:
    _DEFAULT_DUCKDB_PATH.reset(token)


def get_storage_runtime_config() -> StorageRuntimeConfig:
    return StorageRuntimeConfig(backend=get_default_storage_backend(), duckdb_path=get_default_duckdb_path())
