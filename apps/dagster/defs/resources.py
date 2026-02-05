from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DagsterPaths:
    """Filesystem paths shared by Dagster orchestration resources."""

    repo_root: Path
    data_dir: Path
    portfolio_path: Path
    watchlists_path: Path
    duckdb_path: Path


@dataclass(frozen=True)
class DagsterRuntimeConfig:
    """Runtime configuration placeholders used by future assets/jobs."""

    provider: str


def _path_from_env(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip()).expanduser()


def build_paths() -> DagsterPaths:
    repo_root = Path(__file__).resolve().parents[3]
    data_dir = _path_from_env("OPTIONS_HELPER_DATA_DIR", repo_root / "data")
    return DagsterPaths(
        repo_root=repo_root,
        data_dir=data_dir,
        portfolio_path=_path_from_env("OPTIONS_HELPER_PORTFOLIO_PATH", repo_root / "portfolio.json"),
        watchlists_path=_path_from_env(
            "OPTIONS_HELPER_WATCHLISTS_PATH",
            data_dir / "watchlists.json",
        ),
        duckdb_path=_path_from_env(
            "OPTIONS_HELPER_DUCKDB_PATH",
            data_dir / "warehouse" / "options.duckdb",
        ),
    )


def build_resources() -> dict[str, object]:
    """Return a minimal resources mapping for Dagster `Definitions`."""

    return {
        "paths": build_paths(),
        "runtime_config": DagsterRuntimeConfig(
            provider=(os.environ.get("OPTIONS_HELPER_PROVIDER") or "alpaca").strip() or "alpaca",
        ),
    }
