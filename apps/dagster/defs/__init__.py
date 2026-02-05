from __future__ import annotations

from dagster import Definitions

from .jobs import build_jobs
from .resources import build_resources
from .schedules import build_schedules


def build_definitions() -> Definitions:
    """Assemble minimal Dagster definitions for optional orchestration."""

    return Definitions(
        assets=[],
        jobs=list(build_jobs()),
        schedules=list(build_schedules()),
        resources=build_resources(),
    )


defs = build_definitions()

__all__ = ["build_definitions", "defs"]
