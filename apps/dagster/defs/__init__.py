from __future__ import annotations

from dagster import Definitions

from .assets import ASSET_DEFINITIONS
from .checks import build_asset_checks
from .jobs import build_jobs
from .resources import build_resources
from .schedules import build_schedules


def build_definitions() -> Definitions:
    return Definitions(
        assets=list(ASSET_DEFINITIONS),
        asset_checks=list(build_asset_checks()),
        jobs=list(build_jobs()),
        schedules=list(build_schedules()),
        resources=build_resources(),
    )


defs = build_definitions()

__all__ = ["build_definitions", "defs"]
