from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.compat import sync_legacy_seams


@wraps(legacy.daily_performance)
def daily_performance(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.daily_performance(*args, **kwargs)


@wraps(legacy.snapshot_options)
def snapshot_options(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.snapshot_options(*args, **kwargs)


@wraps(legacy.earnings)
def earnings(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.earnings(*args, **kwargs)


@wraps(legacy.refresh_earnings)
def refresh_earnings(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.refresh_earnings(*args, **kwargs)


__all__ = [
    "daily_performance",
    "snapshot_options",
    "earnings",
    "refresh_earnings",
]
