from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import workflows_legacy as legacy
from options_helper.commands.workflows.compat import sync_legacy_seams


@wraps(legacy.research)
def research(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.research(*args, **kwargs)


@wraps(legacy.refresh_candles)
def refresh_candles(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.refresh_candles(*args, **kwargs)


@wraps(legacy.analyze)
def analyze(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.analyze(*args, **kwargs)


@wraps(legacy.watch)
def watch(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.watch(*args, **kwargs)


__all__ = ["research", "refresh_candles", "analyze", "watch"]
