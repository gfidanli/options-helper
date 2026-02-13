from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams


@wraps(legacy.briefing)
def briefing(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.briefing(*args, **kwargs)


@wraps(legacy.dashboard)
def dashboard(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.dashboard(*args, **kwargs)


__all__ = ["briefing", "dashboard"]
