from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams


@wraps(legacy.flow_report)
def flow_report(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.flow_report(*args, **kwargs)


__all__ = ["flow_report"]
