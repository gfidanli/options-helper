from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams


@wraps(legacy.report_pack)
def report_pack(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.report_pack(*args, **kwargs)


__all__ = ["report_pack"]
