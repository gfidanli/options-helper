from __future__ import annotations

from functools import wraps
from typing import Any

from options_helper.commands import reports_legacy as legacy
from options_helper.commands.reports.compat import sync_legacy_seams


@wraps(legacy.chain_report)
def chain_report(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.chain_report(*args, **kwargs)


@wraps(legacy.compare_report)
def compare_report(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.compare_report(*args, **kwargs)


@wraps(legacy.roll_plan)
def roll_plan(*args: Any, **kwargs: Any):
    sync_legacy_seams()
    return legacy.roll_plan(*args, **kwargs)


__all__ = ["chain_report", "compare_report", "roll_plan"]
