from __future__ import annotations

import typer

from options_helper.commands import workflows_legacy as _legacy
from options_helper.commands.workflows.core import daily_performance, earnings, refresh_earnings, snapshot_options
from options_helper.commands.workflows.research import analyze, refresh_candles, research, watch

# Compatibility seams expected by tests and callers that monkeypatch module globals.
datetime = _legacy.datetime
safe_next_earnings_date = _legacy.safe_next_earnings_date
run_snapshot_options_job = _legacy.run_snapshot_options_job
_position_metrics = _legacy._position_metrics


def register(app: typer.Typer) -> None:
    app.command("daily")(daily_performance)
    app.command("snapshot-options")(snapshot_options)
    app.command("earnings")(earnings)
    app.command("refresh-earnings")(refresh_earnings)
    app.command("research")(research)
    app.command("refresh-candles")(refresh_candles)
    app.command("analyze")(analyze)
    app.command("watch")(watch)


__all__ = [
    "register",
    "datetime",
    "safe_next_earnings_date",
    "run_snapshot_options_job",
    "_position_metrics",
    "daily_performance",
    "snapshot_options",
    "earnings",
    "refresh_earnings",
    "research",
    "refresh_candles",
    "analyze",
    "watch",
]
