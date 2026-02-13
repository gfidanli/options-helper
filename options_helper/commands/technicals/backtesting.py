from __future__ import annotations

import typer

from options_helper.commands import technicals_legacy as legacy

technicals_compute_indicators = legacy.technicals_compute_indicators
technicals_optimize = legacy.technicals_optimize
technicals_walk_forward = legacy.technicals_walk_forward
technicals_run_all = legacy.technicals_run_all


def register(app: typer.Typer) -> None:
    app.command("compute-indicators")(technicals_compute_indicators)
    app.command("optimize")(technicals_optimize)
    app.command("walk-forward")(technicals_walk_forward)
    app.command("run-all")(technicals_run_all)


__all__ = [
    "register",
    "technicals_compute_indicators",
    "technicals_optimize",
    "technicals_walk_forward",
    "technicals_run_all",
]
