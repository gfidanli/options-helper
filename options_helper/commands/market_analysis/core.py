from __future__ import annotations

import typer

from options_helper.commands import market_analysis_legacy as legacy

tail_risk = legacy.tail_risk
iv_surface = legacy.iv_surface
exposure = legacy.exposure
levels = legacy.levels


def register(app: typer.Typer) -> None:
    app.command("tail-risk")(tail_risk)
    app.command("iv-surface")(iv_surface)
    app.command("exposure")(exposure)
    app.command("levels")(levels)


__all__ = ["register", "tail_risk", "iv_surface", "exposure", "levels"]
