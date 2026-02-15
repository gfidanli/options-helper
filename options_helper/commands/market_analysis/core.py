from __future__ import annotations

import typer

from options_helper.commands.market_analysis.core_impl import exposure, iv_surface, levels, tail_risk


def register(app: typer.Typer) -> None:
    app.command("tail-risk")(tail_risk)
    app.command("iv-surface")(iv_surface)
    app.command("exposure")(exposure)
    app.command("levels")(levels)


__all__ = ["register", "tail_risk", "iv_surface", "exposure", "levels"]
