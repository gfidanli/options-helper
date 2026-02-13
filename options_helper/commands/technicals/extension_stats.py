from __future__ import annotations

import typer

from options_helper.commands import technicals_legacy as legacy

technicals_extension_stats = legacy.technicals_extension_stats


def register(app: typer.Typer) -> None:
    app.command("extension-stats")(technicals_extension_stats)


__all__ = ["register", "technicals_extension_stats"]
