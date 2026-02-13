from __future__ import annotations

import typer

from options_helper.commands.technicals.extension_stats_legacy import technicals_extension_stats


def register(app: typer.Typer) -> None:
    app.command("extension-stats")(technicals_extension_stats)


__all__ = ["register", "technicals_extension_stats"]
