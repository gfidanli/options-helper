from __future__ import annotations

import typer

from options_helper.commands import technicals_legacy as legacy

technicals_strategy_model = legacy.technicals_strategy_model


def register(app: typer.Typer) -> None:
    app.command("strategy-model")(technicals_strategy_model)


__all__ = ["register", "technicals_strategy_model"]
