from __future__ import annotations

import typer

from options_helper.commands import market_analysis_legacy as legacy

zero_dte_put_study = legacy.zero_dte_put_study
zero_dte_put_forward_snapshot = legacy.zero_dte_put_forward_snapshot


def register(app: typer.Typer) -> None:
    app.command("zero-dte-put-study")(zero_dte_put_study)
    app.command("zero-dte-put-forward-snapshot")(zero_dte_put_forward_snapshot)


__all__ = ["register", "zero_dte_put_study", "zero_dte_put_forward_snapshot"]
