from __future__ import annotations

import typer

from options_helper.commands import technicals_legacy as legacy

technicals_sfp_scan = legacy.technicals_sfp_scan
technicals_msb_scan = legacy.technicals_msb_scan


def register(app: typer.Typer) -> None:
    app.command("sfp-scan")(technicals_sfp_scan)
    app.command("msb-scan")(technicals_msb_scan)


__all__ = ["register", "technicals_sfp_scan", "technicals_msb_scan"]
