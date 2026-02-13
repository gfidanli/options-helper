from __future__ import annotations

import typer

from options_helper.commands import technicals_legacy as _legacy
from options_helper.commands.technicals.backtesting import (
    register as register_backtesting,
    technicals_compute_indicators,
    technicals_optimize,
    technicals_run_all,
    technicals_walk_forward,
)
from options_helper.commands.technicals.extension_stats import (
    register as register_extension_stats,
    technicals_extension_stats,
)
from options_helper.commands.technicals.scans import (
    register as register_scans,
    technicals_msb_scan,
    technicals_sfp_scan,
)
from options_helper.commands.technicals.strategy_model import (
    register as register_strategy_model,
    technicals_strategy_model,
)

app = typer.Typer(help="Technical indicators + backtesting/optimization.")

register_backtesting(app)
register_scans(app)
register_extension_stats(app)
register_strategy_model(app)

# Compatibility seam for tests that monkeypatch
# `options_helper.commands.technicals.cli_deps.*`.
cli_deps = _legacy.cli_deps

__all__ = [
    "app",
    "cli_deps",
    "technicals_compute_indicators",
    "technicals_sfp_scan",
    "technicals_msb_scan",
    "technicals_extension_stats",
    "technicals_optimize",
    "technicals_walk_forward",
    "technicals_run_all",
    "technicals_strategy_model",
]
