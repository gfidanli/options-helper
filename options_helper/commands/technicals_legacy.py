from __future__ import annotations

import importlib

import typer

import options_helper.cli_deps as _cli_deps
from options_helper.commands.technicals.extension_stats_legacy import (
    technicals_extension_stats as _technicals_extension_stats_impl,
)

cli_deps = _cli_deps

app = typer.Typer(help="Technical indicators + backtesting/optimization.")


def _load_command(module_name: str, function_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


technicals_compute_indicators = _load_command(
    "options_helper.commands.technicals.backtesting",
    "technicals_compute_indicators",
)
app.command("compute-indicators")(technicals_compute_indicators)

technicals_optimize = _load_command(
    "options_helper.commands.technicals.backtesting",
    "technicals_optimize",
)
app.command("optimize")(technicals_optimize)

technicals_walk_forward = _load_command(
    "options_helper.commands.technicals.backtesting",
    "technicals_walk_forward",
)
app.command("walk-forward")(technicals_walk_forward)

technicals_run_all = _load_command(
    "options_helper.commands.technicals.backtesting",
    "technicals_run_all",
)
app.command("run-all")(technicals_run_all)

technicals_sfp_scan = _load_command("options_helper.commands.technicals.scans", "technicals_sfp_scan")
app.command("sfp-scan")(technicals_sfp_scan)

technicals_msb_scan = _load_command("options_helper.commands.technicals.scans", "technicals_msb_scan")
app.command("msb-scan")(technicals_msb_scan)

technicals_extension_stats = _technicals_extension_stats_impl
app.command("extension-stats")(technicals_extension_stats)

technicals_strategy_model = _load_command(
    "options_helper.commands.technicals.strategy_model",
    "technicals_strategy_model",
)
app.command("strategy-model")(technicals_strategy_model)


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
