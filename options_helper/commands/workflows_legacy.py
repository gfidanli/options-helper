# ruff: noqa: F401
from __future__ import annotations

from contextlib import contextmanager
import json
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from rich.console import Console

import options_helper.cli_deps as cli_deps
from options_helper.analysis.advice import Advice, PositionMetrics, advise
from options_helper.analysis.confluence import ConfluenceScore, score_confluence
from options_helper.analysis.extension_scan import compute_current_extension_percentile
from options_helper.analysis.performance import compute_daily_performance_quote
from options_helper.analysis.portfolio_risk import PortfolioExposure, compute_portfolio_exposure, run_stress
from options_helper.analysis.research import (
    Direction,
    UnderlyingSetup,
    analyze_underlying,
    build_confluence_inputs,
    choose_expiry,
    compute_volatility_context,
    select_option_candidate,
    suggest_trade_levels,
)
from options_helper.commands.common import _build_stress_scenarios, _parse_date
from options_helper.commands.position_metrics import _extract_float, _position_metrics
from options_helper.data.candles import close_asof, last_close
from options_helper.data.confluence_config import ConfigError as ConfluenceConfigError, load_confluence_config
from options_helper.data.earnings import EarningsRecord, safe_next_earnings_date
from options_helper.data.market_types import DataFetchError
from options_helper.data.options_snapshots import OptionsSnapshotStore, find_snapshot_row
from options_helper.data.providers.runtime import get_default_provider_name
from options_helper.data.technical_backtesting_config import (
    ConfigError as TechnicalConfigError,
    load_technical_backtesting_config,
)
from options_helper.data.yf_client import contract_row_by_strike
from options_helper.models import MultiLegPosition, OptionType, Position
from options_helper.pipelines.visibility_jobs import (
    VisibilityJobParameterError,
    run_snapshot_options_job,
)
from options_helper.reporting import MultiLegSummary, render_multi_leg_positions, render_positions, render_summary
from options_helper.storage import load_portfolio
from options_helper.watchlists import load_watchlists

if TYPE_CHECKING:
    import pandas as pd


pd: object | None = None

JOB_SNAPSHOT_OPTIONS = "snapshot_options"
ASSET_OPTIONS_SNAPSHOTS = "options_snapshots"

NOOP_LEDGER_WARNING = (
    "Run ledger disabled for filesystem storage backend (NoopRunLogger active)."
)

_SAVED_SNAPSHOT_RE = re.compile(r"^([A-Z0-9._-]+)\s+\d{4}-\d{2}-\d{2}: saved\b")
_WARNING_SYMBOL_RE = re.compile(r"warning:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)
_ERROR_SYMBOL_RE = re.compile(r"error:\s*([A-Z0-9._-]+):", flags=re.IGNORECASE)


def _ensure_pandas() -> None:
    global pd
    if pd is None:
        import pandas as _pd

        pd = _pd


def _is_noop_run_logger(run_logger: object) -> bool:
    return run_logger.__class__.__name__ == "NoopRunLogger"


def _strip_rich_markup(text: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", text).strip()


def _snapshot_status_by_symbol(*, symbols: list[str], messages: list[str]) -> dict[str, str]:
    status_by_symbol = {sym.upper(): "skipped" for sym in symbols}
    for message in messages:
        plain = _strip_rich_markup(message)
        saved_match = _SAVED_SNAPSHOT_RE.match(plain)
        if saved_match:
            status_by_symbol[saved_match.group(1).upper()] = "success"
            continue

        error_match = _ERROR_SYMBOL_RE.search(plain)
        if error_match:
            status_by_symbol[error_match.group(1).upper()] = "failed"
            continue

        if "skipping snapshot" not in plain.lower():
            continue
        warning_match = _WARNING_SYMBOL_RE.search(plain)
        if warning_match and status_by_symbol.get(warning_match.group(1).upper()) != "success":
            status_by_symbol[warning_match.group(1).upper()] = "skipped"
    return status_by_symbol


@contextmanager
def _observed_run(*, console: Console, job_name: str, args: dict[str, Any]):
    run_logger = cli_deps.build_run_logger(
        job_name=job_name,
        provider=get_default_provider_name(),
        args=args,
    )
    if _is_noop_run_logger(run_logger):
        console.print(f"[yellow]Warning:[/yellow] {NOOP_LEDGER_WARNING}")
    try:
        yield run_logger
    except typer.Exit as exc:
        exit_code = int(getattr(exc, "exit_code", 1) or 0)
        if exit_code == 0:
            run_logger.finalize_success()
        else:
            run_logger.finalize_failure(exc.__cause__ if exc.__cause__ is not None else exc)
        raise
    except Exception as exc:  # noqa: BLE001
        run_logger.finalize_failure(exc)
        raise
    else:
        run_logger.finalize_success()


def _load_workflows_command(module_name: str, function_name: str):
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def register(app: typer.Typer) -> None:
    app.command("daily")(daily_performance)
    app.command("snapshot-options")(snapshot_options)
    app.command("earnings")(earnings)
    app.command("refresh-earnings")(refresh_earnings)
    app.command("research")(research)
    app.command("refresh-candles")(refresh_candles)
    app.command("analyze")(analyze)
    app.command("watch")(watch)


daily_performance = _load_workflows_command("options_helper.commands.workflows.core", "daily_performance")


snapshot_options = _load_workflows_command("options_helper.commands.workflows.core", "snapshot_options")


earnings = _load_workflows_command("options_helper.commands.workflows.core", "earnings")


refresh_earnings = _load_workflows_command("options_helper.commands.workflows.core", "refresh_earnings")


research = _load_workflows_command("options_helper.commands.workflows.research", "research")


refresh_candles = _load_workflows_command("options_helper.commands.workflows.research", "refresh_candles")


analyze = _load_workflows_command("options_helper.commands.workflows.research", "analyze")


watch = _load_workflows_command("options_helper.commands.workflows.research", "watch")


def _render_portfolio_risk(
    console: Console,
    exposure: PortfolioExposure,
    *,
    stress_spot_pct: list[float],
    stress_vol_pp: float,
    stress_days: int,
) -> None:
    from rich.table import Table

    def _fmt_num(val: float | None, *, digits: int = 2) -> str:
        if val is None:
            return "-"
        return f"{val:,.{digits}f}"

    def _fmt_money(val: float | None) -> str:
        if val is None:
            return "-"
        return f"${val:,.2f}"

    def _fmt_pct(val: float | None) -> str:
        if val is None:
            return "-"
        return f"{val:.1%}"

    table = Table(title="Portfolio Greeks (best-effort)")
    table.add_column("As-of")
    table.add_column("Delta (shares)", justify="right")
    table.add_column("Theta/day ($)", justify="right")
    table.add_column("Vega ($/IV)", justify="right")
    table.add_row(
        "-" if exposure.as_of is None else exposure.as_of.isoformat(),
        _fmt_num(exposure.total_delta_shares),
        _fmt_money(exposure.total_theta_dollars_per_day),
        _fmt_money(exposure.total_vega_dollars_per_iv),
    )
    console.print(table)

    if exposure.assumptions:
        console.print("Assumptions: " + "; ".join(exposure.assumptions))
    if exposure.warnings:
        console.print("[yellow]Warnings:[/yellow] " + "; ".join(exposure.warnings))

    scenarios = _build_stress_scenarios(
        stress_spot_pct=stress_spot_pct,
        stress_vol_pp=stress_vol_pp,
        stress_days=stress_days,
    )
    if not scenarios:
        return

    stress_results = run_stress(exposure, scenarios)
    stress_table = Table(title="Portfolio Stress (best-effort)")
    stress_table.add_column("Scenario")
    stress_table.add_column("PnL $", justify="right")
    stress_table.add_column("PnL %", justify="right")
    stress_table.add_column("Notes")

    for result in stress_results:
        notes = ", ".join(result.warnings) if result.warnings else "-"
        stress_table.add_row(
            result.name,
            _fmt_money(result.pnl),
            _fmt_pct(result.pnl_pct),
            notes,
        )
    console.print(stress_table)
