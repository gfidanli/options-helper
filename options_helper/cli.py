from __future__ import annotations

from enum import Enum
from pathlib import Path

import typer

from options_helper.commands.backtest import app as backtest_app
from options_helper.commands.db import app as db_app
from options_helper.commands.debug import app as debug_app
from options_helper.commands.derived import app as derived_app
from options_helper.commands.events import app as events_app
from options_helper.commands.ingest import app as ingest_app
from options_helper.commands.intraday import app as intraday_app
from options_helper.commands.journal import app as journal_app
from options_helper.commands.market_analysis import app as market_analysis_app
from options_helper.commands.portfolio import register as register_portfolio_commands
from options_helper.commands.reports import register as register_report_commands
from options_helper.commands.scanner import app as scanner_app
from options_helper.commands.stream import app as stream_app
from options_helper.commands.technicals import app as technicals_app
from options_helper.commands.ui import register as register_ui_commands
from options_helper.commands.watchlists import app as watchlists_app
from options_helper.commands.workflows import register as register_workflow_commands
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.data.storage_runtime import (
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)
from options_helper.observability import finalize_run_logger, setup_run_logger


class StorageBackend(str, Enum):
    filesystem = "filesystem"
    duckdb = "duckdb"


app = typer.Typer(add_completion=False)
app.add_typer(watchlists_app, name="watchlists")
app.add_typer(debug_app, name="debug")
app.add_typer(derived_app, name="derived")
app.add_typer(technicals_app, name="technicals")
app.add_typer(scanner_app, name="scanner")
app.add_typer(journal_app, name="journal")
app.add_typer(market_analysis_app, name="market-analysis")
app.add_typer(backtest_app, name="backtest")
app.add_typer(intraday_app, name="intraday")
app.add_typer(events_app, name="events")
app.add_typer(stream_app, name="stream")
app.add_typer(db_app, name="db")
app.add_typer(ingest_app, name="ingest")
register_portfolio_commands(app)
register_report_commands(app)
register_workflow_commands(app)
register_ui_commands(app)


@app.callback()
def main(
    ctx: typer.Context,
    log_dir: Path = typer.Option(
        Path("data/logs"),
        "--log-dir",
        help="Directory to write per-command logs.",
    ),
    provider: str = typer.Option(
        "alpaca",
        "--provider",
        help="Market data provider (default: alpaca).",
    ),
    storage: StorageBackend = typer.Option(
        StorageBackend.duckdb,
        "--storage",
        help="Storage backend (filesystem or duckdb).",
    ),
    duckdb_path: Path | None = typer.Option(
        None,
        "--duckdb-path",
        help="DuckDB file path (defaults to data/warehouse/options.duckdb).",
    ),
) -> None:
    command_name = ctx.info_name or "options-helper"
    if ctx.invoked_subcommand:
        command_name = f"{command_name} {ctx.invoked_subcommand}"
    run_logger = setup_run_logger(log_dir, command_name)
    provider_token = set_default_provider_name(provider)
    storage_token = set_default_storage_backend(storage.value)
    duckdb_path_token = set_default_duckdb_path(duckdb_path)

    def _on_close() -> None:
        reset_default_duckdb_path(duckdb_path_token)
        reset_default_storage_backend(storage_token)
        reset_default_provider_name(provider_token)
        if storage is StorageBackend.duckdb:
            from options_helper.data.store_factory import close_warehouses

            close_warehouses()
        if run_logger is not None:
            finalize_run_logger(run_logger)

    ctx.call_on_close(_on_close)

    if run_logger is None:
        return
