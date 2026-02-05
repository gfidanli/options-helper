from __future__ import annotations

from pathlib import Path

import typer

from options_helper.commands.backtest import app as backtest_app
from options_helper.commands.derived import app as derived_app
from options_helper.commands.events import app as events_app
from options_helper.commands.intraday import app as intraday_app
from options_helper.commands.journal import app as journal_app
from options_helper.commands.portfolio import register as register_portfolio_commands
from options_helper.commands.reports import register as register_report_commands
from options_helper.commands.scanner import app as scanner_app
from options_helper.commands.stream import app as stream_app
from options_helper.commands.technicals import app as technicals_app
from options_helper.commands.watchlists import app as watchlists_app
from options_helper.commands.workflows import register as register_workflow_commands
from options_helper.data.providers.runtime import reset_default_provider_name, set_default_provider_name
from options_helper.observability import finalize_run_logger, setup_run_logger

app = typer.Typer(add_completion=False)
app.add_typer(watchlists_app, name="watchlists")
app.add_typer(derived_app, name="derived")
app.add_typer(technicals_app, name="technicals")
app.add_typer(scanner_app, name="scanner")
app.add_typer(journal_app, name="journal")
app.add_typer(backtest_app, name="backtest")
app.add_typer(intraday_app, name="intraday")
app.add_typer(events_app, name="events")
app.add_typer(stream_app, name="stream")
register_portfolio_commands(app)
register_report_commands(app)
register_workflow_commands(app)


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
) -> None:
    command_name = ctx.info_name or "options-helper"
    if ctx.invoked_subcommand:
        command_name = f"{command_name} {ctx.invoked_subcommand}"
    run_logger = setup_run_logger(log_dir, command_name)
    provider_token = set_default_provider_name(provider)

    def _on_close() -> None:
        reset_default_provider_name(provider_token)
        if run_logger is not None:
            finalize_run_logger(run_logger)

    ctx.call_on_close(_on_close)

    if run_logger is None:
        return
