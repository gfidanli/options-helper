from __future__ import annotations

import typer

from options_helper.commands import reports_legacy as _legacy
from options_helper.commands.reports.chain import chain_report, compare_report, roll_plan
from options_helper.commands.reports.daily import briefing, dashboard
from options_helper.commands.reports.flow import flow_report
from options_helper.commands.reports.pack import report_pack

# Compatibility seams expected by tests and callers that monkeypatch module globals.
safe_next_earnings_date = _legacy.safe_next_earnings_date
run_flow_report_job = _legacy.run_flow_report_job
run_briefing_job = _legacy.run_briefing_job
run_dashboard_job = _legacy.run_dashboard_job
render_dashboard_report = _legacy.render_dashboard_report
technicals_extension_stats = _legacy.run_extension_stats_for_symbol


def register(app: typer.Typer) -> None:
    app.command("flow")(flow_report)
    app.command("chain-report")(chain_report)
    app.command("compare")(compare_report)
    app.command("report-pack")(report_pack)
    app.command("briefing")(briefing)
    app.command("dashboard")(dashboard)
    app.command("roll-plan")(roll_plan)


__all__ = [
    "register",
    "safe_next_earnings_date",
    "run_flow_report_job",
    "run_briefing_job",
    "run_dashboard_job",
    "render_dashboard_report",
    "technicals_extension_stats",
    "flow_report",
    "chain_report",
    "compare_report",
    "report_pack",
    "briefing",
    "dashboard",
    "roll_plan",
]
