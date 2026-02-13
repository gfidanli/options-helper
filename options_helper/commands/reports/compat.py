from __future__ import annotations

from options_helper.commands import reports_legacy as legacy


def sync_legacy_seams() -> None:
    """Keep legacy module seams synchronized with package-level monkeypatch targets."""
    import options_helper.commands.reports as reports_pkg

    legacy.safe_next_earnings_date = reports_pkg.safe_next_earnings_date
    legacy.run_flow_report_job = reports_pkg.run_flow_report_job
    legacy.run_briefing_job = reports_pkg.run_briefing_job
    legacy.run_dashboard_job = reports_pkg.run_dashboard_job
    legacy.render_dashboard_report = reports_pkg.render_dashboard_report
    legacy.run_extension_stats_for_symbol = reports_pkg.technicals_extension_stats


__all__ = ["sync_legacy_seams"]
