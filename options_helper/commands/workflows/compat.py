from __future__ import annotations

from options_helper.commands import workflows_legacy as legacy


def sync_legacy_seams() -> None:
    """Keep legacy module seams synchronized with package-level monkeypatch targets."""
    import options_helper.commands.workflows as workflows_pkg

    legacy.datetime = workflows_pkg.datetime
    legacy.safe_next_earnings_date = workflows_pkg.safe_next_earnings_date
    legacy.run_snapshot_options_job = workflows_pkg.run_snapshot_options_job
    legacy._position_metrics = workflows_pkg._position_metrics


__all__ = ["sync_legacy_seams"]
