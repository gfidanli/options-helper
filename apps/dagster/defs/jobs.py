from __future__ import annotations

from collections.abc import Sequence

from dagster import define_asset_job


daily_visibility_job = define_asset_job(
    name="daily_visibility_job",
    description=(
        "Daily visibility pipeline: candles -> options bars -> snapshots -> flow -> derived -> briefing."
    ),
)


def build_jobs() -> Sequence[object]:
    return (daily_visibility_job,)


__all__ = ["build_jobs", "daily_visibility_job"]
