from __future__ import annotations

from collections.abc import Sequence

from dagster import build_schedule_from_partitioned_job

from .jobs import daily_visibility_job


daily_visibility_schedule = build_schedule_from_partitioned_job(
    job=daily_visibility_job,
    name="daily_visibility_schedule",
)


def build_schedules() -> Sequence[object]:
    return (daily_visibility_schedule,)


__all__ = ["build_schedules", "daily_visibility_schedule"]
