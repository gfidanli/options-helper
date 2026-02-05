from __future__ import annotations

import pandas as pd

from apps.streamlit.components.gap_planner import build_gap_backfill_plan


def test_gap_planner_orders_dependencies_and_commands() -> None:
    watermarks_df = pd.DataFrame(
        [
            {"asset_key": "candles_daily", "scope_key": "AAPL", "staleness_days": 1},
            {"asset_key": "options_snapshots", "scope_key": "AAPL", "staleness_days": 1},
            {"asset_key": "options_flow", "scope_key": "AAPL", "staleness_days": 9},
            {"asset_key": "derived_daily", "scope_key": "AAPL", "staleness_days": 7},
        ]
    )
    latest_runs_df = pd.DataFrame(
        [
            {"job_name": "ingest_candles", "status": "success"},
            {"job_name": "compute_flow", "status": "success"},
        ]
    )
    failed_checks_df = pd.DataFrame(
        [
            {
                "asset_key": "options_flow",
                "partition_key": "AAPL|2026-02-05",
                "check_name": "flow_no_null_primary_keys",
            }
        ]
    )

    plan_df = build_gap_backfill_plan(
        watermarks_df=watermarks_df,
        latest_runs_df=latest_runs_df,
        failed_checks_df=failed_checks_df,
        stale_days=3,
    )

    assert list(plan_df["step"]) == [1, 2, 3, 4]
    assert list(plan_df["asset_key"]) == ["candles_daily", "options_snapshots", "options_flow", "derived_daily"]
    assert list(plan_df["scope_key"]) == ["AAPL", "AAPL", "AAPL", "AAPL"]

    command_by_asset = dict(zip(plan_df["asset_key"], plan_df["command"], strict=True))
    assert command_by_asset["candles_daily"] == "./.venv/bin/options-helper ingest candles --symbol AAPL"
    assert (
        command_by_asset["options_snapshots"]
        == "./.venv/bin/options-helper snapshot-options portfolio.json --all-expiries --full-chain"
    )
    assert (
        command_by_asset["options_flow"]
        == "./.venv/bin/options-helper flow portfolio.json --symbol AAPL --window 1 --group-by expiry-strike"
    )
    assert command_by_asset["derived_daily"] == "./.venv/bin/options-helper derived update --symbol AAPL --as-of latest"


def test_gap_planner_collapses_to_all_scope_for_failed_latest_run() -> None:
    watermarks_df = pd.DataFrame(
        [
            {"asset_key": "candles_daily", "scope_key": "ALL", "staleness_days": 0},
            {"asset_key": "options_snapshots", "scope_key": "ALL", "staleness_days": 0},
            {"asset_key": "options_flow", "scope_key": "ALL", "staleness_days": 0},
            {"asset_key": "derived_daily", "scope_key": "AAPL", "staleness_days": 0},
        ]
    )
    latest_runs_df = pd.DataFrame([{"job_name": "compute_derived", "status": "failed"}])
    failed_checks_df = pd.DataFrame()

    plan_df = build_gap_backfill_plan(
        watermarks_df=watermarks_df,
        latest_runs_df=latest_runs_df,
        failed_checks_df=failed_checks_df,
        stale_days=3,
    )

    derived_row = plan_df[plan_df["asset_key"] == "derived_daily"].iloc[0].to_dict()
    assert derived_row["scope_key"] == "ALL"
    assert derived_row["command"] == "./.venv/bin/options-helper derived update --symbol <SYMBOL> --as-of latest"
    assert "latest failed run: compute_derived" in str(derived_row["reason"])

    candles_row = plan_df[plan_df["asset_key"] == "candles_daily"].iloc[0].to_dict()
    assert candles_row["scope_key"] == "ALL"
    assert "upstream for: derived_daily" in str(candles_row["reason"])
