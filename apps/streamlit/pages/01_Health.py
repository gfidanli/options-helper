from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

from apps.streamlit.components.gap_planner import build_gap_backfill_plan
from apps.streamlit.components.health_page import HealthSnapshot, load_health_snapshot

st.title("Health")
st.caption("Informational and educational use only. Not financial advice.")
st.info("Read-only ingestion visibility view. This page does not trigger writes.")

with st.sidebar:
    st.markdown("### Data Sources")
    database_path = st.text_input(
        "DuckDB path",
        value="",
        help="Optional. Leave blank to use OPTIONS_HELPER_DUCKDB_PATH or data/warehouse/options.duckdb.",
    )
    lookback_days = st.slider("Lookback days", min_value=1, max_value=120, value=30, step=1)
    stale_days = st.slider("Stale watermark threshold (days)", min_value=0, max_value=45, value=3, step=1)
    row_limit = st.slider("Rows per section", min_value=10, max_value=500, value=200, step=10)

database_arg = database_path or None
snapshot = load_health_snapshot(
    database_path=database_arg,
    days=lookback_days,
    stale_days=stale_days,
    limit=row_limit,
)

if snapshot.guidance:
    st.warning(snapshot.guidance)
    st.markdown("**Bootstrap commands**")
    st.code(
        "\n".join(
            [
                "./.venv/bin/options-helper db init",
                "./.venv/bin/options-helper ingest candles --watchlist positions --watchlist monitor",
                "./.venv/bin/options-helper snapshot-options portfolio.json --all-expiries --full-chain",
                "./.venv/bin/options-helper flow portfolio.json --window 1 --group-by expiry-strike",
                "./.venv/bin/options-helper derived update --symbol <SYMBOL> --as-of latest",
            ]
        ),
        language="bash",
    )


def _render_overview_metrics(*, snapshot: HealthSnapshot, stale_days: int) -> None:
    latest_failed = 0
    if not snapshot.latest_runs.empty and "status" in snapshot.latest_runs.columns:
        latest_failed = int((snapshot.latest_runs["status"].astype("string").str.lower() == "failed").sum())

    stale_count = 0
    if not snapshot.watermarks.empty and "freshness" in snapshot.watermarks.columns:
        stale_count = int((snapshot.watermarks["freshness"].astype("string") == "stale").sum())

    metrics = st.columns(4)
    metrics[0].metric("Tracked Jobs", value=f"{len(snapshot.latest_runs)}")
    metrics[1].metric("Latest Job Failures", value=f"{latest_failed}")
    metrics[2].metric(f"Stale Watermarks (>= {stale_days}d)", value=f"{stale_count}")
    metrics[3].metric("Recent Failed Checks", value=f"{len(snapshot.failed_checks)}")


def _render_latest_runs(*, snapshot: HealthSnapshot) -> None:
    st.subheader("Latest Runs")
    if snapshot.latest_runs.empty:
        st.info("No run ledger rows found for the selected lookback.")
        return
    display = snapshot.latest_runs.copy()
    display = _format_datetime_columns(display, columns=["started_at", "ended_at"])
    if "duration_ms" in display.columns:
        display["duration_ms"] = pd.to_numeric(display["duration_ms"], errors="coerce")
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_recurring_failures(*, snapshot: HealthSnapshot) -> None:
    st.subheader("Recurring Failures (Stack Hash)")
    if snapshot.recurring_failures.empty:
        st.info("No recurring failure groups detected in the selected window.")
        return
    display = snapshot.recurring_failures.copy()
    display = _format_datetime_columns(display, columns=["latest_started_at", "first_started_at"])
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_watermarks(*, snapshot: HealthSnapshot) -> None:
    st.subheader("Watermark Freshness")
    if snapshot.watermarks.empty:
        st.info("No asset watermark rows found.")
        return
    display = snapshot.watermarks.copy()
    display = _format_datetime_columns(display, columns=["watermark_ts", "updated_at"])
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_failed_checks(*, snapshot: HealthSnapshot) -> None:
    st.subheader("Recent Failed Checks")
    if snapshot.failed_checks.empty:
        st.info("No failed checks found for the selected lookback.")
        return
    display = snapshot.failed_checks.copy()
    display = _format_datetime_columns(display, columns=["checked_at"])
    st.dataframe(display, hide_index=True, use_container_width=True)


def _render_gap_backfill_plan(*, snapshot: HealthSnapshot, stale_days: int) -> None:
    st.subheader("Gap Backfill Planner (Display Only)")
    plan_df = build_gap_backfill_plan(
        watermarks_df=snapshot.watermarks,
        latest_runs_df=snapshot.latest_runs,
        failed_checks_df=snapshot.failed_checks,
        stale_days=stale_days,
    )
    if plan_df.empty:
        st.success("No immediate backfill actions suggested from current health signals.")
        return

    st.caption("Copy/paste commands manually. This page does not execute commands.")
    for _, row in plan_df.iterrows():
        step = int(row.get("step") or 0)
        asset_key = str(row.get("asset_key") or "")
        scope_key = str(row.get("scope_key") or "ALL")
        reason = str(row.get("reason") or "")
        command = str(row.get("command") or "")
        st.markdown(f"**Step {step}: `{asset_key}` (`{scope_key}`)**")
        if reason:
            st.caption(reason)
        st.code(command, language="bash")

    with st.expander("Planner Table"):
        st.dataframe(plan_df, hide_index=True, use_container_width=True)


def _format_datetime_columns(df: pd.DataFrame, *, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        parsed = pd.to_datetime(out[column], errors="coerce")
        out[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
        out[column] = out[column].fillna("-")
    return out


_render_overview_metrics(snapshot=snapshot, stale_days=stale_days)
_render_latest_runs(snapshot=snapshot)
_render_recurring_failures(snapshot=snapshot)
_render_watermarks(snapshot=snapshot)
_render_failed_checks(snapshot=snapshot)
_render_gap_backfill_plan(snapshot=snapshot, stale_days=stale_days)
