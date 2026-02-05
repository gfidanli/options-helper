from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping, Sequence
import re
from typing import Any

import pandas as pd

try:
    import streamlit as st
except Exception:  # pragma: no cover - exercised when optional UI deps are absent.
    st = None


PLAN_COLUMNS = ["step", "priority", "asset_key", "scope_key", "reason", "command"]

_REQUIRED_WATERMARK_ASSETS = ("candles_daily", "options_snapshots", "options_flow", "derived_daily")
_ASSET_PRIORITY = {
    "candles_daily": 10,
    "option_bars": 20,
    "options_snapshots": 30,
    "options_flow": 40,
    "derived_daily": 50,
}
_DEPENDENCIES = {
    "options_snapshots": ("candles_daily",),
    "options_flow": ("candles_daily", "options_snapshots"),
    "derived_daily": ("candles_daily", "options_snapshots"),
}
_JOB_TO_ASSETS = {
    "ingest_candles": ("candles_daily",),
    "ingest_options_bars": ("option_bars",),
    "snapshot_options": ("options_snapshots",),
    "compute_flow": ("options_flow",),
    "compute_derived": ("derived_daily",),
}
_ASSET_ALIASES = {
    "option_contracts": "option_bars",
    "option_bars": "option_bars",
    "candles_daily": "candles_daily",
    "options_snapshots": "options_snapshots",
    "options_flow": "options_flow",
    "derived_daily": "derived_daily",
}
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9._-]{0,15}$")


def _query_params_store(
    query_params: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    if query_params is not None:
        return query_params
    if st is None:
        raise RuntimeError("streamlit is required when query_params is not provided")
    return st.query_params


def read_query_param(
    name: str,
    default: str | None = None,
    *,
    query_params: MutableMapping[str, Any] | None = None,
) -> str | None:
    params = _query_params_store(query_params)
    raw_value = params.get(name, default)
    if isinstance(raw_value, list):
        raw_value = raw_value[0] if raw_value else default
    if raw_value in (None, ""):
        return default
    return str(raw_value)


def sync_query_param(
    name: str,
    value: str | None,
    *,
    default: str | None = None,
    query_params: MutableMapping[str, Any] | None = None,
) -> str | None:
    params = _query_params_store(query_params)
    normalized = value if value not in (None, "") else default
    if normalized in (None, ""):
        params.pop(name, None)
        return None
    current = read_query_param(name=name, default=default, query_params=params)
    if current != normalized:
        params[name] = normalized
    return normalized


def sync_csv_query_param(
    name: str,
    values: Sequence[str],
    *,
    query_params: MutableMapping[str, Any] | None = None,
) -> list[str]:
    cleaned_values = [value.strip() for value in values if value and value.strip()]
    serialized = ",".join(cleaned_values) if cleaned_values else None
    sync_query_param(name=name, value=serialized, query_params=query_params)
    return cleaned_values


def build_gap_backfill_plan(
    *,
    watermarks_df: pd.DataFrame,
    latest_runs_df: pd.DataFrame,
    failed_checks_df: pd.DataFrame,
    stale_days: int = 3,
) -> pd.DataFrame:
    issues = _collect_issues(
        watermarks_df=watermarks_df,
        latest_runs_df=latest_runs_df,
        failed_checks_df=failed_checks_df,
        stale_days=max(0, int(stale_days)),
    )
    if not issues:
        return pd.DataFrame(columns=PLAN_COLUMNS)

    records: list[dict[str, Any]] = []
    for asset_key, scope_key, reasons in _sorted_issues(issues):
        records.append(
            {
                "priority": _ASSET_PRIORITY.get(asset_key, 999),
                "asset_key": asset_key,
                "scope_key": scope_key,
                "reason": _render_reason(reasons, stale_days=max(0, int(stale_days))),
                "command": _command_for(asset_key=asset_key, scope_key=scope_key),
            }
        )

    out = pd.DataFrame(records)
    out = out.sort_values(
        by=["priority", "asset_key", "scope_key", "command"],
        ascending=[True, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    out.insert(0, "step", [idx + 1 for idx in range(len(out))])
    return out.reindex(columns=PLAN_COLUMNS)


def _collect_issues(
    *,
    watermarks_df: pd.DataFrame,
    latest_runs_df: pd.DataFrame,
    failed_checks_df: pd.DataFrame,
    stale_days: int,
) -> dict[tuple[str, str], set[str]]:
    issues: dict[tuple[str, str], set[str]] = defaultdict(set)

    normalized_watermarks = watermarks_df if isinstance(watermarks_df, pd.DataFrame) else pd.DataFrame()
    normalized_runs = latest_runs_df if isinstance(latest_runs_df, pd.DataFrame) else pd.DataFrame()
    normalized_checks = failed_checks_df if isinstance(failed_checks_df, pd.DataFrame) else pd.DataFrame()

    present_assets: set[str] = set()
    if not normalized_watermarks.empty and {"asset_key"}.issubset(set(normalized_watermarks.columns)):
        for raw_asset in normalized_watermarks["asset_key"].tolist():
            canonical_asset = _canonical_asset(raw_asset)
            if canonical_asset:
                present_assets.add(canonical_asset)

    for asset_key in _REQUIRED_WATERMARK_ASSETS:
        if asset_key not in present_assets:
            issues[(asset_key, "ALL")].add("missing_watermark")

    if not normalized_watermarks.empty and {"asset_key", "scope_key"}.issubset(set(normalized_watermarks.columns)):
        for row in normalized_watermarks.to_dict(orient="records"):
            asset_key = _canonical_asset(row.get("asset_key"))
            if not asset_key:
                continue
            scope_key = _normalize_scope(row.get("scope_key"))
            staleness_days = _coerce_int(row.get("staleness_days"))
            if staleness_days is not None and staleness_days >= stale_days:
                issues[(asset_key, scope_key)].add(f"stale_watermark:{staleness_days}")

    if not normalized_checks.empty and "asset_key" in normalized_checks.columns:
        for row in normalized_checks.to_dict(orient="records"):
            asset_key = _canonical_asset(row.get("asset_key"))
            if not asset_key:
                continue
            scope_key = _normalize_scope(row.get("partition_key"))
            check_name = str(row.get("check_name") or "").strip()
            if check_name:
                issues[(asset_key, scope_key)].add(f"failed_check:{check_name}")
            else:
                issues[(asset_key, scope_key)].add("failed_check")

    if not normalized_runs.empty and {"job_name", "status"}.issubset(set(normalized_runs.columns)):
        for row in normalized_runs.to_dict(orient="records"):
            status = str(row.get("status") or "").strip().lower()
            if status != "failed":
                continue
            job_name = str(row.get("job_name") or "").strip()
            mapped_assets = _JOB_TO_ASSETS.get(job_name, ())
            for asset_key in mapped_assets:
                issues[(asset_key, "ALL")].add(f"latest_run_failed:{job_name}")

    issues = _expand_dependencies(issues)
    return _collapse_all_scope(issues)


def _expand_dependencies(
    issues: dict[tuple[str, str], set[str]],
) -> dict[tuple[str, str], set[str]]:
    expanded: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (asset_key, scope_key), reasons in issues.items():
        expanded[(asset_key, scope_key)].update(reasons)
        for dependency_asset in _DEPENDENCIES.get(asset_key, ()):
            expanded[(dependency_asset, scope_key)].add(f"dependency:{asset_key}")
    return expanded


def _collapse_all_scope(
    issues: dict[tuple[str, str], set[str]],
) -> dict[tuple[str, str], set[str]]:
    by_asset: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for (asset_key, scope_key), reasons in issues.items():
        if asset_key not in _ASSET_PRIORITY:
            continue
        by_asset[asset_key][scope_key].update(reasons)

    collapsed: dict[tuple[str, str], set[str]] = {}
    for asset_key, scope_map in by_asset.items():
        if "ALL" in scope_map:
            merged: set[str] = set()
            for reason_set in scope_map.values():
                merged.update(reason_set)
            collapsed[(asset_key, "ALL")] = merged
            continue
        for scope_key, reasons in scope_map.items():
            collapsed[(asset_key, scope_key)] = set(reasons)
    return collapsed


def _sorted_issues(
    issues: dict[tuple[str, str], set[str]],
) -> list[tuple[str, str, set[str]]]:
    items = [(asset_key, scope_key, reasons) for (asset_key, scope_key), reasons in issues.items()]
    return sorted(
        items,
        key=lambda item: (
            _ASSET_PRIORITY.get(item[0], 999),
            item[0],
            0 if item[1] == "ALL" else 1,
            item[1],
        ),
    )


def _render_reason(reasons: set[str], *, stale_days: int) -> str:
    reason_tokens = sorted(reasons)
    pieces: list[str] = []

    stale_values = [
        _coerce_int(token.split(":", 1)[1]) for token in reason_tokens if token.startswith("stale_watermark:")
    ]
    stale_values = sorted(value for value in stale_values if value is not None)
    if stale_values:
        pieces.append(f"watermark stale ({max(stale_values)}d >= {stale_days}d)")

    if "missing_watermark" in reason_tokens:
        pieces.append("watermark missing")

    failed_checks = sorted(
        {
            token.split(":", 1)[1]
            for token in reason_tokens
            if token.startswith("failed_check:") and ":" in token
        }
    )
    if failed_checks:
        joined = ", ".join(failed_checks[:3])
        if len(failed_checks) > 3:
            joined = f"{joined}, +{len(failed_checks) - 3} more"
        pieces.append(f"failed checks: {joined}")
    elif "failed_check" in reason_tokens:
        pieces.append("failed checks present")

    failed_jobs = sorted(
        {
            token.split(":", 1)[1]
            for token in reason_tokens
            if token.startswith("latest_run_failed:") and ":" in token
        }
    )
    if failed_jobs:
        pieces.append(f"latest failed run: {', '.join(failed_jobs)}")

    dependencies = sorted(
        {
            token.split(":", 1)[1]
            for token in reason_tokens
            if token.startswith("dependency:") and ":" in token
        }
    )
    if dependencies:
        pieces.append(f"upstream for: {', '.join(dependencies)}")

    if not pieces:
        return "manual review"
    return "; ".join(pieces)


def _command_for(*, asset_key: str, scope_key: str) -> str:
    if asset_key == "candles_daily":
        if scope_key != "ALL":
            return f"./.venv/bin/options-helper ingest candles --symbol {scope_key}"
        return "./.venv/bin/options-helper ingest candles --watchlist positions --watchlist monitor"

    if asset_key == "option_bars":
        if scope_key != "ALL":
            return f"./.venv/bin/options-helper ingest options-bars --symbol {scope_key} --lookback-years 2 --resume"
        return (
            "./.venv/bin/options-helper ingest options-bars --watchlist positions "
            "--watchlist monitor --lookback-years 2 --resume"
        )

    if asset_key == "options_snapshots":
        return "./.venv/bin/options-helper snapshot-options portfolio.json --all-expiries --full-chain"

    if asset_key == "options_flow":
        if scope_key != "ALL":
            return (
                f"./.venv/bin/options-helper flow portfolio.json --symbol {scope_key} "
                "--window 1 --group-by expiry-strike"
            )
        return "./.venv/bin/options-helper flow portfolio.json --window 1 --group-by expiry-strike"

    if asset_key == "derived_daily":
        if scope_key != "ALL":
            return f"./.venv/bin/options-helper derived update --symbol {scope_key} --as-of latest"
        return "./.venv/bin/options-helper derived update --symbol <SYMBOL> --as-of latest"

    return "./.venv/bin/options-helper --help"


def _canonical_asset(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    return _ASSET_ALIASES.get(raw)


def _normalize_scope(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw or raw == "ALL":
        return "ALL"
    candidate = raw.split("|", 1)[0].strip()
    candidate = "".join(ch for ch in candidate if ch.isalnum() or ch in {".", "-", "_"})
    if not candidate or not _SYMBOL_RE.match(candidate):
        return "ALL"
    return candidate


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
