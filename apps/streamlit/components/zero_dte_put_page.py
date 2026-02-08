from __future__ import annotations

from collections.abc import Sequence
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


DEFAULT_REPORTS_ROOT = Path("data/reports")
_STUDY_DIR = "zero_dte_put_study"

_PROBABILITY_COLUMNS = [
    "symbol",
    "session_date",
    "decision_ts",
    "decision_time_et",
    "decision_mode",
    "risk_tier",
    "strike_return",
    "breach_probability",
    "breach_probability_ci_low",
    "breach_probability_ci_high",
    "sample_size",
    "quote_quality_status",
    "skip_reason",
    "model_version",
    "assumptions_hash",
    "entry_anchor_ts",
    "close_label_ts",
]

_STRIKE_COLUMNS = [
    "symbol",
    "session_date",
    "decision_ts",
    "decision_time_et",
    "decision_mode",
    "risk_tier",
    "ladder_rank",
    "strike_price",
    "strike_return",
    "breach_probability",
    "premium_estimate",
    "fill_model",
    "quote_quality_status",
    "skip_reason",
    "entry_anchor_ts",
]

_WALK_FORWARD_COLUMNS = [
    "risk_tier",
    "decision_mode",
    "exit_mode",
    "trade_count",
    "avg_pnl_per_contract",
    "median_pnl_per_contract",
    "win_rate",
]

_CALIBRATION_COLUMNS = [
    "source",
    "risk_tier",
    "bin_index",
    "sample_size",
    "predicted_mean",
    "observed_rate",
    "abs_gap",
]

_FORWARD_COLUMNS = [
    "symbol",
    "session_date",
    "decision_ts",
    "decision_time_et",
    "risk_tier",
    "ladder_rank",
    "strike_return",
    "strike_price",
    "breach_probability",
    "breach_probability_ci_low",
    "breach_probability_ci_high",
    "premium_estimate",
    "policy_status",
    "policy_reason",
    "reconciliation_status",
    "realized_close_return_from_entry",
    "model_version",
    "assumptions_hash",
]


def normalize_symbol(value: object, *, default: str = "SPY") -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return default.strip().upper()
    return "".join(ch for ch in raw if ch.isalnum() or ch in {".", "-", "_"})


def resolve_reports_root(reports_root: str | Path | None = None) -> Path:
    value = DEFAULT_REPORTS_ROOT if reports_root is None else Path(str(reports_root))
    return value.expanduser().resolve()


def list_zero_dte_symbols(
    *,
    reports_root: str | Path | None = None,
) -> tuple[list[str], str | None]:
    symbols, note = _list_symbols_cached(reports_root=str(resolve_reports_root(reports_root)))
    return symbols, note


@st.cache_data(ttl=60, show_spinner=False)
def _list_symbols_cached(*, reports_root: str) -> tuple[list[str], str | None]:
    study_root = Path(reports_root) / _STUDY_DIR
    if not study_root.exists():
        return [], f"Study reports directory not found: {study_root}"
    if not study_root.is_dir():
        return [], f"Expected a directory at {study_root}"
    symbols = [normalize_symbol(path.name) for path in study_root.iterdir() if path.is_dir()]
    out = sorted(symbol for symbol in symbols if symbol)
    if not out:
        return [], f"No symbol folders found under {study_root}"
    return out, None


def load_probability_surface(
    symbol: str,
    *,
    reports_root: str | Path | None = None,
    decision_mode: str | None = None,
    decision_time_et: str | None = None,
    risk_tier: float | None = None,
    max_strike_distance_pct: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    payload, notes = _load_latest_study_payload(
        symbol=normalize_symbol(symbol),
        reports_root=str(resolve_reports_root(reports_root)),
    )
    if payload is None:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS), notes
    rows = payload.get("probability_rows")
    if not isinstance(rows, list):
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS), [*notes, "No probability rows found in study artifact."]
    frame = _probability_rows_to_frame(rows, symbol=normalize_symbol(symbol))
    frame = _apply_common_filters(
        frame,
        decision_mode=decision_mode,
        decision_time_et=decision_time_et,
        risk_tier=risk_tier,
        max_strike_distance_pct=max_strike_distance_pct,
    )
    return frame.reset_index(drop=True).reindex(columns=_PROBABILITY_COLUMNS), notes


def load_strike_table(
    symbol: str,
    *,
    reports_root: str | Path | None = None,
    decision_mode: str | None = None,
    decision_time_et: str | None = None,
    risk_tier: float | None = None,
    max_strike_distance_pct: float | None = None,
    fill_model: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    payload, notes = _load_latest_study_payload(
        symbol=normalize_symbol(symbol),
        reports_root=str(resolve_reports_root(reports_root)),
    )
    if payload is None:
        return pd.DataFrame(columns=_STRIKE_COLUMNS), notes
    rows = payload.get("strike_ladder_rows")
    if not isinstance(rows, list):
        return pd.DataFrame(columns=_STRIKE_COLUMNS), [*notes, "No strike ladder rows found in study artifact."]
    frame = _strike_rows_to_frame(rows, symbol=normalize_symbol(symbol))
    frame = _apply_common_filters(
        frame,
        decision_mode=decision_mode,
        decision_time_et=decision_time_et,
        risk_tier=risk_tier,
        max_strike_distance_pct=max_strike_distance_pct,
    )
    fill = str(fill_model or "").strip().lower()
    if fill and fill != "all":
        frame = frame.loc[frame["fill_model"].astype(str).str.lower() == fill]
    return frame.reset_index(drop=True).reindex(columns=_STRIKE_COLUMNS), notes


def load_walk_forward_summary(
    symbol: str,
    *,
    reports_root: str | Path | None = None,
    decision_mode: str | None = None,
    risk_tier: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    payload, notes = _load_latest_study_payload(
        symbol=normalize_symbol(symbol),
        reports_root=str(resolve_reports_root(reports_root)),
    )
    if payload is None:
        return pd.DataFrame(columns=_WALK_FORWARD_COLUMNS), notes
    rows = payload.get("simulation_rows")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(columns=_WALK_FORWARD_COLUMNS), [*notes, "No simulation rows found in study artifact."]
    simulation = _simulation_rows_to_frame(rows, symbol=normalize_symbol(symbol))
    mode = str(decision_mode or "").strip().lower()
    if mode and mode != "all":
        simulation = simulation.loc[simulation["decision_mode"].astype(str).str.lower() == mode]
    if risk_tier is not None:
        tier_value = float(risk_tier)
        simulation = simulation.loc[pd.to_numeric(simulation["risk_tier"], errors="coerce").eq(tier_value)]
    if simulation.empty:
        return pd.DataFrame(columns=_WALK_FORWARD_COLUMNS), notes
    grouped = (
        simulation.groupby(["risk_tier", "decision_mode", "exit_mode"], dropna=False, sort=True)
        .agg(
            trade_count=("pnl_per_contract", "size"),
            avg_pnl_per_contract=("pnl_per_contract", "mean"),
            median_pnl_per_contract=("pnl_per_contract", "median"),
            win_rate=("pnl_per_contract", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    return grouped.reindex(columns=_WALK_FORWARD_COLUMNS), notes


def load_calibration_curves(
    symbol: str,
    *,
    reports_root: str | Path | None = None,
    risk_tier: float | None = None,
    bins: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    forward_df, forward_notes = load_forward_snapshots(
        symbol=symbol,
        reports_root=reports_root,
        risk_tier=risk_tier,
    )
    notes = [*forward_notes]
    finalized = forward_df.loc[
        forward_df["reconciliation_status"].astype(str).str.lower().eq("finalized")
    ].copy()
    if finalized.empty:
        notes.append("No finalized forward rows yet; calibration requires reconciled close outcomes.")
        return pd.DataFrame(columns=_CALIBRATION_COLUMNS), _dedupe_notes(notes)

    finalized["breach_observed"] = (
        pd.to_numeric(finalized["realized_close_return_from_entry"], errors="coerce")
        <= pd.to_numeric(finalized["strike_return"], errors="coerce")
    ).astype("float64")
    finalized = finalized.loc[
        pd.to_numeric(finalized["breach_probability"], errors="coerce").between(0.0, 1.0, inclusive="both")
        & finalized["breach_observed"].isin([0.0, 1.0])
    ].copy()
    if finalized.empty:
        notes.append("Forward rows were present but lacked valid probability/outcome pairs for calibration.")
        return pd.DataFrame(columns=_CALIBRATION_COLUMNS), _dedupe_notes(notes)

    bins = max(2, int(bins))
    edges = [idx / float(bins) for idx in range(bins + 1)]
    finalized["bin_index"] = pd.cut(
        pd.to_numeric(finalized["breach_probability"], errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    ).fillna(0)
    finalized["bin_index"] = pd.to_numeric(finalized["bin_index"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    for tier, subset in finalized.groupby("risk_tier", dropna=False, sort=True):
        for bin_index, bin_rows in subset.groupby("bin_index", dropna=False, sort=True):
            predicted = pd.to_numeric(bin_rows["breach_probability"], errors="coerce")
            observed = pd.to_numeric(bin_rows["breach_observed"], errors="coerce")
            if predicted.empty or observed.empty:
                continue
            predicted_mean = float(predicted.mean())
            observed_rate = float(observed.mean())
            rows.append(
                {
                    "source": "forward_test",
                    "risk_tier": float(tier) if pd.notna(tier) else float("nan"),
                    "bin_index": int(bin_index),
                    "sample_size": int(len(bin_rows)),
                    "predicted_mean": predicted_mean,
                    "observed_rate": observed_rate,
                    "abs_gap": abs(predicted_mean - observed_rate),
                }
            )
    if not rows:
        notes.append("No calibration bins could be produced from finalized forward rows.")
        return pd.DataFrame(columns=_CALIBRATION_COLUMNS), _dedupe_notes(notes)

    out = pd.DataFrame(rows).sort_values(by=["risk_tier", "bin_index"], kind="stable")
    return out.reset_index(drop=True).reindex(columns=_CALIBRATION_COLUMNS), _dedupe_notes(notes)


def load_forward_snapshots(
    symbol: str,
    *,
    reports_root: str | Path | None = None,
    decision_mode: str | None = None,
    decision_time_et: str | None = None,
    risk_tier: float | None = None,
    max_strike_distance_pct: float | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    rows, notes = _load_forward_rows(
        symbol=normalize_symbol(symbol),
        reports_root=str(resolve_reports_root(reports_root)),
    )
    if not rows:
        return pd.DataFrame(columns=_FORWARD_COLUMNS), notes

    frame = _forward_rows_to_frame(rows, symbol=normalize_symbol(symbol))
    frame = _apply_common_filters(
        frame,
        decision_mode=decision_mode,
        decision_time_et=decision_time_et,
        risk_tier=risk_tier,
        max_strike_distance_pct=max_strike_distance_pct,
    )
    return frame.reset_index(drop=True).reindex(columns=_FORWARD_COLUMNS), notes


@st.cache_data(ttl=60, show_spinner=False)
def _load_latest_study_payload(*, symbol: str, reports_root: str) -> tuple[dict[str, Any] | None, list[str]]:
    symbol_dir = Path(reports_root) / _STUDY_DIR / normalize_symbol(symbol)
    if not symbol_dir.exists():
        return None, [f"No study artifact directory found for {normalize_symbol(symbol)} at {symbol_dir}."]
    if not symbol_dir.is_dir():
        return None, [f"Expected directory for {normalize_symbol(symbol)} artifacts: {symbol_dir}."]

    artifact_files = sorted(
        [path for path in symbol_dir.glob("*.json") if path.name != "active_model.json"],
        key=lambda path: path.name,
        reverse=True,
    )
    if not artifact_files:
        return None, [f"No study artifact JSON files found under {symbol_dir}."]

    notes: list[str] = []
    for artifact_path in artifact_files:
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            notes.append(f"Could not parse JSON artifact: {artifact_path.name}")
            continue
        if not isinstance(payload, dict):
            notes.append(f"Artifact {artifact_path.name} is not a JSON object.")
            continue
        payload["_artifact_file"] = artifact_path.name
        return payload, notes
    notes.append("No readable study artifact JSON payload found.")
    return None, notes


@st.cache_data(ttl=60, show_spinner=False)
def _load_forward_rows(*, symbol: str, reports_root: str) -> tuple[list[dict[str, Any]], list[str]]:
    symbol_dir = Path(reports_root) / _STUDY_DIR / normalize_symbol(symbol)
    snapshot_path = symbol_dir / "forward_snapshots.jsonl"
    if not snapshot_path.exists():
        return [], [f"Forward snapshot file not found: {snapshot_path}."]
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for line_number, line in enumerate(snapshot_path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            notes.append(f"Skipped malformed forward snapshot row at line {line_number}.")
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    if not rows:
        notes.append("No valid forward snapshot rows found.")
    return rows, notes


def _probability_rows_to_frame(rows: Sequence[dict[str, Any]], *, symbol: str) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        anchor = row.get("anchor") if isinstance(row, dict) else {}
        anchor = anchor if isinstance(anchor, dict) else {}
        decision_ts = _parse_timestamp(anchor.get("decision_ts"))
        normalized.append(
            {
                "symbol": normalize_symbol(row.get("symbol"), default=symbol),
                "session_date": _parse_session_date(anchor.get("session_date")),
                "decision_ts": decision_ts,
                "decision_time_et": _decision_time_label(decision_ts),
                "decision_mode": str(anchor.get("decision_mode") or "").strip().lower() or "unknown",
                "risk_tier": _as_float(row.get("risk_tier")),
                "strike_return": _as_float(row.get("strike_return")),
                "breach_probability": _as_float(row.get("breach_probability")),
                "breach_probability_ci_low": _as_float(row.get("breach_probability_ci_low")),
                "breach_probability_ci_high": _as_float(row.get("breach_probability_ci_high")),
                "sample_size": _as_int(row.get("sample_size")),
                "quote_quality_status": str(row.get("quote_quality_status") or "unknown"),
                "skip_reason": _as_clean_text(row.get("skip_reason")),
                "model_version": str(row.get("model_version") or ""),
                "assumptions_hash": str(row.get("assumptions_hash") or ""),
                "entry_anchor_ts": _parse_timestamp(anchor.get("entry_anchor_ts")),
                "close_label_ts": _parse_timestamp(anchor.get("close_label_ts")),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=_PROBABILITY_COLUMNS)
    out = pd.DataFrame(normalized)
    out = out.dropna(subset=["decision_ts", "risk_tier", "strike_return", "breach_probability"])
    return out.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "strike_return"],
        ascending=[True, True, True, True],
        kind="stable",
    )


def _strike_rows_to_frame(rows: Sequence[dict[str, Any]], *, symbol: str) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        anchor = row.get("anchor") if isinstance(row, dict) else {}
        anchor = anchor if isinstance(anchor, dict) else {}
        decision_ts = _parse_timestamp(anchor.get("decision_ts"))
        normalized.append(
            {
                "symbol": normalize_symbol(row.get("symbol"), default=symbol),
                "session_date": _parse_session_date(anchor.get("session_date")),
                "decision_ts": decision_ts,
                "decision_time_et": _decision_time_label(decision_ts),
                "decision_mode": str(anchor.get("decision_mode") or "").strip().lower() or "unknown",
                "risk_tier": _as_float(row.get("risk_tier")),
                "ladder_rank": _as_int(row.get("ladder_rank")),
                "strike_price": _as_float(row.get("strike_price")),
                "strike_return": _as_float(row.get("strike_return")),
                "breach_probability": _as_float(row.get("breach_probability")),
                "premium_estimate": _as_float(row.get("premium_estimate")),
                "fill_model": str(row.get("fill_model") or "unknown").strip().lower(),
                "quote_quality_status": str(row.get("quote_quality_status") or "unknown"),
                "skip_reason": _as_clean_text(row.get("skip_reason")),
                "entry_anchor_ts": _parse_timestamp(anchor.get("entry_anchor_ts")),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=_STRIKE_COLUMNS)
    out = pd.DataFrame(normalized)
    out = out.dropna(subset=["decision_ts", "risk_tier", "ladder_rank", "strike_return"])
    return out.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "ladder_rank"],
        ascending=[True, True, True, True],
        kind="stable",
    )


def _simulation_rows_to_frame(rows: Sequence[dict[str, Any]], *, symbol: str) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        anchor = row.get("anchor") if isinstance(row, dict) else {}
        anchor = anchor if isinstance(anchor, dict) else {}
        normalized.append(
            {
                "symbol": normalize_symbol(row.get("symbol"), default=symbol),
                "session_date": _parse_session_date(anchor.get("session_date")),
                "decision_ts": _parse_timestamp(anchor.get("decision_ts")),
                "decision_mode": str(anchor.get("decision_mode") or "").strip().lower() or "unknown",
                "risk_tier": _as_float(row.get("risk_tier")),
                "exit_mode": str(row.get("exit_mode") or "").strip().lower(),
                "pnl_per_contract": _as_float(row.get("pnl_per_contract")),
            }
        )
    if not normalized:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session_date",
                "decision_ts",
                "decision_mode",
                "risk_tier",
                "exit_mode",
                "pnl_per_contract",
            ]
        )
    out = pd.DataFrame(normalized)
    return out.dropna(subset=["decision_ts", "risk_tier", "exit_mode"])


def _forward_rows_to_frame(rows: Sequence[dict[str, Any]], *, symbol: str) -> pd.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        decision_ts = _parse_timestamp(row.get("decision_ts"))
        normalized.append(
            {
                "symbol": normalize_symbol(row.get("symbol"), default=symbol),
                "session_date": _parse_session_date(row.get("session_date")),
                "decision_ts": decision_ts,
                "decision_time_et": _decision_time_label(decision_ts),
                "decision_mode": str(row.get("decision_mode") or "unknown").strip().lower(),
                "risk_tier": _as_float(row.get("risk_tier")),
                "ladder_rank": _as_int(row.get("ladder_rank")),
                "strike_return": _as_float(row.get("strike_return")),
                "strike_price": _as_float(row.get("strike_price")),
                "breach_probability": _as_float(row.get("breach_probability")),
                "breach_probability_ci_low": _as_float(row.get("breach_probability_ci_low")),
                "breach_probability_ci_high": _as_float(row.get("breach_probability_ci_high")),
                "premium_estimate": _as_float(row.get("premium_estimate")),
                "policy_status": _as_clean_text(row.get("policy_status")) or "unknown",
                "policy_reason": _as_clean_text(row.get("policy_reason")),
                "reconciliation_status": _as_clean_text(row.get("reconciliation_status")) or "pending_close",
                "realized_close_return_from_entry": _as_float(row.get("realized_close_return_from_entry")),
                "model_version": str(row.get("model_version") or ""),
                "assumptions_hash": str(row.get("assumptions_hash") or ""),
            }
        )
    if not normalized:
        return pd.DataFrame(columns=_FORWARD_COLUMNS)
    out = pd.DataFrame(normalized)
    out = out.dropna(subset=["decision_ts", "risk_tier", "strike_return", "breach_probability"])
    return out.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "ladder_rank"],
        ascending=[True, True, True, True],
        kind="stable",
    )


def _apply_common_filters(
    frame: pd.DataFrame,
    *,
    decision_mode: str | None,
    decision_time_et: str | None,
    risk_tier: float | None,
    max_strike_distance_pct: float | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    mode = str(decision_mode or "").strip().lower()
    if mode and mode != "all":
        out = out.loc[out["decision_mode"].astype(str).str.lower() == mode]
    label = str(decision_time_et or "").strip()
    if label and label.lower() != "all":
        out = out.loc[out["decision_time_et"].astype(str) == label]
    if risk_tier is not None:
        tier_value = float(risk_tier)
        out = out.loc[pd.to_numeric(out["risk_tier"], errors="coerce").eq(tier_value)]
    if max_strike_distance_pct is not None and "strike_return" in out.columns:
        max_distance = abs(float(max_strike_distance_pct)) / 100.0
        strike_return = pd.to_numeric(out["strike_return"], errors="coerce").abs()
        out = out.loc[strike_return <= max_distance]
    return out


def _decision_time_label(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    try:
        return value.tz_convert("America/New_York").strftime("%H:%M")
    except Exception:  # noqa: BLE001
        return "-"


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _parse_session_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _as_int(value: object) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _as_clean_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _dedupe_notes(notes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for note in notes:
        text = str(note or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
