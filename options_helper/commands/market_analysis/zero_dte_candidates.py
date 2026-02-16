from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from options_helper.analysis.zero_dte_features import compute_zero_dte_features
from options_helper.analysis.zero_dte_labels import build_zero_dte_labels
from options_helper.commands.common import _parse_date
from options_helper.commands.market_analysis.core_helpers import _as_float
from options_helper.commands.market_analysis.zero_dte_utils import _first_text
from options_helper.data.zero_dte_dataset import ZeroDTEIntradayDatasetLoader
from options_helper.schemas.zero_dte_put_study import DecisionMode, FillModel, QuoteQualityStatus


def _build_zero_dte_candidates(
    *,
    loader: ZeroDTEIntradayDatasetLoader,
    candles: pd.DataFrame,
    symbol: str,
    sessions: list[date],
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
    strike_grid: tuple[float, ...],
    risk_tiers: tuple[float, ...],
    fill_model: FillModel,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    candidate_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    snapshot_parts: list[pd.DataFrame] = []
    warnings: list[str] = []
    for session_date in sessions:
        previous_close = _resolve_previous_close(candles, session_date)
        if previous_close is None:
            warnings.append(f"Missing previous close for {session_date.isoformat()}.")
            continue
        day_times = _resolve_decision_times_for_session(
            session_date=session_date,
            decision_mode=decision_mode,
            decision_times=decision_times,
            rolling_interval_minutes=rolling_interval_minutes,
        )
        dataset = loader.load_day(
            session_date,
            underlying_symbol=symbol,
            decision_times=day_times,
        )
        if dataset.state_rows.empty or dataset.underlying_bars.empty:
            continue
        features = compute_zero_dte_features(
            dataset.state_rows,
            dataset.underlying_bars,
            previous_close=previous_close,
        )
        labels = build_zero_dte_labels(
            dataset.state_rows,
            dataset.underlying_bars,
            market_close_ts=pd.Timestamp(dataset.session.market_close).tz_convert("UTC"),
        )
        snapshots = _build_strike_snapshot_rows(
            loader=loader,
            session_date=session_date,
            labels=labels,
            previous_close=previous_close,
            strike_grid=strike_grid,
            symbol=symbol,
        )
        combined = _assemble_zero_dte_candidate_rows(
            features=features,
            labels=labels,
            snapshots=snapshots,
            fill_model=fill_model,
            decision_mode=decision_mode,
            risk_tiers=risk_tiers,
        )
        if not combined.empty:
            candidate_parts.append(combined)
        feature_parts.append(features)
        label_parts.append(labels)
        snapshot_parts.append(snapshots)
    candidates = pd.concat(candidate_parts, ignore_index=True) if candidate_parts else pd.DataFrame()
    features_all = pd.concat(feature_parts, ignore_index=True) if feature_parts else pd.DataFrame()
    labels_all = pd.concat(label_parts, ignore_index=True) if label_parts else pd.DataFrame()
    snapshots_all = pd.concat(snapshot_parts, ignore_index=True) if snapshot_parts else pd.DataFrame()
    return candidates, features_all, labels_all, snapshots_all, warnings


def _resolve_zero_dte_study_range(
    *,
    start_date: str | None,
    end_date: str | None,
    intraday_dir: Path,
    symbol: str,
) -> tuple[date, date]:
    end = _parse_date(end_date) if end_date else _resolve_latest_intraday_session(intraday_dir, symbol)
    start = _parse_date(start_date) if start_date else (end - timedelta(days=60))
    if start > end:
        raise ValueError("--start-date must be on or before --end-date")
    return start, end


def _resolve_latest_intraday_session(intraday_dir: Path, symbol: str) -> date:
    root = intraday_dir / "stocks" / "bars" / "1Min" / symbol
    if not root.exists():
        raise ValueError(f"No intraday partitions found for {symbol} at {root}")
    days: list[date] = []
    for file_path in root.glob("*.parquet"):
        try:
            days.append(date.fromisoformat(file_path.stem))
        except ValueError:
            continue
    if not days:
        raise ValueError(f"No intraday session files found for {symbol} at {root}")
    return max(days)


def _resolve_decision_times_for_session(
    *,
    session_date: date,
    decision_mode: DecisionMode,
    decision_times: tuple[str, ...],
    rolling_interval_minutes: int,
) -> tuple[str, ...]:
    if decision_mode == DecisionMode.FIXED_TIME:
        return decision_times
    out: list[str] = []
    cursor = datetime.combine(session_date, time(9, 30))
    end = datetime.combine(session_date, time(16, 0))
    step = timedelta(minutes=rolling_interval_minutes)
    while cursor <= end:
        out.append(cursor.strftime("%H:%M"))
        cursor += step
    return tuple(out)


def _resolve_previous_close(candles: pd.DataFrame, session_date: date) -> float | None:
    if candles.empty:
        return None
    prior = candles.loc[candles["date"] < pd.Timestamp(session_date)].copy()
    if prior.empty:
        return None
    close = pd.to_numeric(prior["close"], errors="coerce").dropna()
    if close.empty:
        return None
    value = float(close.iloc[-1])
    return value if value > 0 else None


def _build_strike_snapshot_rows(
    *,
    loader: ZeroDTEIntradayDatasetLoader,
    session_date: date,
    labels: pd.DataFrame,
    previous_close: float,
    strike_grid: tuple[float, ...],
    symbol: str,
) -> pd.DataFrame:
    if labels is None or labels.empty:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for _, label_row in labels.iterrows():
        entry_anchor_ts = pd.to_datetime(label_row.get("entry_anchor_ts"), errors="coerce", utc=True)
        decision_ts = pd.to_datetime(label_row.get("decision_ts"), errors="coerce", utc=True)
        if pd.isna(entry_anchor_ts) or pd.isna(decision_ts):
            continue
        snap = loader.load_strike_premium_snapshot(
            session_date,
            previous_close=previous_close,
            strike_returns=strike_grid,
            entry_anchor_ts=entry_anchor_ts,
            underlying_symbol=symbol,
        )
        if snap.empty:
            continue
        snap = snap.copy()
        snap["decision_ts"] = decision_ts
        snap["decision_bar_completed_ts"] = pd.to_datetime(
            label_row.get("decision_bar_completed_ts"),
            errors="coerce",
            utc=True,
        )
        snap["close_label_ts"] = pd.to_datetime(label_row.get("close_label_ts"), errors="coerce", utc=True)
        snap["close_price"] = pd.to_numeric(label_row.get("close_price"), errors="coerce")
        snap["close_return_from_entry"] = pd.to_numeric(
            label_row.get("close_return_from_entry"),
            errors="coerce",
        )
        snap["entry_anchor_price"] = pd.to_numeric(label_row.get("entry_anchor_price"), errors="coerce")
        snap["label_status"] = label_row.get("label_status")
        snap["label_skip_reason"] = label_row.get("skip_reason")
        parts.append(snap)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _assemble_zero_dte_candidate_rows(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: pd.DataFrame,
    fill_model: FillModel,
    decision_mode: DecisionMode,
    risk_tiers: tuple[float, ...],
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty:
        return pd.DataFrame()
    merged = _merge_candidate_inputs(features=features, labels=labels, snapshots=snapshots)
    merged = _apply_candidate_policy_fields(merged, fill_model=fill_model, decision_mode=decision_mode)
    out = _finalize_candidate_rows(merged)
    return _expand_risk_tiers(out, risk_tiers=risk_tiers)


def _merge_candidate_inputs(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    feature_base = features.copy()
    feature_base["session_date"] = pd.to_datetime(feature_base["session_date"], errors="coerce").dt.date
    feature_base["decision_ts"] = pd.to_datetime(feature_base["decision_ts"], errors="coerce", utc=True)

    label_base = labels.copy()
    label_base["session_date"] = pd.to_datetime(label_base["session_date"], errors="coerce").dt.date
    label_base["decision_ts"] = pd.to_datetime(label_base["decision_ts"], errors="coerce", utc=True)
    base = feature_base.merge(
        label_base,
        on=["session_date", "decision_ts"],
        how="outer",
        suffixes=("_feature", "_label"),
    )

    snap = snapshots.copy()
    snap["session_date"] = pd.to_datetime(snap["session_date"], errors="coerce").dt.date
    snap["decision_ts"] = pd.to_datetime(snap["decision_ts"], errors="coerce", utc=True)
    snap["entry_anchor_ts"] = pd.to_datetime(snap["entry_anchor_ts"], errors="coerce", utc=True)
    merged = snap.merge(base, on=["session_date", "decision_ts"], how="left", sort=True)
    return _coalesce_candidate_columns(merged)


def _coalesce_candidate_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for column in (
        "entry_anchor_ts",
        "decision_bar_completed_ts",
        "close_label_ts",
        "close_price",
        "entry_anchor_price",
        "close_return_from_entry",
        "label_status",
        "skip_reason",
    ):
        if column in out.columns:
            continue
        left = out.get(f"{column}_x")
        right = out.get(f"{column}_y")
        if left is None and right is None:
            continue
        if left is None:
            out[column] = right
            continue
        if right is None:
            out[column] = left
            continue
        out[column] = left
        missing_mask = out[column].isna()
        if bool(missing_mask.any()):
            out.loc[missing_mask, column] = right.loc[missing_mask]
    return out


def _apply_candidate_policy_fields(
    merged: pd.DataFrame,
    *,
    fill_model: FillModel,
    decision_mode: DecisionMode,
) -> pd.DataFrame:
    out = merged.copy()
    out["strike_return"] = pd.to_numeric(out["target_strike_return"], errors="coerce")
    out["target_strike_price"] = pd.to_numeric(out["target_strike_price"], errors="coerce")
    out["strike_price"] = pd.to_numeric(out.get("strike_price"), errors="coerce")
    out["premium_estimate"] = out.apply(
        lambda row: _apply_fill_model_to_premium(
            entry_premium=row.get("entry_premium"),
            spread=row.get("spread"),
            fill_model=fill_model,
        ),
        axis=1,
    )
    out["decision_mode"] = decision_mode.value
    out["policy_status"] = "ok"
    out["policy_reason"] = None

    skip_snapshot = out.get("skip_reason")
    if skip_snapshot is None:
        skip_snapshot = pd.Series([None] * len(out), index=out.index, dtype="object")
    skip_label = out.get("label_skip_reason")
    if skip_label is None:
        skip_label = pd.Series([None] * len(out), index=out.index, dtype="object")
    invalid_premium = ~pd.to_numeric(out["premium_estimate"], errors="coerce").gt(0.0)
    has_snapshot_skip = skip_snapshot.notna() & skip_snapshot.astype(str).str.strip().ne("")
    has_label_skip = skip_label.notna() & skip_label.astype(str).str.strip().ne("")
    has_skip = invalid_premium | has_snapshot_skip | has_label_skip
    out.loc[has_skip, "policy_status"] = "skip"
    out.loc[has_skip, "policy_reason"] = out.loc[has_skip].apply(
        lambda row: _first_text(row.get("skip_reason"), row.get("label_skip_reason"), "invalid_premium_estimate"),
        axis=1,
    )
    return out


def _finalize_candidate_rows(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    out["feature_status"] = out.get("feature_status").fillna("unknown")
    out["label_status"] = out.get("label_status").fillna("unknown")
    out["quote_quality_status"] = out.get("quote_quality_status").fillna(QuoteQualityStatus.UNKNOWN.value)
    out["decision_bar_completed_ts"] = pd.to_datetime(
        out.get("decision_bar_completed_ts"),
        errors="coerce",
        utc=True,
    )
    out["close_label_ts"] = pd.to_datetime(out.get("close_label_ts"), errors="coerce", utc=True)
    keep_cols = [
        "session_date",
        "decision_ts",
        "decision_mode",
        "decision_bar_completed_ts",
        "entry_anchor_ts",
        "close_label_ts",
        "time_of_day_bucket",
        "intraday_return",
        "iv_regime",
        "strike_return",
        "strike_price",
        "target_strike_price",
        "premium_estimate",
        "quote_quality_status",
        "policy_status",
        "policy_reason",
        "close_price",
        "entry_anchor_price",
        "close_return_from_entry",
        "feature_status",
        "label_status",
    ]
    result = out.loc[:, keep_cols].copy()
    result["intraday_min_return_from_entry"] = float("nan")
    result["intraday_max_return_from_entry"] = float("nan")
    result["adaptive_exit_premium"] = float("nan")
    return result


def _expand_risk_tiers(frame: pd.DataFrame, *, risk_tiers: tuple[float, ...]) -> pd.DataFrame:
    if frame.empty:
        return frame
    parts = []
    for tier in risk_tiers:
        part = frame.copy()
        part["risk_tier"] = float(tier)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _apply_fill_model_to_premium(*, entry_premium: object, spread: object, fill_model: FillModel) -> float:
    premium = _as_float(entry_premium)
    if premium is None or premium <= 0:
        return float("nan")
    spread_value = _as_float(spread)
    if spread_value is None or spread_value < 0.0:
        spread_value = 0.0
    if fill_model == FillModel.BID:
        return premium
    if fill_model == FillModel.MID:
        return premium + (spread_value / 2.0)
    return premium + spread_value


__all__ = [
    "_build_zero_dte_candidates",
    "_resolve_zero_dte_study_range",
    "_resolve_latest_intraday_session",
    "_resolve_decision_times_for_session",
    "_resolve_previous_close",
    "_build_strike_snapshot_rows",
    "_assemble_zero_dte_candidate_rows",
    "_expand_risk_tiers",
    "_apply_fill_model_to_premium",
]
