from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from options_helper.analysis.zero_dte_calibration import (
    ZeroDTECalibrationConfig,
    compute_zero_dte_calibration,
)
from options_helper.analysis.zero_dte_tail_model import (
    ZeroDTETailModelConfig,
    fit_zero_dte_tail_model,
    score_zero_dte_tail_model,
)
from options_helper.backtesting.zero_dte_put import (
    ZeroDTEPutSimulatorConfig,
    simulate_zero_dte_put_outcomes,
)


_SCORED_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "entry_anchor_ts",
    "close_label_ts",
    "risk_tier",
    "decision_mode",
    "time_of_day_bucket",
    "iv_regime",
    "intraday_return",
    "strike_return",
    "strike_price",
    "premium_estimate",
    "quote_quality_status",
    "policy_status",
    "policy_reason",
    "close_price",
    "entry_anchor_price",
    "close_return_from_entry",
    "intraday_min_return_from_entry",
    "intraday_max_return_from_entry",
    "adaptive_exit_premium",
    "adaptive_exit_ts",
    "adaptive_stop_exit_ts",
    "adaptive_take_profit_exit_ts",
    "breach_probability",
    "breach_probability_ci_low",
    "breach_probability_ci_high",
    "sample_size",
    "breach_observed",
    "fold_id",
    "model_version",
    "trained_through_session",
)


@dataclass(frozen=True)
class ZeroDTEWalkForwardConfig:
    train_sessions: int = 20
    test_sessions: int = 5
    step_sessions: int = 5
    min_training_rows: int = 20
    strike_returns: tuple[float, ...] | None = None
    tail_model_config: ZeroDTETailModelConfig | None = None
    calibration_bins: int = 10
    simulator_config: ZeroDTEPutSimulatorConfig | None = None

    def __post_init__(self) -> None:
        if self.train_sessions < 1:
            raise ValueError("train_sessions must be >= 1")
        if self.test_sessions < 1:
            raise ValueError("test_sessions must be >= 1")
        if self.step_sessions < 1:
            raise ValueError("step_sessions must be >= 1")
        if self.min_training_rows < 1:
            raise ValueError("min_training_rows must be >= 1")
        if self.calibration_bins < 2:
            raise ValueError("calibration_bins must be >= 2")


@dataclass(frozen=True)
class ZeroDTEWalkForwardResult:
    folds: list[dict[str, object]]
    scored_rows: pd.DataFrame
    model_snapshots: pd.DataFrame
    calibration_summary: pd.DataFrame
    trade_rows: pd.DataFrame
    trade_summary: pd.DataFrame


def run_zero_dte_walk_forward(
    rows: pd.DataFrame,
    *,
    config: ZeroDTEWalkForwardConfig | None = None,
) -> ZeroDTEWalkForwardResult:
    cfg = config or ZeroDTEWalkForwardConfig()
    normalized = _normalize_rows(rows)
    sessions = _sorted_sessions(normalized)
    splits = _generate_splits(
        sessions=sessions,
        train_sessions=cfg.train_sessions,
        test_sessions=cfg.test_sessions,
        step_sessions=cfg.step_sessions,
    )
    if not splits:
        empty = pd.DataFrame()
        return ZeroDTEWalkForwardResult(
            folds=[],
            scored_rows=empty,
            model_snapshots=empty,
            calibration_summary=empty,
            trade_rows=empty,
            trade_summary=empty,
        )

    strike_grid = _resolve_strike_grid(normalized, configured=cfg.strike_returns)
    tail_cfg = cfg.tail_model_config or ZeroDTETailModelConfig()

    scored_parts: list[pd.DataFrame] = []
    snapshot_rows: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []

    session_to_idx = {value: idx for idx, value in enumerate(sessions)}

    for fold_id, split in enumerate(splits, start=1):
        fold_train_sessions = split["train_sessions"]
        fold_test_sessions = split["test_sessions"]
        fold_scored_parts: list[pd.DataFrame] = []

        for test_session in fold_test_sessions:
            test_idx = session_to_idx[test_session]
            train_slice = sessions[max(0, test_idx - cfg.train_sessions) : test_idx]
            if len(train_slice) < cfg.train_sessions:
                continue
            if any(train_day >= test_session for train_day in train_slice):
                raise ValueError("Detected non-causal train/test split; training includes test/future day")

            train_frame = normalized.loc[normalized["session_date"].isin(train_slice)].copy()
            train_events = _training_event_rows(train_frame)
            if len(train_events) < cfg.min_training_rows:
                continue

            model = fit_zero_dte_tail_model(
                train_events,
                strike_returns=strike_grid,
                config=tail_cfg,
            )

            test_frame = normalized.loc[normalized["session_date"] == test_session].copy()
            state_rows = _state_rows_for_scoring(test_frame)
            if state_rows.empty:
                continue
            scored_state = score_zero_dte_tail_model(model, state_rows, strike_returns=strike_grid)
            if scored_state.empty:
                continue

            joined = _join_predictions_with_candidates(test_frame, scored_state)
            if joined.empty:
                continue

            trained_through = max(train_slice)
            if trained_through >= test_session:
                raise ValueError("trained_through_session must be strictly before test session")

            model_version = f"wf_fold{fold_id}_{test_session.isoformat()}"
            joined["fold_id"] = int(fold_id)
            joined["model_version"] = model_version
            joined["trained_through_session"] = trained_through
            joined["breach_observed"] = (
                pd.to_numeric(joined["close_return_from_entry"], errors="coerce")
                <= pd.to_numeric(joined["strike_return"], errors="coerce")
            ).astype("float64")
            joined.loc[
                pd.to_numeric(joined["close_return_from_entry"], errors="coerce").isna(),
                "breach_observed",
            ] = float("nan")

            fold_scored_parts.append(joined.loc[:, list(_SCORED_COLUMNS)].copy())
            snapshot_rows.append(
                {
                    "fold_id": int(fold_id),
                    "model_version": model_version,
                    "session_date": test_session,
                    "trained_through_session": trained_through,
                    "train_session_start": min(train_slice),
                    "train_session_end": max(train_slice),
                    "train_row_count": int(len(train_events)),
                    "training_sample_size": int(model.training_sample_size),
                }
            )

        fold_scored = _concat_or_empty(fold_scored_parts, columns=list(_SCORED_COLUMNS))
        folds.append(
            {
                "fold_id": int(fold_id),
                "train_sessions": fold_train_sessions,
                "test_sessions": fold_test_sessions,
                "scored_rows": int(len(fold_scored)),
            }
        )
        if not fold_scored.empty:
            scored_parts.append(fold_scored)

    scored_rows = _concat_or_empty(scored_parts, columns=list(_SCORED_COLUMNS))
    if not scored_rows.empty:
        _assert_no_future_leakage(scored_rows)

    snapshots = pd.DataFrame(snapshot_rows)
    if snapshots.empty:
        snapshots = pd.DataFrame(
            columns=[
                "fold_id",
                "model_version",
                "session_date",
                "trained_through_session",
                "train_session_start",
                "train_session_end",
                "train_row_count",
                "training_sample_size",
            ]
        )
    else:
        snapshots = snapshots.sort_values(
            by=["session_date", "fold_id"],
            ascending=[True, True],
            kind="mergesort",
        ).reset_index(drop=True)

    calibration_summary = _summarize_calibration(scored_rows, bins=cfg.calibration_bins)
    simulator_cfg = cfg.simulator_config or ZeroDTEPutSimulatorConfig()
    trade_rows = simulate_zero_dte_put_outcomes(scored_rows, config=simulator_cfg)
    trade_summary = _summarize_trades(trade_rows)

    return ZeroDTEWalkForwardResult(
        folds=folds,
        scored_rows=scored_rows,
        model_snapshots=snapshots,
        calibration_summary=calibration_summary,
        trade_rows=trade_rows,
        trade_summary=trade_summary,
    )


def _summarize_calibration(scored_rows: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    if scored_rows is None or scored_rows.empty:
        return pd.DataFrame(
            columns=[
                "risk_tier",
                "decision_mode",
                "time_of_day_bucket",
                "iv_regime",
                "sample_size",
                "brier_score",
                "observed_rate",
                "predicted_mean",
                "sharpness",
                "expected_calibration_error",
            ]
        )

    group_cols = ["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime"]
    rows: list[dict[str, object]] = []
    grouped = scored_rows.groupby(group_cols, sort=True, dropna=False)
    for keys, subset in grouped:
        clean = subset.loc[
            pd.to_numeric(subset["breach_probability"], errors="coerce").between(0.0, 1.0, inclusive="both")
            & pd.to_numeric(subset["breach_observed"], errors="coerce").isin([0.0, 1.0])
        ].copy()
        if clean.empty:
            continue
        result = compute_zero_dte_calibration(
            clean,
            config=ZeroDTECalibrationConfig(num_bins=bins),
        )
        rows.append(
            {
                "risk_tier": keys[0],
                "decision_mode": keys[1],
                "time_of_day_bucket": keys[2],
                "iv_regime": keys[3],
                "sample_size": result.sample_size,
                "brier_score": result.brier_score,
                "observed_rate": result.observed_rate,
                "predicted_mean": result.predicted_mean,
                "sharpness": result.sharpness,
                "expected_calibration_error": result.expected_calibration_error,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "risk_tier",
                "decision_mode",
                "time_of_day_bucket",
                "iv_regime",
                "sample_size",
                "brier_score",
                "observed_rate",
                "predicted_mean",
                "sharpness",
                "expected_calibration_error",
            ]
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        by=["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _summarize_trades(trade_rows: pd.DataFrame) -> pd.DataFrame:
    if trade_rows is None or trade_rows.empty:
        return pd.DataFrame(
            columns=[
                "risk_tier",
                "decision_mode",
                "time_of_day_bucket",
                "iv_regime",
                "exit_mode",
                "filled_trades",
                "skipped_trades",
                "total_pnl",
                "avg_pnl",
                "win_rate",
            ]
        )

    trades = trade_rows.copy()
    status = trades["status"].astype(str).str.lower() if "status" in trades.columns else pd.Series()
    trades["filled"] = status == "filled"
    trades["pnl_total_num"] = pd.to_numeric(trades.get("pnl_total"), errors="coerce")
    trades["is_win"] = trades["filled"] & trades["pnl_total_num"].gt(0.0)

    group_cols = ["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime", "exit_mode"]
    out = (
        trades.groupby(group_cols, sort=True, dropna=False)
        .agg(
            filled_trades=("filled", "sum"),
            skipped_trades=("filled", lambda s: int((~s.astype(bool)).sum())),
            total_pnl=("pnl_total_num", "sum"),
            avg_pnl=("pnl_total_num", "mean"),
            win_rate=("is_win", lambda s: float(s.sum()) / max(int((trades.loc[s.index, "filled"]).sum()), 1)),
        )
        .reset_index()
    )
    return out.sort_values(
        by=["risk_tier", "decision_mode", "time_of_day_bucket", "iv_regime", "exit_mode"],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _training_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean = clean.loc[_valid_anchor_mask(clean)].copy()
    if "feature_status" in clean.columns:
        clean = clean.loc[clean["feature_status"].astype(str).str.lower().eq("ok")].copy()
    if "label_status" in clean.columns:
        clean = clean.loc[clean["label_status"].astype(str).str.lower().eq("ok")].copy()
    clean = clean.loc[pd.to_numeric(clean["close_return_from_entry"], errors="coerce").notna()].copy()

    dedupe_cols = ["session_date", "decision_ts", "entry_anchor_ts"]
    keep_cols = [
        "session_date",
        "decision_ts",
        "time_of_day_bucket",
        "intraday_return",
        "iv_regime",
        "close_return_from_entry",
        "feature_status",
        "label_status",
    ]
    for col in keep_cols:
        if col not in clean.columns:
            clean[col] = None
    clean = clean.sort_values(
        by=["session_date", "decision_ts", "entry_anchor_ts"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    return clean.drop_duplicates(subset=dedupe_cols, keep="first").loc[:, keep_cols].reset_index(drop=True)


def _state_rows_for_scoring(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    clean = frame.copy()
    clean = clean.loc[_valid_anchor_mask(clean)].copy()
    if "feature_status" in clean.columns:
        clean = clean.loc[clean["feature_status"].astype(str).str.lower().eq("ok")].copy()
    keep_cols = [
        "session_date",
        "decision_ts",
        "time_of_day_bucket",
        "intraday_return",
        "iv_regime",
    ]
    for col in keep_cols:
        if col not in clean.columns:
            clean[col] = None
    clean = clean.sort_values(
        by=["session_date", "decision_ts"],
        ascending=[True, True],
        kind="mergesort",
    )
    return clean.drop_duplicates(subset=["session_date", "decision_ts"], keep="first").loc[:, keep_cols]


def _join_predictions_with_candidates(candidates: pd.DataFrame, scored_state: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty or scored_state.empty:
        return pd.DataFrame(columns=list(_SCORED_COLUMNS))
    working = candidates.copy()
    join_cols = ["session_date", "decision_ts", "strike_return"]
    merged = working.merge(
        scored_state.loc[
            :,
            [
                "session_date",
                "decision_ts",
                "strike_return",
                "breach_probability",
                "breach_probability_ci_low",
                "breach_probability_ci_high",
                "sample_size",
            ],
        ],
        how="left",
        on=join_cols,
        sort=True,
    )
    for col in _SCORED_COLUMNS:
        if col not in merged.columns:
            merged[col] = None
    return merged


def _valid_anchor_mask(frame: pd.DataFrame) -> pd.Series:
    decision = pd.to_datetime(frame.get("decision_ts"), errors="coerce", utc=True)
    entry_anchor = pd.to_datetime(frame.get("entry_anchor_ts"), errors="coerce", utc=True)
    close_label = pd.to_datetime(frame.get("close_label_ts"), errors="coerce", utc=True)
    return decision.notna() & entry_anchor.notna() & close_label.notna() & (entry_anchor > decision) & (
        close_label >= entry_anchor
    )


def _assert_no_future_leakage(scored_rows: pd.DataFrame) -> None:
    session_ts = pd.to_datetime(scored_rows["session_date"], errors="coerce")
    trained_ts = pd.to_datetime(scored_rows["trained_through_session"], errors="coerce")
    leakage = trained_ts >= session_ts
    if bool(leakage.fillna(False).any()):
        raise ValueError("Detected future leakage: trained_through_session must be before session_date")


def _resolve_strike_grid(frame: pd.DataFrame, *, configured: tuple[float, ...] | None) -> tuple[float, ...]:
    if configured:
        return tuple(sorted({float(item) for item in configured}))
    if frame.empty or "strike_return" not in frame.columns:
        return (-0.01, -0.02, -0.03)
    values = pd.to_numeric(frame["strike_return"], errors="coerce").dropna().astype(float).tolist()
    if not values:
        return (-0.01, -0.02, -0.03)
    return tuple(sorted(set(values)))


def _generate_splits(
    *,
    sessions: list[date],
    train_sessions: int,
    test_sessions: int,
    step_sessions: int,
) -> list[dict[str, list[date]]]:
    if len(sessions) < (train_sessions + test_sessions):
        return []
    out: list[dict[str, list[date]]] = []
    test_start = train_sessions
    while test_start + test_sessions <= len(sessions):
        out.append(
            {
                "train_sessions": sessions[test_start - train_sessions : test_start],
                "test_sessions": sessions[test_start : test_start + test_sessions],
            }
        )
        test_start += step_sessions
    return out


def _sorted_sessions(frame: pd.DataFrame) -> list[date]:
    if frame.empty or "session_date" not in frame.columns:
        return []
    values = pd.to_datetime(frame["session_date"], errors="coerce").dropna().dt.date.unique().tolist()
    return sorted(values)


def _normalize_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(_SCORED_COLUMNS))
    out = frame.copy()
    out.columns = [str(col) for col in out.columns]

    out["session_date"] = pd.to_datetime(out.get("session_date"), errors="coerce").dt.date
    for ts_col in (
        "decision_ts",
        "entry_anchor_ts",
        "close_label_ts",
        "adaptive_exit_ts",
        "adaptive_stop_exit_ts",
        "adaptive_take_profit_exit_ts",
    ):
        out[ts_col] = pd.to_datetime(out.get(ts_col), errors="coerce", utc=True)

    numeric_cols = (
        "risk_tier",
        "intraday_return",
        "strike_return",
        "strike_price",
        "target_strike_price",
        "premium_estimate",
        "entry_premium",
        "close_price",
        "entry_anchor_price",
        "close_return_from_entry",
        "intraday_min_return_from_entry",
        "intraday_max_return_from_entry",
        "adaptive_exit_premium",
    )
    for col in numeric_cols:
        out[col] = pd.to_numeric(out.get(col), errors="coerce")

    if "strike_price" not in out.columns or out["strike_price"].isna().all():
        out["strike_price"] = pd.to_numeric(out.get("target_strike_price"), errors="coerce")
    if "premium_estimate" not in out.columns or out["premium_estimate"].isna().all():
        out["premium_estimate"] = pd.to_numeric(out.get("entry_premium"), errors="coerce")

    defaults: dict[str, object] = {
        "decision_mode": "fixed_time",
        "time_of_day_bucket": "unknown_time",
        "iv_regime": "unknown_regime",
        "quote_quality_status": "unknown",
        "policy_status": "ok",
        "policy_reason": None,
        "feature_status": "ok",
        "label_status": "ok",
    }
    for key, value in defaults.items():
        if key not in out.columns:
            out[key] = value
        else:
            if value is None:
                out[key] = out[key].where(out[key].notna(), None)
            else:
                out[key] = out[key].fillna(value)

    out = out.sort_values(
        by=["session_date", "decision_ts", "risk_tier", "strike_return"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _concat_or_empty(parts: list[pd.DataFrame], *, columns: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=columns)
    out = pd.concat(parts, ignore_index=True)
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out.loc[:, columns].copy()


__all__ = [
    "ZeroDTEWalkForwardConfig",
    "ZeroDTEWalkForwardResult",
    "run_zero_dte_walk_forward",
]
