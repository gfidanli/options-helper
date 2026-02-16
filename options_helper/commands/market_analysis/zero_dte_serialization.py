from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Any

import pandas as pd

from options_helper.analysis.zero_dte_tail_model import (
    ZeroDTETailModel,
    ZeroDTETailModelConfig,
    fit_zero_dte_tail_model,
)
from options_helper.commands.market_analysis.core_helpers import _as_float
from options_helper.commands.market_analysis.zero_dte_types import _ZERO_DTE_DEFAULT_STRIKE_GRID
from options_helper.commands.market_analysis.zero_dte_utils import (
    _as_clean_text,
    _coerce_quote_quality_status,
    _coerce_skip_reason,
    _frame_records,
    _resolve_as_of_date,
    _timestamp_to_iso,
)
from options_helper.data.zero_dte_dataset import DEFAULT_PROXY_UNDERLYING
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.zero_dte_put_study import (
    DecisionAnchorMetadata,
    DecisionMode,
    FillModel,
    QuoteQualityStatus,
    ZeroDteProbabilityRow,
    ZeroDteSimulationRow,
    ZeroDteStrikeLadderRow,
    ZeroDteStudyAssumptions,
)


def _build_zero_dte_probability_rows(
    scored_rows: pd.DataFrame,
    *,
    assumptions_hash: str,
) -> list[ZeroDteProbabilityRow]:
    if scored_rows is None or scored_rows.empty:
        return []
    out: list[ZeroDteProbabilityRow] = []
    for _, raw in scored_rows.iterrows():
        risk_tier = _as_float(raw.get("risk_tier"))
        strike_return = _as_float(raw.get("strike_return"))
        probability = _as_float(raw.get("breach_probability"))
        if risk_tier is None or strike_return is None or probability is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        out.append(
            ZeroDteProbabilityRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                risk_tier=risk_tier,
                strike_return=strike_return,
                breach_probability=probability,
                breach_probability_ci_low=_as_float(raw.get("breach_probability_ci_low")),
                breach_probability_ci_high=_as_float(raw.get("breach_probability_ci_high")),
                sample_size=int(_as_float(raw.get("sample_size")) or 0),
                model_version=str(raw.get("model_version") or "unknown_model"),
                assumptions_hash=assumptions_hash,
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("policy_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_strike_ladder_rows(
    recommendations: pd.DataFrame,
    *,
    fill_model: FillModel,
    top_k_per_tier: int,
) -> list[ZeroDteStrikeLadderRow]:
    if recommendations is None or recommendations.empty:
        return []
    out: list[ZeroDteStrikeLadderRow] = []
    ordered = recommendations.sort_values(
        by=["decision_ts", "risk_tier", "ladder_rank"],
        ascending=[True, True, True],
        kind="mergesort",
    )
    for _, raw in ordered.iterrows():
        rank = int(_as_float(raw.get("ladder_rank")) or 1)
        if rank > top_k_per_tier:
            continue
        risk_tier = _as_float(raw.get("risk_tier"))
        strike_price = _as_float(raw.get("strike_price"))
        strike_return = _as_float(raw.get("strike_return"))
        breach_probability = _as_float(raw.get("breach_probability"))
        if risk_tier is None or strike_price is None or strike_return is None or breach_probability is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        out.append(
            ZeroDteStrikeLadderRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                risk_tier=risk_tier,
                ladder_rank=rank,
                strike_price=strike_price,
                strike_return=strike_return,
                breach_probability=breach_probability,
                premium_estimate=_as_float(raw.get("premium_estimate")),
                fill_model=fill_model,
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("policy_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_simulation_rows(
    trade_rows: pd.DataFrame,
    *,
    fill_model: FillModel,
) -> list[ZeroDteSimulationRow]:
    if trade_rows is None or trade_rows.empty:
        return []
    out: list[ZeroDteSimulationRow] = []
    filled = trade_rows.loc[
        trade_rows.get("status", pd.Series(dtype=object)).astype(str).str.lower().eq("filled")
    ].copy()
    for _, raw in filled.iterrows():
        risk_tier = _as_float(raw.get("risk_tier"))
        if risk_tier is None:
            continue
        anchor = _build_zero_dte_anchor(raw)
        if anchor is None:
            continue
        mode = str(raw.get("exit_mode") or "").strip().lower()
        if mode not in {"hold_to_close", "adaptive_exit"}:
            continue
        out.append(
            ZeroDteSimulationRow(
                symbol=str(raw.get("symbol") or DEFAULT_PROXY_UNDERLYING),
                contract_symbol=_as_clean_text(raw.get("contract_symbol")),
                risk_tier=risk_tier,
                exit_mode=mode,
                fill_model=fill_model,
                entry_premium=_as_float(raw.get("entry_premium_gross")),
                exit_premium=_as_float(raw.get("exit_premium_gross")),
                pnl_per_contract=_as_float(raw.get("pnl_per_contract")),
                max_loss_proxy=_as_float(raw.get("max_loss_proxy_per_contract")),
                quote_quality_status=_coerce_quote_quality_status(raw.get("quote_quality_status")),
                skip_reason=_coerce_skip_reason(raw.get("skip_reason")),
                anchor=anchor,
            )
        )
    return out


def _build_zero_dte_anchor(row: pd.Series) -> DecisionAnchorMetadata | None:
    session = pd.to_datetime(row.get("session_date"), errors="coerce")
    decision_ts = pd.to_datetime(row.get("decision_ts"), errors="coerce", utc=True)
    decision_bar_ts = pd.to_datetime(
        row.get("decision_bar_completed_ts") or row.get("decision_ts"),
        errors="coerce",
        utc=True,
    )
    close_ts = pd.to_datetime(row.get("close_label_ts"), errors="coerce", utc=True)
    if pd.isna(session) or pd.isna(decision_ts) or pd.isna(decision_bar_ts) or pd.isna(close_ts):
        return None
    mode_text = str(row.get("decision_mode") or DecisionMode.FIXED_TIME.value).strip().lower()
    mode = DecisionMode(mode_text) if mode_text in {item.value for item in DecisionMode} else DecisionMode.FIXED_TIME
    entry_anchor = pd.to_datetime(row.get("entry_anchor_ts"), errors="coerce", utc=True)
    return DecisionAnchorMetadata(
        session_date=session.date(),
        decision_ts=decision_ts.to_pydatetime(),
        decision_bar_completed_ts=decision_bar_ts.to_pydatetime(),
        close_label_ts=close_ts.to_pydatetime(),
        decision_mode=mode,
        entry_anchor_ts=entry_anchor.to_pydatetime() if pd.notna(entry_anchor) else None,
    )


def _fit_and_serialize_active_model(
    candidates: pd.DataFrame,
    *,
    symbol: str,
    strike_grid: tuple[float, ...],
    assumptions: ZeroDteStudyAssumptions,
    assumptions_hash: str,
    fallback_trained_through: date,
) -> dict[str, Any] | None:
    if candidates is None or candidates.empty:
        return None
    training = candidates.copy()
    training = training.loc[
        training["feature_status"].astype(str).str.lower().eq("ok")
        & training["label_status"].astype(str).str.lower().eq("ok")
    ].copy()
    training = training.loc[pd.to_numeric(training["close_return_from_entry"], errors="coerce").notna()].copy()
    training = training.sort_values(
        by=["session_date", "decision_ts", "entry_anchor_ts"],
        ascending=[True, True, True],
        kind="mergesort",
    ).drop_duplicates(subset=["session_date", "decision_ts", "entry_anchor_ts"], keep="first")
    if training.empty:
        return None
    model = fit_zero_dte_tail_model(training, strike_returns=strike_grid)
    trained_through = _resolve_as_of_date(training, default=fallback_trained_through)
    return _serialize_tail_model_payload(
        model=model,
        symbol=symbol,
        model_version=f"active_{trained_through.isoformat()}",
        trained_through_session=trained_through,
        assumptions=assumptions,
        assumptions_hash=assumptions_hash,
    )


def _serialize_tail_model_payload(
    *,
    model: ZeroDTETailModel,
    symbol: str,
    model_version: str,
    trained_through_session: date,
    assumptions: ZeroDteStudyAssumptions,
    assumptions_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": utc_now().isoformat(),
        "symbol": symbol,
        "model_version": model_version,
        "trained_through_session": trained_through_session.isoformat(),
        "assumptions_hash": assumptions_hash,
        "assumptions": clean_nan(assumptions.model_dump(mode="json")),
        "tail_model": {
            "config": asdict(model.config),
            "strike_returns": list(model.strike_returns),
            "training_sample_size": int(model.training_sample_size),
            "global_stats": _frame_records(model.global_stats),
            "parent_stats": _frame_records(model.parent_stats),
            "bucket_stats": _frame_records(model.bucket_stats),
        },
    }


def _deserialize_tail_model(payload: dict[str, Any]) -> ZeroDTETailModel:
    config = ZeroDTETailModelConfig(**(payload.get("config") or {}))
    strike_returns = tuple(float(v) for v in payload.get("strike_returns") or _ZERO_DTE_DEFAULT_STRIKE_GRID)
    return ZeroDTETailModel(
        config=config,
        strike_returns=strike_returns,
        training_sample_size=int(payload.get("training_sample_size", 0)),
        global_stats=pd.DataFrame(payload.get("global_stats") or []),
        parent_stats=pd.DataFrame(payload.get("parent_stats") or []),
        bucket_stats=pd.DataFrame(payload.get("bucket_stats") or []),
    )


def _build_forward_snapshot_rows(
    *,
    recommendations: pd.DataFrame,
    scored_rows: pd.DataFrame,
    symbol: str,
    model_version: str,
    assumptions_hash: str,
    trained_through_session: str,
) -> list[dict[str, Any]]:
    if recommendations is None or recommendations.empty:
        return []
    lookup = scored_rows.loc[
        :,
        [
            "session_date",
            "decision_ts",
            "entry_anchor_ts",
            "risk_tier",
            "strike_return",
            "close_return_from_entry",
            "close_price",
            "close_label_ts",
        ],
    ].copy()
    merged = recommendations.merge(
        lookup,
        on=["session_date", "decision_ts", "entry_anchor_ts", "risk_tier", "strike_return"],
        how="left",
        sort=True,
    )
    out: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        session = pd.to_datetime(row.get("session_date"), errors="coerce")
        decision_ts = pd.to_datetime(row.get("decision_ts"), errors="coerce", utc=True)
        risk_tier = _as_float(row.get("risk_tier"))
        if pd.isna(session) or pd.isna(decision_ts) or risk_tier is None:
            continue
        realized = _as_float(row.get("close_return_from_entry"))
        out.append(
            clean_nan(
                {
                    "symbol": symbol,
                    "session_date": session.date().isoformat(),
                    "decision_ts": decision_ts.isoformat(),
                    "entry_anchor_ts": _timestamp_to_iso(row.get("entry_anchor_ts")),
                    "close_label_ts": _timestamp_to_iso(row.get("close_label_ts")),
                    "risk_tier": risk_tier,
                    "ladder_rank": int(_as_float(row.get("ladder_rank")) or 1),
                    "strike_return": _as_float(row.get("strike_return")),
                    "strike_price": _as_float(row.get("strike_price")),
                    "breach_probability": _as_float(row.get("breach_probability")),
                    "breach_probability_ci_low": _as_float(row.get("breach_probability_ci_low")),
                    "breach_probability_ci_high": _as_float(row.get("breach_probability_ci_high")),
                    "premium_estimate": _as_float(row.get("premium_estimate")),
                    "quote_quality_status": _as_clean_text(row.get("quote_quality_status"))
                    or QuoteQualityStatus.UNKNOWN.value,
                    "policy_status": _as_clean_text(row.get("policy_status")) or "ok",
                    "policy_reason": _as_clean_text(row.get("policy_reason")),
                    "model_version": model_version,
                    "assumptions_hash": assumptions_hash,
                    "trained_through_session": trained_through_session,
                    "realized_close_return_from_entry": realized,
                    "realized_close_price": _as_float(row.get("close_price")),
                    "reconciliation_status": "finalized" if realized is not None else "pending_close",
                    "generated_at": utc_now().isoformat(),
                }
            )
        )
    return out


__all__ = [
    "_build_zero_dte_probability_rows",
    "_build_zero_dte_strike_ladder_rows",
    "_build_zero_dte_simulation_rows",
    "_build_zero_dte_anchor",
    "_fit_and_serialize_active_model",
    "_serialize_tail_model_payload",
    "_deserialize_tail_model",
    "_build_forward_snapshot_rows",
]
