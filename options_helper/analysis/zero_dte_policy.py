from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import pandas as pd


_POLICY_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "entry_anchor_ts",
    "risk_tier",
    "ladder_rank",
    "strike_return",
    "strike_price",
    "breach_probability",
    "breach_probability_ci_low",
    "breach_probability_ci_high",
    "sample_size",
    "premium_estimate",
    "quote_quality_status",
    "ev_proxy_low",
    "ev_proxy_mid",
    "ev_proxy_high",
    "policy_status",
    "policy_reason",
    "fallback_used",
)


@dataclass(frozen=True)
class ZeroDTEPolicyConfig:
    allowed_quote_quality_statuses: tuple[str, ...] = ("good", "unknown")
    top_k_per_tier: int = 3

    def __post_init__(self) -> None:
        if self.top_k_per_tier < 1:
            raise ValueError("top_k_per_tier must be >= 1")
        if not self.allowed_quote_quality_statuses:
            raise ValueError("allowed_quote_quality_statuses must not be empty")


def recommend_zero_dte_put_strikes(
    probability_rows: pd.DataFrame,
    strike_snapshot: pd.DataFrame,
    *,
    risk_tiers: Iterable[float],
    config: ZeroDTEPolicyConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ZeroDTEPolicyConfig()
    tiers = _normalize_risk_tiers(risk_tiers)
    if not tiers:
        return pd.DataFrame(columns=list(_POLICY_COLUMNS))

    probabilities = _normalize_probability_rows(probability_rows)
    quotes = _normalize_snapshot_rows(strike_snapshot)
    merged = _merge_probability_and_quotes(probabilities, quotes)
    if merged.empty:
        return _empty_policy_rows_for_tiers(tiers, reason="no_data")

    merged["quote_quality_rank"] = merged["quote_quality_status"].map(_quote_quality_rank).fillna(99).astype(int)
    merged["quote_ok"] = merged["quote_quality_status"].isin(
        {status.strip().lower() for status in cfg.allowed_quote_quality_statuses}
    )
    merged["premium_ok"] = pd.to_numeric(merged["premium_estimate"], errors="coerce").gt(0.0)
    merged["skip_ok"] = merged["skip_reason"].isna() | (merged["skip_reason"].astype(str).str.strip() == "")
    merged["probability_ok"] = pd.to_numeric(merged["breach_probability"], errors="coerce").between(0.0, 1.0, inclusive="both")
    merged["candidate_ok"] = merged["quote_ok"] & merged["premium_ok"] & merged["skip_ok"] & merged["probability_ok"]

    group_cols = [col for col in ("session_date", "decision_ts", "entry_anchor_ts") if col in merged.columns]
    grouped = merged.groupby(group_cols, sort=True, dropna=False) if group_cols else [((), merged)]

    all_rows: list[dict[str, object]] = []
    for _, group in grouped:
        all_rows.extend(_recommend_group(group, tiers=tiers, config=cfg))

    out = pd.DataFrame(all_rows, columns=list(_POLICY_COLUMNS))
    if out.empty:
        return pd.DataFrame(columns=list(_POLICY_COLUMNS))
    return out.sort_values(
        by=["decision_ts", "risk_tier", "ladder_rank"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _recommend_group(
    group: pd.DataFrame,
    *,
    tiers: tuple[float, ...],
    config: ZeroDTEPolicyConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    meta = {
        "session_date": group.iloc[0].get("session_date") if not group.empty else None,
        "decision_ts": group.iloc[0].get("decision_ts") if not group.empty else pd.NaT,
        "entry_anchor_ts": group.iloc[0].get("entry_anchor_ts") if not group.empty else pd.NaT,
    }

    candidates = group.loc[group["candidate_ok"]].copy()
    for tier in tiers:
        if candidates.empty:
            reason = _infer_skip_reason(group)
            rows.append(
                {
                    **meta,
                    "risk_tier": tier,
                    "ladder_rank": 1,
                    "strike_return": float("nan"),
                    "strike_price": float("nan"),
                    "breach_probability": float("nan"),
                    "breach_probability_ci_low": float("nan"),
                    "breach_probability_ci_high": float("nan"),
                    "sample_size": 0,
                    "premium_estimate": float("nan"),
                    "quote_quality_status": "missing",
                    "ev_proxy_low": float("nan"),
                    "ev_proxy_mid": float("nan"),
                    "ev_proxy_high": float("nan"),
                    "policy_status": "skip",
                    "policy_reason": reason,
                    "fallback_used": False,
                }
            )
            continue

        within_tier = candidates.loc[pd.to_numeric(candidates["breach_probability"], errors="coerce") <= tier].copy()
        fallback_used = False
        policy_reason = "ok"
        working = within_tier
        if working.empty:
            fallback_used = True
            policy_reason = "fallback_no_candidate_within_risk_tier"
            working = candidates.copy()

        ranked = _rank_candidates(working, tier=tier, fallback_mode=fallback_used)
        for rank, (_, selected) in enumerate(ranked.head(config.top_k_per_tier).iterrows(), start=1):
            rows.append(
                {
                    **meta,
                    "risk_tier": tier,
                    "ladder_rank": rank,
                    "strike_return": float(selected.get("strike_return")),
                    "strike_price": float(pd.to_numeric(selected.get("strike_price"), errors="coerce")),
                    "breach_probability": float(pd.to_numeric(selected.get("breach_probability"), errors="coerce")),
                    "breach_probability_ci_low": float(
                        pd.to_numeric(selected.get("breach_probability_ci_low"), errors="coerce")
                    ),
                    "breach_probability_ci_high": float(
                        pd.to_numeric(selected.get("breach_probability_ci_high"), errors="coerce")
                    ),
                    "sample_size": int(pd.to_numeric(selected.get("sample_size"), errors="coerce")),
                    "premium_estimate": float(pd.to_numeric(selected.get("premium_estimate"), errors="coerce")),
                    "quote_quality_status": str(selected.get("quote_quality_status")),
                    "ev_proxy_low": float(pd.to_numeric(selected.get("ev_proxy_low"), errors="coerce")),
                    "ev_proxy_mid": float(pd.to_numeric(selected.get("ev_proxy_mid"), errors="coerce")),
                    "ev_proxy_high": float(pd.to_numeric(selected.get("ev_proxy_high"), errors="coerce")),
                    "policy_status": "fallback" if fallback_used else "ok",
                    "policy_reason": policy_reason,
                    "fallback_used": fallback_used,
                }
            )
    return rows


def _rank_candidates(frame: pd.DataFrame, *, tier: float, fallback_mode: bool) -> pd.DataFrame:
    ranked = frame.copy()
    breach = pd.to_numeric(ranked["breach_probability"], errors="coerce")
    premium = pd.to_numeric(ranked["premium_estimate"], errors="coerce")
    ranked["tier_gap"] = (float(tier) - breach).abs() if fallback_mode else (float(tier) - breach)
    ranked["tier_gap"] = ranked["tier_gap"].abs()
    ranked["premium_sort"] = -premium
    ranked["ev_sort"] = -pd.to_numeric(ranked["ev_proxy_mid"], errors="coerce").fillna(float("-inf"))
    ranked = ranked.sort_values(
        by=[
            "tier_gap",
            "quote_quality_rank",
            "ev_sort",
            "premium_sort",
            "strike_return",
        ],
        ascending=[True, True, True, True, not fallback_mode],
        kind="mergesort",
    )
    return ranked


def _merge_probability_and_quotes(probabilities: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    if probabilities.empty and quotes.empty:
        return pd.DataFrame()
    if probabilities.empty:
        merged = quotes.copy()
        merged["breach_probability"] = float("nan")
        merged["breach_probability_ci_low"] = float("nan")
        merged["breach_probability_ci_high"] = float("nan")
        merged["sample_size"] = 0
        return _add_ev_proxy(merged)
    if quotes.empty:
        merged = probabilities.copy()
        merged["strike_price"] = float("nan")
        merged["premium_estimate"] = float("nan")
        merged["quote_quality_status"] = "missing"
        merged["skip_reason"] = "missing_quotes"
        return _add_ev_proxy(merged)

    join_cols = [col for col in ("session_date", "decision_ts", "entry_anchor_ts", "strike_return") if col in probabilities.columns and col in quotes.columns]
    if "strike_return" not in join_cols:
        join_cols.append("strike_return")
    merged = probabilities.merge(
        quotes,
        on=join_cols,
        how="outer",
        suffixes=("_prob", "_quote"),
        sort=True,
    )
    for col in ("session_date", "decision_ts", "entry_anchor_ts", "strike_price", "premium_estimate", "quote_quality_status", "skip_reason"):
        if col not in merged.columns:
            merged[col] = float("nan")
    return _add_ev_proxy(merged)


def _add_ev_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    premium = pd.to_numeric(out.get("premium_estimate"), errors="coerce")
    p = pd.to_numeric(out.get("breach_probability"), errors="coerce")
    low = pd.to_numeric(out.get("breach_probability_ci_low"), errors="coerce")
    high = pd.to_numeric(out.get("breach_probability_ci_high"), errors="coerce")

    out["ev_proxy_mid"] = premium * (1.0 - p)
    out["ev_proxy_low"] = premium * (1.0 - high)
    out["ev_proxy_high"] = premium * (1.0 - low)
    return out


def _normalize_probability_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "decision_ts",
                "entry_anchor_ts",
                "strike_return",
                "breach_probability",
                "breach_probability_ci_low",
                "breach_probability_ci_high",
                "sample_size",
            ]
        )

    out = frame.copy()
    out.columns = [str(col) for col in out.columns]
    strike_col = _first_present(out.columns, "strike_return", "target_strike_return")
    if strike_col is None:
        raise ValueError("probability_rows must include strike_return or target_strike_return")
    if strike_col != "strike_return":
        out["strike_return"] = pd.to_numeric(out[strike_col], errors="coerce")
    else:
        out["strike_return"] = pd.to_numeric(out["strike_return"], errors="coerce")

    out["decision_ts"] = pd.to_datetime(out.get("decision_ts"), errors="coerce", utc=True)
    out["entry_anchor_ts"] = pd.to_datetime(out.get("entry_anchor_ts"), errors="coerce", utc=True)
    out["breach_probability"] = pd.to_numeric(out.get("breach_probability"), errors="coerce")
    out["breach_probability_ci_low"] = pd.to_numeric(out.get("breach_probability_ci_low"), errors="coerce")
    out["breach_probability_ci_high"] = pd.to_numeric(out.get("breach_probability_ci_high"), errors="coerce")
    out["sample_size"] = pd.to_numeric(out.get("sample_size"), errors="coerce").fillna(0).astype(int)

    if "session_date" not in out.columns:
        out["session_date"] = None
    needed_cols = [
        "session_date",
        "decision_ts",
        "entry_anchor_ts",
        "strike_return",
        "breach_probability",
        "breach_probability_ci_low",
        "breach_probability_ci_high",
        "sample_size",
    ]
    return out.loc[:, needed_cols].copy()


def _normalize_snapshot_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=[
                "session_date",
                "decision_ts",
                "entry_anchor_ts",
                "strike_return",
                "strike_price",
                "premium_estimate",
                "quote_quality_status",
                "skip_reason",
            ]
        )
    out = frame.copy()
    out.columns = [str(col) for col in out.columns]
    strike_col = _first_present(out.columns, "strike_return", "target_strike_return")
    if strike_col is None:
        raise ValueError("strike_snapshot must include strike_return or target_strike_return")
    out["strike_return"] = pd.to_numeric(out[strike_col], errors="coerce")

    price_col = _first_present(out.columns, "strike_price", "target_strike_price")
    premium_col = _first_present(out.columns, "premium_estimate", "entry_premium")
    quality_col = _first_present(out.columns, "quote_quality_status")
    skip_col = _first_present(out.columns, "skip_reason")

    out["strike_price"] = pd.to_numeric(out[price_col], errors="coerce") if price_col is not None else float("nan")
    out["premium_estimate"] = pd.to_numeric(out[premium_col], errors="coerce") if premium_col is not None else float("nan")
    out["quote_quality_status"] = (
        out[quality_col].astype(str).str.lower().str.strip() if quality_col is not None else "missing"
    )
    out["skip_reason"] = out[skip_col] if skip_col is not None else None
    out["decision_ts"] = pd.to_datetime(out.get("decision_ts"), errors="coerce", utc=True)
    out["entry_anchor_ts"] = pd.to_datetime(out.get("entry_anchor_ts"), errors="coerce", utc=True)
    if "session_date" not in out.columns:
        out["session_date"] = None

    needed_cols = [
        "session_date",
        "decision_ts",
        "entry_anchor_ts",
        "strike_return",
        "strike_price",
        "premium_estimate",
        "quote_quality_status",
        "skip_reason",
    ]
    return out.loc[:, needed_cols].copy()


def _empty_policy_rows_for_tiers(tiers: tuple[float, ...], *, reason: str) -> pd.DataFrame:
    rows = []
    for tier in tiers:
        rows.append(
            {
                "session_date": None,
                "decision_ts": pd.NaT,
                "entry_anchor_ts": pd.NaT,
                "risk_tier": tier,
                "ladder_rank": 1,
                "strike_return": float("nan"),
                "strike_price": float("nan"),
                "breach_probability": float("nan"),
                "breach_probability_ci_low": float("nan"),
                "breach_probability_ci_high": float("nan"),
                "sample_size": 0,
                "premium_estimate": float("nan"),
                "quote_quality_status": "missing",
                "ev_proxy_low": float("nan"),
                "ev_proxy_mid": float("nan"),
                "ev_proxy_high": float("nan"),
                "policy_status": "skip",
                "policy_reason": reason,
                "fallback_used": False,
            }
        )
    return pd.DataFrame(rows, columns=list(_POLICY_COLUMNS))


def _infer_skip_reason(group: pd.DataFrame) -> str:
    if group.empty:
        return "no_data"

    probability_ok = pd.to_numeric(group["breach_probability"], errors="coerce").between(0.0, 1.0, inclusive="both")
    if not probability_ok.any():
        return "missing_probability"

    premium_ok = pd.to_numeric(group["premium_estimate"], errors="coerce").gt(0.0)
    if not premium_ok.any():
        return "no_valid_premium"

    quote_ok = group["quote_ok"] if "quote_ok" in group.columns else pd.Series([False] * len(group), index=group.index)
    if not bool(quote_ok.any()):
        return "bad_quote_quality"

    skip_ok = group["skip_ok"] if "skip_ok" in group.columns else pd.Series([False] * len(group), index=group.index)
    if not bool(skip_ok.any()):
        return "snapshot_skip_reason"
    return "no_eligible_candidates"


def _quote_quality_rank(status: object) -> int:
    text = str(status or "").strip().lower()
    if text == "good":
        return 0
    if text == "unknown":
        return 1
    return 2


def _first_present(columns: list[str] | pd.Index, *candidates: str) -> str | None:
    existing = {str(col) for col in columns}
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def _normalize_risk_tiers(risk_tiers: Iterable[float]) -> tuple[float, ...]:
    values = sorted({float(value) for value in risk_tiers})
    out: list[float] = []
    for value in values:
        if not math.isfinite(value) or value <= 0.0 or value >= 1.0:
            raise ValueError("risk_tiers must contain finite values in (0, 1)")
        out.append(value)
    return tuple(out)


__all__ = [
    "ZeroDTEPolicyConfig",
    "recommend_zero_dte_put_strikes",
]
