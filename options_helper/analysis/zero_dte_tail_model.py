from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


_PREDICTION_COLUMNS: tuple[str, ...] = (
    "session_date",
    "decision_ts",
    "time_of_day_bucket",
    "extension_bucket",
    "iv_regime",
    "strike_return",
    "breach_probability",
    "breach_probability_ci_low",
    "breach_probability_ci_high",
    "sample_size",
    "parent_sample_size",
    "effective_sample_size",
    "breach_count",
    "low_sample_bin",
    "fallback_source",
    "monotonic_adjusted",
)


@dataclass(frozen=True)
class ZeroDTETailModelConfig:
    time_bucket_col: str = "time_of_day_bucket"
    extension_col: str = "intraday_return"
    iv_regime_col: str = "iv_regime"
    label_return_col: str = "close_return_from_entry"
    extension_bin_edges: tuple[float, ...] = (-0.03, -0.02, -0.01, 0.0, 0.01)
    min_bucket_samples: int = 20
    parent_prior_strength: float = 20.0
    local_prior_strength: float = 25.0
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        if self.min_bucket_samples < 1:
            raise ValueError("min_bucket_samples must be >= 1")
        if self.parent_prior_strength <= 0.0:
            raise ValueError("parent_prior_strength must be > 0")
        if self.local_prior_strength <= 0.0:
            raise ValueError("local_prior_strength must be > 0")
        if self.confidence_level <= 0.0 or self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        edges = tuple(float(value) for value in self.extension_bin_edges)
        if len(edges) == 0:
            raise ValueError("extension_bin_edges must not be empty")
        if sorted(edges) != list(edges):
            raise ValueError("extension_bin_edges must be sorted ascending")


@dataclass(frozen=True)
class ZeroDTETailModel:
    config: ZeroDTETailModelConfig
    strike_returns: tuple[float, ...]
    training_sample_size: int
    global_stats: pd.DataFrame
    parent_stats: pd.DataFrame
    bucket_stats: pd.DataFrame


def fit_zero_dte_tail_model(
    training_rows: pd.DataFrame,
    *,
    strike_returns: Iterable[float],
    config: ZeroDTETailModelConfig | None = None,
) -> ZeroDTETailModel:
    cfg = config or ZeroDTETailModelConfig()
    strikes = _normalize_strike_returns(strike_returns)
    prepared = _prepare_training_rows(training_rows, config=cfg)

    global_stats = _build_global_stats(prepared, strikes, label_col=cfg.label_return_col)
    parent_stats = _build_parent_stats(
        prepared,
        strikes,
        global_stats=global_stats,
        label_col=cfg.label_return_col,
        config=cfg,
    )
    bucket_stats = _build_bucket_stats(prepared, strikes, label_col=cfg.label_return_col)

    return ZeroDTETailModel(
        config=cfg,
        strike_returns=strikes,
        training_sample_size=int(len(prepared)),
        global_stats=global_stats,
        parent_stats=parent_stats,
        bucket_stats=bucket_stats,
    )


def score_zero_dte_tail_model(
    model: ZeroDTETailModel,
    state_rows: pd.DataFrame,
    *,
    strike_returns: Iterable[float] | None = None,
) -> pd.DataFrame:
    state = state_rows.copy() if state_rows is not None else pd.DataFrame()
    if state.empty:
        return pd.DataFrame(columns=list(_PREDICTION_COLUMNS))

    cfg = model.config
    strikes = _normalize_strike_returns(strike_returns or model.strike_returns)
    global_lookup = _stats_lookup(
        model.global_stats,
        key_cols=("strike_return",),
    )
    parent_lookup = _stats_lookup(
        model.parent_stats,
        key_cols=("time_of_day_bucket", "iv_regime", "strike_return"),
    )
    bucket_lookup = _stats_lookup(
        model.bucket_stats,
        key_cols=("time_of_day_bucket", "extension_bucket", "iv_regime", "strike_return"),
    )

    rows: list[dict[str, object]] = []
    for row_idx, row in state.reset_index(drop=True).iterrows():
        time_bucket = _normalize_bucket_value(row.get(cfg.time_bucket_col), default="unknown_time")
        extension_bucket = _bucketize_extension(
            row.get(cfg.extension_col),
            edges=cfg.extension_bin_edges,
        )
        iv_regime = _normalize_bucket_value(row.get(cfg.iv_regime_col), default="unknown_regime")

        for strike in strikes:
            global_key = (strike,)
            parent_key = (time_bucket, iv_regime, strike)
            local_key = (time_bucket, extension_bucket, iv_regime, strike)

            global_item = global_lookup.get(global_key)
            parent_item = parent_lookup.get(parent_key)
            local_item = bucket_lookup.get(local_key)

            global_prob = float(global_item["smoothed_probability"]) if global_item else 0.5
            parent_prob = float(parent_item["smoothed_probability"]) if parent_item else global_prob

            local_n = int(local_item["sample_size"]) if local_item else 0
            local_k = int(local_item["breach_count"]) if local_item else 0
            parent_n = int(parent_item["sample_size"]) if parent_item else 0
            prior_prob = parent_prob if parent_n > 0 else global_prob
            posterior = _posterior_mean(
                count=local_k,
                sample_size=local_n,
                prior_probability=prior_prob,
                prior_strength=cfg.local_prior_strength,
            )
            k_eff = float(local_k) + prior_prob * float(cfg.local_prior_strength)
            n_eff = float(local_n) + float(cfg.local_prior_strength)
            ci_low, ci_high = _wilson_interval(
                count=k_eff,
                sample_size=n_eff,
                confidence_level=cfg.confidence_level,
            )
            fallback_source = _fallback_source(
                local_sample_size=local_n,
                parent_sample_size=parent_n,
                min_bucket_samples=cfg.min_bucket_samples,
            )

            rows.append(
                {
                    "__row_id": row_idx,
                    "session_date": row.get("session_date"),
                    "decision_ts": pd.to_datetime(row.get("decision_ts"), errors="coerce", utc=True),
                    "time_of_day_bucket": time_bucket,
                    "extension_bucket": extension_bucket,
                    "iv_regime": iv_regime,
                    "strike_return": strike,
                    "breach_probability": posterior,
                    "breach_probability_ci_low": ci_low,
                    "breach_probability_ci_high": ci_high,
                    "sample_size": local_n,
                    "parent_sample_size": parent_n,
                    "effective_sample_size": n_eff,
                    "breach_count": local_k,
                    "low_sample_bin": local_n < cfg.min_bucket_samples,
                    "fallback_source": fallback_source,
                    "monotonic_adjusted": False,
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=list(_PREDICTION_COLUMNS))

    adjusted_parts: list[pd.DataFrame] = []
    for _, group in out.groupby("__row_id", sort=False):
        adjusted_parts.append(_enforce_monotonic_strike_probabilities(group))
    adjusted = pd.concat(adjusted_parts, ignore_index=True)
    adjusted = adjusted.drop(columns=["__row_id"])
    adjusted = adjusted.sort_values(
        by=["decision_ts", "strike_return"],
        ascending=[True, True],
        kind="mergesort",
    )
    return adjusted.reset_index(drop=True).loc[:, list(_PREDICTION_COLUMNS)]


def _prepare_training_rows(
    training_rows: pd.DataFrame,
    *,
    config: ZeroDTETailModelConfig,
) -> pd.DataFrame:
    frame = training_rows.copy() if training_rows is not None else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                config.time_bucket_col,
                config.iv_regime_col,
                "extension_bucket",
                config.label_return_col,
            ]
        )

    if "feature_status" in frame.columns:
        status = frame["feature_status"].astype(str).str.lower()
        frame = frame.loc[status == "ok"].copy()
    if "label_status" in frame.columns:
        label_status = frame["label_status"].astype(str).str.lower()
        frame = frame.loc[label_status == "ok"].copy()

    if config.label_return_col not in frame.columns:
        raise ValueError(f"training_rows must include '{config.label_return_col}'")
    label_values = pd.to_numeric(frame[config.label_return_col], errors="coerce")
    frame = frame.loc[label_values.notna()].copy()
    frame[config.label_return_col] = label_values.loc[frame.index]

    if config.time_bucket_col not in frame.columns:
        frame[config.time_bucket_col] = "unknown_time"
    if config.iv_regime_col not in frame.columns:
        frame[config.iv_regime_col] = "unknown_regime"
    if config.extension_col not in frame.columns:
        frame[config.extension_col] = float("nan")

    frame[config.time_bucket_col] = frame[config.time_bucket_col].map(
        lambda value: _normalize_bucket_value(value, default="unknown_time")
    )
    frame[config.iv_regime_col] = frame[config.iv_regime_col].map(
        lambda value: _normalize_bucket_value(value, default="unknown_regime")
    )
    frame["extension_bucket"] = frame[config.extension_col].map(
        lambda value: _bucketize_extension(value, edges=config.extension_bin_edges)
    )
    return frame.reset_index(drop=True)


def _build_global_stats(
    frame: pd.DataFrame,
    strikes: tuple[float, ...],
    *,
    label_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    values = pd.to_numeric(frame.get(label_col), errors="coerce")
    total_n = int(values.notna().sum())
    clean = values.dropna().astype("float64")
    for strike in strikes:
        breaches = int((clean <= float(strike)).sum()) if total_n else 0
        probability = _posterior_mean(
            count=breaches,
            sample_size=total_n,
            prior_probability=0.5,
            prior_strength=1.0,
        )
        rows.append(
            {
                "strike_return": strike,
                "sample_size": total_n,
                "breach_count": breaches,
                "smoothed_probability": probability,
            }
        )
    return pd.DataFrame(rows)


def _build_parent_stats(
    frame: pd.DataFrame,
    strikes: tuple[float, ...],
    *,
    global_stats: pd.DataFrame,
    label_col: str,
    config: ZeroDTETailModelConfig,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "time_of_day_bucket",
                "iv_regime",
                "strike_return",
                "sample_size",
                "breach_count",
                "smoothed_probability",
            ]
        )

    global_prob_lookup = _stats_lookup(global_stats, key_cols=("strike_return",))
    out_rows: list[dict[str, object]] = []
    grouped = frame.groupby([config.time_bucket_col, config.iv_regime_col], sort=True)
    for (time_bucket, iv_regime), sub in grouped:
        returns = pd.to_numeric(sub[label_col], errors="coerce").dropna().astype("float64")
        n = int(len(returns))
        for strike in strikes:
            k = int((returns <= float(strike)).sum()) if n else 0
            global_prob = float(global_prob_lookup[(strike,)]["smoothed_probability"])
            probability = _posterior_mean(
                count=k,
                sample_size=n,
                prior_probability=global_prob,
                prior_strength=config.parent_prior_strength,
            )
            out_rows.append(
                {
                    "time_of_day_bucket": str(time_bucket),
                    "iv_regime": str(iv_regime),
                    "strike_return": strike,
                    "sample_size": n,
                    "breach_count": k,
                    "smoothed_probability": probability,
                }
            )
    return pd.DataFrame(out_rows)


def _build_bucket_stats(
    frame: pd.DataFrame,
    strikes: tuple[float, ...],
    *,
    label_col: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "time_of_day_bucket",
                "extension_bucket",
                "iv_regime",
                "strike_return",
                "sample_size",
                "breach_count",
                "raw_probability",
            ]
        )

    rows: list[dict[str, object]] = []
    grouped = frame.groupby(["time_of_day_bucket", "extension_bucket", "iv_regime"], sort=True)
    for (time_bucket, extension_bucket, iv_regime), sub in grouped:
        returns = pd.to_numeric(sub[label_col], errors="coerce").dropna().astype("float64")
        n = int(len(returns))
        for strike in strikes:
            k = int((returns <= float(strike)).sum()) if n else 0
            raw_probability = float(k / n) if n > 0 else float("nan")
            rows.append(
                {
                    "time_of_day_bucket": str(time_bucket),
                    "extension_bucket": str(extension_bucket),
                    "iv_regime": str(iv_regime),
                    "strike_return": strike,
                    "sample_size": n,
                    "breach_count": k,
                    "raw_probability": raw_probability,
                }
            )
    return pd.DataFrame(rows)


def _enforce_monotonic_strike_probabilities(group: pd.DataFrame) -> pd.DataFrame:
    sorted_group = group.sort_values(by="strike_return", ascending=True, kind="mergesort").copy()
    prob = pd.to_numeric(sorted_group["breach_probability"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(sorted_group["breach_probability_ci_low"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(sorted_group["breach_probability_ci_high"], errors="coerce").to_numpy(dtype="float64")

    adjusted_prob = np.maximum.accumulate(prob)
    adjusted_low = np.maximum.accumulate(low)
    adjusted_high = np.maximum.accumulate(high)
    adjusted_low = np.minimum(adjusted_low, adjusted_prob)
    adjusted_high = np.maximum(adjusted_high, adjusted_prob)
    adjusted_low = np.clip(adjusted_low, 0.0, 1.0)
    adjusted_high = np.clip(adjusted_high, 0.0, 1.0)
    adjusted_prob = np.clip(adjusted_prob, 0.0, 1.0)

    changed = np.logical_or.reduce(
        (
            ~np.isclose(prob, adjusted_prob, equal_nan=True),
            ~np.isclose(low, adjusted_low, equal_nan=True),
            ~np.isclose(high, adjusted_high, equal_nan=True),
        )
    )

    sorted_group["breach_probability"] = adjusted_prob
    sorted_group["breach_probability_ci_low"] = adjusted_low
    sorted_group["breach_probability_ci_high"] = adjusted_high
    sorted_group["monotonic_adjusted"] = bool(changed.any())
    return sorted_group


def _posterior_mean(
    *,
    count: int,
    sample_size: int,
    prior_probability: float,
    prior_strength: float,
) -> float:
    prior_p = _clamp_probability(prior_probability)
    prior_strength_f = float(max(prior_strength, 0.0))
    numerator = float(count) + prior_p * prior_strength_f
    denominator = float(sample_size) + prior_strength_f
    if denominator <= 0.0:
        return prior_p
    return _clamp_probability(numerator / denominator)


def _wilson_interval(
    *,
    count: float,
    sample_size: float,
    confidence_level: float,
) -> tuple[float, float]:
    n = float(sample_size)
    if not math.isfinite(n) or n <= 0.0:
        return 0.0, 1.0

    p_hat = _clamp_probability(float(count) / n)
    z = NormalDist().inv_cdf(0.5 + (float(confidence_level) / 2.0))
    denom = 1.0 + ((z * z) / n)
    center = (p_hat + ((z * z) / (2.0 * n))) / denom
    margin_num = p_hat * (1.0 - p_hat) + ((z * z) / (4.0 * n))
    margin = (z * math.sqrt(max(margin_num, 0.0) / n)) / denom
    low = _clamp_probability(center - margin)
    high = _clamp_probability(center + margin)
    return low, high


def _bucketize_extension(value: object, *, edges: tuple[float, ...]) -> str:
    numeric = float(pd.to_numeric(value, errors="coerce"))
    if not math.isfinite(numeric):
        return "unknown_ext"
    cut_edges = tuple(float(edge) for edge in edges)
    index = int(np.searchsorted(cut_edges, numeric, side="right"))
    if index <= 0:
        return f"le_{cut_edges[0]:+.4f}"
    if index >= len(cut_edges):
        return f"gt_{cut_edges[-1]:+.4f}"
    left = cut_edges[index - 1]
    right = cut_edges[index]
    return f"({left:+.4f},{right:+.4f}]"


def _normalize_bucket_value(value: object, *, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text else default


def _normalize_strike_returns(strike_returns: Iterable[float]) -> tuple[float, ...]:
    strikes = sorted({float(value) for value in strike_returns})
    if not strikes:
        raise ValueError("strike_returns must not be empty")
    return tuple(strikes)


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(min(max(value, 0.0), 1.0))


def _stats_lookup(frame: pd.DataFrame, *, key_cols: tuple[str, ...]) -> dict[tuple[object, ...], dict[str, object]]:
    if frame is None or frame.empty:
        return {}
    out: dict[tuple[object, ...], dict[str, object]] = {}
    for _, row in frame.iterrows():
        key = tuple(row[col] for col in key_cols)
        out[key] = row.to_dict()
    return out


def _fallback_source(*, local_sample_size: int, parent_sample_size: int, min_bucket_samples: int) -> str:
    if local_sample_size <= 0:
        return "parent" if parent_sample_size > 0 else "global"
    if local_sample_size < min_bucket_samples:
        return "shrunk_local"
    return "local"


__all__ = [
    "ZeroDTETailModel",
    "ZeroDTETailModelConfig",
    "fit_zero_dte_tail_model",
    "score_zero_dte_tail_model",
]
