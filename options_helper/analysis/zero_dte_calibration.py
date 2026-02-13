from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ZeroDTECalibrationConfig:
    probability_col: str = "breach_probability"
    outcome_col: str = "breach_observed"
    num_bins: int = 10

    def __post_init__(self) -> None:
        if self.num_bins < 2:
            raise ValueError("num_bins must be >= 2")


@dataclass(frozen=True)
class ZeroDTECalibrationResult:
    sample_size: int
    brier_score: float
    observed_rate: float
    predicted_mean: float
    sharpness: float
    expected_calibration_error: float
    reliability_bins: pd.DataFrame
    observed_vs_predicted: pd.DataFrame


def compute_zero_dte_calibration(
    predictions: pd.DataFrame,
    outcomes: pd.Series | pd.DataFrame | None = None,
    *,
    config: ZeroDTECalibrationConfig | None = None,
) -> ZeroDTECalibrationResult:
    cfg = config or ZeroDTECalibrationConfig()
    cleaned = _prepare_probability_outcome_frame(predictions, outcomes=outcomes, config=cfg)
    if cleaned.empty:
        empty = pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "sample_size",
                "predicted_mean",
                "observed_rate",
                "abs_gap",
                "bin_brier_score",
            ]
        )
        return ZeroDTECalibrationResult(
            sample_size=0,
            brier_score=float("nan"),
            observed_rate=float("nan"),
            predicted_mean=float("nan"),
            sharpness=float("nan"),
            expected_calibration_error=float("nan"),
            reliability_bins=empty,
            observed_vs_predicted=empty,
        )

    probabilities = cleaned["probability"]
    observed = cleaned["observed"]
    brier = float(((probabilities - observed) ** 2).mean())
    observed_rate = float(observed.mean())
    predicted_mean = float(probabilities.mean())
    sharpness = float(((probabilities - 0.5) ** 2).mean())

    reliability = _build_reliability_bins(cleaned, num_bins=cfg.num_bins)
    weighted_gap = (reliability["sample_size"] * reliability["abs_gap"]).sum()
    ece = float(weighted_gap / max(float(len(cleaned)), 1.0))

    return ZeroDTECalibrationResult(
        sample_size=int(len(cleaned)),
        brier_score=brier,
        observed_rate=observed_rate,
        predicted_mean=predicted_mean,
        sharpness=sharpness,
        expected_calibration_error=ece,
        reliability_bins=reliability,
        observed_vs_predicted=reliability.copy(),
    )


def _prepare_probability_outcome_frame(
    predictions: pd.DataFrame,
    *,
    outcomes: pd.Series | pd.DataFrame | None,
    config: ZeroDTECalibrationConfig,
) -> pd.DataFrame:
    frame = predictions.copy() if predictions is not None else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["probability", "observed"])
    frame.columns = [str(col) for col in frame.columns]
    if config.probability_col not in frame.columns:
        raise ValueError(f"predictions must include '{config.probability_col}'")

    probability = pd.to_numeric(frame[config.probability_col], errors="coerce")
    observed: pd.Series
    if outcomes is not None:
        observed = _coerce_outcomes(outcomes, outcome_col=config.outcome_col, index=frame.index)
    else:
        if config.outcome_col not in frame.columns:
            raise ValueError(
                f"Either outcomes must be provided or predictions must include '{config.outcome_col}'"
            )
        observed = _coerce_outcome_series(frame[config.outcome_col])

    out = pd.DataFrame(
        {
            "probability": probability.astype("float64"),
            "observed": observed.astype("float64"),
        },
        index=frame.index,
    )
    valid = out["probability"].between(0.0, 1.0, inclusive="both") & out["observed"].isin([0.0, 1.0])
    out = out.loc[valid].copy()
    return out.reset_index(drop=True)


def _coerce_outcomes(
    outcomes: pd.Series | pd.DataFrame,
    *,
    outcome_col: str,
    index: pd.Index,
) -> pd.Series:
    if isinstance(outcomes, pd.DataFrame):
        columns = [str(col) for col in outcomes.columns]
        if outcome_col in columns:
            series = outcomes[outcome_col]
        elif len(columns) == 1:
            series = outcomes.iloc[:, 0]
        else:
            raise ValueError("outcomes DataFrame must contain the configured outcome column")
    else:
        series = outcomes

    aligned = series.reindex(index) if isinstance(series, pd.Series) else pd.Series(series, index=index)
    return _coerce_outcome_series(aligned)


def _coerce_outcome_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    bool_like = series.astype(str).str.strip().str.lower()
    numeric = numeric.where(numeric.notna(), bool_like.map({"true": 1.0, "false": 0.0}))
    numeric = numeric.where(numeric.isin([0.0, 1.0]))
    return numeric


def _build_reliability_bins(frame: pd.DataFrame, *, num_bins: int) -> pd.DataFrame:
    out = frame.copy()
    edges = [idx / float(num_bins) for idx in range(num_bins + 1)]
    out["bin_index"] = pd.cut(
        out["probability"],
        bins=edges,
        labels=False,
        include_lowest=True,
        right=True,
    )
    out["bin_index"] = out["bin_index"].fillna(0).astype(int)
    grouped = out.groupby("bin_index", sort=True)

    rows: list[dict[str, object]] = []
    for bin_index, sub in grouped:
        lower = float(edges[int(bin_index)])
        upper = float(edges[min(int(bin_index) + 1, num_bins)])
        predicted_mean = float(sub["probability"].mean())
        observed_rate = float(sub["observed"].mean())
        abs_gap = abs(predicted_mean - observed_rate)
        rows.append(
            {
                "bin_index": int(bin_index),
                "bin_lower": lower,
                "bin_upper": upper,
                "sample_size": int(len(sub)),
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
                "abs_gap": abs_gap,
                "bin_brier_score": float(((sub["probability"] - sub["observed"]) ** 2).mean()),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "bin_index",
                "bin_lower",
                "bin_upper",
                "sample_size",
                "predicted_mean",
                "observed_rate",
                "abs_gap",
                "bin_brier_score",
            ]
        )

    reliability = pd.DataFrame(rows)
    reliability = reliability.sort_values(by="bin_index", kind="mergesort").reset_index(drop=True)
    return reliability


__all__ = [
    "ZeroDTECalibrationConfig",
    "ZeroDTECalibrationResult",
    "compute_zero_dte_calibration",
]
