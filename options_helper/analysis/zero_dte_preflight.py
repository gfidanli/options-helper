from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from options_helper.schemas.zero_dte_put_study import QuoteQualityStatus


@dataclass(frozen=True)
class ZeroDTEPreflightConfig:
    min_sessions: int = 20
    min_feature_rows: int = 200
    min_rows_per_time_bucket: int = 30
    min_rows_per_iv_regime: int = 20
    min_label_coverage_rate: float = 0.9
    min_quote_quality_pass_rate: float = 0.8

    def __post_init__(self) -> None:
        if self.min_sessions < 1:
            raise ValueError("min_sessions must be >= 1")
        if self.min_feature_rows < 1:
            raise ValueError("min_feature_rows must be >= 1")
        if self.min_rows_per_time_bucket < 1:
            raise ValueError("min_rows_per_time_bucket must be >= 1")
        if self.min_rows_per_iv_regime < 1:
            raise ValueError("min_rows_per_iv_regime must be >= 1")
        for value, name in (
            (self.min_label_coverage_rate, "min_label_coverage_rate"),
            (self.min_quote_quality_pass_rate, "min_quote_quality_pass_rate"),
        ):
            if value <= 0.0 or value > 1.0:
                raise ValueError(f"{name} must be in (0, 1]")


@dataclass(frozen=True)
class PreflightDiagnostic:
    code: str
    ok: bool
    message: str


@dataclass(frozen=True)
class ZeroDTEPreflightResult:
    passed: bool
    diagnostics: tuple[PreflightDiagnostic, ...]
    metrics: dict[str, float]


def run_zero_dte_preflight(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    strike_snapshot: pd.DataFrame | None = None,
    *,
    config: ZeroDTEPreflightConfig | None = None,
) -> ZeroDTEPreflightResult:
    cfg = config or ZeroDTEPreflightConfig()

    features_frame = features.copy() if features is not None else pd.DataFrame()
    labels_frame = labels.copy() if labels is not None else pd.DataFrame()
    snapshot_frame = strike_snapshot.copy() if strike_snapshot is not None else pd.DataFrame()

    diagnostics: list[PreflightDiagnostic] = []
    metrics: dict[str, float] = {}

    usable_features = _usable_feature_rows(features_frame)
    feature_rows = int(len(usable_features))
    session_count = int(usable_features["session_date"].nunique()) if "session_date" in usable_features.columns else 0
    metrics["feature_rows"] = float(feature_rows)
    metrics["session_count"] = float(session_count)

    diagnostics.append(
        _threshold_diagnostic(
            code="sessions",
            observed=session_count,
            required=cfg.min_sessions,
            noun="sessions",
        )
    )
    diagnostics.append(
        _threshold_diagnostic(
            code="feature_rows",
            observed=feature_rows,
            required=cfg.min_feature_rows,
            noun="usable feature rows",
        )
    )

    bucket_counts = _count_by_bucket(usable_features, column="time_of_day_bucket")
    if not bucket_counts:
        diagnostics.append(
            PreflightDiagnostic(
                code="time_bucket_coverage",
                ok=False,
                message="No non-null time_of_day_bucket values were found in usable features.",
            )
        )
    else:
        for bucket, count in sorted(bucket_counts.items()):
            diagnostics.append(
                _threshold_diagnostic(
                    code=f"time_bucket_{bucket}",
                    observed=count,
                    required=cfg.min_rows_per_time_bucket,
                    noun=f"rows in time bucket '{bucket}'",
                )
            )

    regime_counts = _count_by_bucket(usable_features, column="iv_regime")
    if regime_counts:
        for regime, count in sorted(regime_counts.items()):
            diagnostics.append(
                _threshold_diagnostic(
                    code=f"iv_regime_{regime}",
                    observed=count,
                    required=cfg.min_rows_per_iv_regime,
                    noun=f"rows in iv_regime '{regime}'",
                )
            )

    label_rate = _label_coverage_rate(labels_frame)
    metrics["label_coverage_rate"] = label_rate
    diagnostics.append(
        _rate_diagnostic(
            code="label_coverage_rate",
            observed=label_rate,
            required=cfg.min_label_coverage_rate,
            noun="label coverage rate",
        )
    )

    quote_rate = _quote_quality_pass_rate(snapshot_frame)
    metrics["quote_quality_pass_rate"] = quote_rate
    diagnostics.append(
        _rate_diagnostic(
            code="quote_quality_pass_rate",
            observed=quote_rate,
            required=cfg.min_quote_quality_pass_rate,
            noun="quote-quality pass rate",
        )
    )

    passed = all(diagnostic.ok for diagnostic in diagnostics)
    return ZeroDTEPreflightResult(
        passed=passed,
        diagnostics=tuple(diagnostics),
        metrics=metrics,
    )


def _usable_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    if "feature_status" not in features.columns:
        return features.copy()
    status = features["feature_status"].astype(str).str.lower()
    return features.loc[status == "ok"].copy()


def _count_by_bucket(frame: pd.DataFrame, *, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    valid = frame[column].dropna().astype(str)
    if valid.empty:
        return {}
    counts = valid.value_counts(dropna=True).to_dict()
    return {str(key): int(value) for key, value in counts.items()}


def _label_coverage_rate(labels: pd.DataFrame) -> float:
    if labels.empty:
        return 0.0
    if "close_return_from_entry" not in labels.columns:
        return 0.0
    values = pd.to_numeric(labels["close_return_from_entry"], errors="coerce")
    return float(values.notna().mean())


def _quote_quality_pass_rate(snapshot: pd.DataFrame) -> float:
    if snapshot.empty:
        return 0.0

    status_col = _first_present(snapshot.columns, "quote_quality_status")
    premium_col = _first_present(snapshot.columns, "entry_premium")
    skip_col = _first_present(snapshot.columns, "skip_reason")
    if status_col is None or premium_col is None:
        return 0.0

    status = snapshot[status_col].astype(str).str.lower()
    premium = pd.to_numeric(snapshot[premium_col], errors="coerce")

    good_statuses = {QuoteQualityStatus.GOOD.value, QuoteQualityStatus.UNKNOWN.value}
    has_no_skip = pd.Series([True] * len(snapshot), index=snapshot.index)
    if skip_col is not None:
        raw_skip = snapshot[skip_col]
        has_no_skip = raw_skip.isna() | (raw_skip.astype(str).str.strip() == "")

    passed = status.isin(good_statuses) & premium.gt(0.0) & has_no_skip
    return float(passed.mean())


def _threshold_diagnostic(*, code: str, observed: int, required: int, noun: str) -> PreflightDiagnostic:
    ok = int(observed) >= int(required)
    comparator = ">=" if ok else "<"
    message = f"Observed {observed} {noun}; required {comparator} {required}."
    return PreflightDiagnostic(code=code, ok=ok, message=message)


def _rate_diagnostic(*, code: str, observed: float, required: float, noun: str) -> PreflightDiagnostic:
    value = float(observed)
    if not math.isfinite(value):
        value = 0.0
    ok = value >= float(required)
    comparator = ">=" if ok else "<"
    message = f"Observed {noun}={value:.3f}; required {comparator} {required:.3f}."
    return PreflightDiagnostic(code=code, ok=ok, message=message)


def _first_present(columns: list[str] | pd.Index, *candidates: str) -> str | None:
    existing = set(str(col) for col in columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


__all__ = [
    "PreflightDiagnostic",
    "ZeroDTEPreflightConfig",
    "ZeroDTEPreflightResult",
    "run_zero_dte_preflight",
]
