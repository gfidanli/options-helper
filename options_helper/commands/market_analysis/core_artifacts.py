from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from options_helper.analysis.exposure import ExposureSlice
from options_helper.analysis.iv_context import classify_iv_rv
from options_helper.analysis.tail_risk import TailRiskConfig, TailRiskResult
from options_helper.commands.market_analysis.core_helpers import _as_float, _dedupe, _to_python_datetime
from options_helper.data.confluence_config import ConfigError, load_confluence_config
from options_helper.schemas.common import clean_nan, utc_now
from options_helper.schemas.exposure import (
    ExposureArtifact,
    ExposureSliceArtifact,
    ExposureStrikeRow,
    ExposureSummaryRow,
    ExposureTopLevelRow,
)
from options_helper.schemas.iv_surface import (
    IvSurfaceArtifact,
    IvSurfaceDeltaBucketChangeRow,
    IvSurfaceDeltaBucketRow,
    IvSurfaceTenorChangeRow,
    IvSurfaceTenorRow,
)
from options_helper.schemas.levels import (
    LevelsAnchoredVwapRow,
    LevelsArtifact,
    LevelsSummaryRow,
    LevelsVolumeProfileRow,
)
from options_helper.schemas.tail_risk import (
    TailRiskArtifact,
    TailRiskConfigArtifact,
    TailRiskIVContext,
    TailRiskPercentileRow,
)


def _build_iv_context(derived_row: dict[str, Any] | None) -> TailRiskIVContext | None:
    if not derived_row:
        return None
    iv_rv = _as_float(derived_row.get("iv_rv_20d"))
    if iv_rv is None:
        return None

    low, high = _load_iv_thresholds()
    regime = classify_iv_rv(iv_rv, low=low, high=high)
    if regime is None:
        return None

    return TailRiskIVContext(
        label=regime.label,
        reason=regime.reason,
        iv_rv_20d=regime.iv_rv,
        low=regime.low,
        high=regime.high,
        atm_iv_near=_as_float(derived_row.get("atm_iv_near")),
        rv_20d=_as_float(derived_row.get("rv_20d")),
        atm_iv_near_percentile=_as_float(derived_row.get("atm_iv_near_percentile")),
        iv_term_slope=_as_float(derived_row.get("iv_term_slope")),
    )


def _load_iv_thresholds() -> tuple[float, float]:
    default_low = 0.8
    default_high = 1.2
    try:
        cfg = load_confluence_config()
    except (ConfigError, OSError, ValueError):
        return default_low, default_high

    iv_cfg = (cfg.get("iv_regime") or {}) if isinstance(cfg, dict) else {}
    try:
        low = float(iv_cfg.get("low", default_low))
        high = float(iv_cfg.get("high", default_high))
    except Exception:  # noqa: BLE001
        return default_low, default_high
    if low >= high:
        return default_low, default_high
    return low, high


def _build_tail_risk_artifact(
    *,
    symbol: str,
    config: TailRiskConfig,
    result: TailRiskResult,
    iv_context: TailRiskIVContext | None,
) -> TailRiskArtifact:
    end_rows: list[TailRiskPercentileRow] = []
    for percentile in sorted(result.end_price_percentiles):
        end_rows.append(
            TailRiskPercentileRow(
                percentile=float(percentile),
                price=_as_float(result.end_price_percentiles.get(percentile)),
                return_pct=_as_float(result.end_return_percentiles.get(percentile)),
            )
        )
    return TailRiskArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=result.as_of,
        symbol=symbol,
        disclaimer="Not financial advice.",
        config=TailRiskConfigArtifact(
            lookback_days=config.lookback_days,
            horizon_days=config.horizon_days,
            num_simulations=config.num_simulations,
            seed=config.seed,
            var_confidence=config.var_confidence,
            end_percentiles=[float(value) for value in config.end_percentiles],
            chart_percentiles=[float(value) for value in config.chart_percentiles],
            sample_paths=min(config.sample_paths, config.num_simulations),
            trading_days_per_year=config.trading_days_per_year,
        ),
        spot=result.spot,
        realized_vol_annual=result.realized_vol_annual,
        expected_return_annual=result.expected_return_annual,
        var_return=result.var_return,
        cvar_return=result.cvar_return,
        end_percentiles=end_rows,
        iv_context=iv_context,
        warnings=list(result.warnings),
    )


def _build_iv_surface_artifact(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    result: Any,
    warnings: list[str],
) -> IvSurfaceArtifact:
    return IvSurfaceArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=spot if spot > 0 else None,
        disclaimer="Not financial advice.",
        tenor=[IvSurfaceTenorRow.model_validate(row) for row in _frame_records(result.tenor)],
        delta_buckets=[IvSurfaceDeltaBucketRow.model_validate(row) for row in _frame_records(result.delta_buckets)],
        tenor_changes=[
            IvSurfaceTenorChangeRow.model_validate(row) for row in _frame_records(result.tenor_changes)
        ],
        delta_bucket_changes=[
            IvSurfaceDeltaBucketChangeRow.model_validate(row)
            for row in _frame_records(result.delta_bucket_changes)
        ],
        warnings=_dedupe([*result.warnings, *warnings]),
    )


def _build_exposure_artifact(
    *,
    symbol: str,
    as_of: date,
    spot: float,
    slices: dict[str, ExposureSlice],
    warnings: list[str],
) -> ExposureArtifact:
    slice_artifacts: list[ExposureSliceArtifact] = []
    for mode in ("near", "monthly", "all"):
        slice_data = slices[mode]
        summary = ExposureSummaryRow.model_validate(clean_nan(slice_data.summary))
        summary.warnings = _dedupe([*summary.warnings, *warnings])
        slice_artifacts.append(
            ExposureSliceArtifact(
                mode=mode,
                available_expiries=list(slice_data.available_expiries),
                included_expiries=list(slice_data.included_expiries),
                strike_rows=[
                    ExposureStrikeRow.model_validate(row) for row in _frame_records(pd.DataFrame(slice_data.strike_rows))
                ],
                summary=summary,
                top_abs_net_levels=[
                    ExposureTopLevelRow(
                        strike=float(level.strike),
                        net_gex=float(level.net_gex),
                        abs_net_gex=float(level.abs_net_gex),
                    )
                    for level in slice_data.top_abs_net_levels
                ],
            )
        )

    return ExposureArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        spot=float(spot),
        disclaimer="Not financial advice.",
        slices=slice_artifacts,
    )


def _build_levels_artifact(
    *,
    symbol: str,
    as_of: date,
    summary: Any,
    anchored: Any,
    profile: Any,
) -> LevelsArtifact:
    summary_row = LevelsSummaryRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        spot=summary.spot,
        prev_close=summary.prev_close,
        session_open=summary.session_open,
        gap_pct=summary.gap_pct,
        prior_high=summary.prior_high,
        prior_low=summary.prior_low,
        rolling_high=summary.rolling_high,
        rolling_low=summary.rolling_low,
        rs_ratio=summary.rs_ratio,
        beta_20d=summary.beta_20d,
        corr_20d=summary.corr_20d,
        warnings=list(summary.warnings),
    )

    anchor_row = LevelsAnchoredVwapRow(
        symbol=symbol,
        as_of=as_of.isoformat(),
        anchor_id="session_open",
        anchor_type="session_open",
        anchor_ts_utc=_to_python_datetime(anchored.anchor_ts_utc),
        anchor_price=anchored.anchor_price,
        anchored_vwap=anchored.anchored_vwap,
        distance_from_spot_pct=anchored.distance_from_spot_pct,
    )

    profile_rows = [
        LevelsVolumeProfileRow(
            symbol=symbol,
            as_of=as_of.isoformat(),
            price_bin_low=float(row.price_bin_low),
            price_bin_high=float(row.price_bin_high),
            volume=float(row.volume),
            volume_pct=float(row.volume_pct),
            is_poc=bool(row.is_poc),
            is_hvn=bool(row.is_hvn),
            is_lvn=bool(row.is_lvn),
        )
        for row in profile.bins
    ]

    return LevelsArtifact(
        schema_version=1,
        generated_at=utc_now(),
        as_of=as_of.isoformat(),
        symbol=symbol,
        disclaimer="Not financial advice.",
        summary=summary_row,
        anchored_vwap=[anchor_row],
        volume_profile=profile_rows,
        volume_profile_poc=profile.poc_price,
        volume_profile_hvn_candidates=[float(value) for value in profile.hvn_candidates],
        volume_profile_lvn_candidates=[float(value) for value in profile.lvn_candidates],
        warnings=_dedupe([*summary.warnings, *anchored.warnings, *profile.warnings]),
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return [clean_nan(row) for row in frame.to_dict(orient="records")]


__all__ = [
    "_build_iv_context",
    "_build_tail_risk_artifact",
    "_build_iv_surface_artifact",
    "_build_exposure_artifact",
    "_build_levels_artifact",
]
