from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Literal


SignedExposureConvention = Literal["calls_positive_puts_negative"]

SIGNED_EXPOSURE_CONVENTION: Final[SignedExposureConvention] = "calls_positive_puts_negative"

IV_SURFACE_TENOR_TARGETS_DTE: Final[tuple[int, ...]] = (7, 14, 30, 60, 90, 180)


DeltaBucketName = Literal["d00_20", "d20_40", "d40_60", "d60_80", "d80_100"]

DELTA_BUCKET_ORDER: Final[tuple[DeltaBucketName, ...]] = (
    "d00_20",
    "d20_40",
    "d40_60",
    "d60_80",
    "d80_100",
)


@dataclass(frozen=True)
class DeltaBucketSpec:
    name: DeltaBucketName
    min_abs_delta: float
    max_abs_delta: float
    max_inclusive: bool = False


DELTA_BUCKET_SPECS: Final[tuple[DeltaBucketSpec, ...]] = (
    DeltaBucketSpec(name="d00_20", min_abs_delta=0.0, max_abs_delta=0.2),
    DeltaBucketSpec(name="d20_40", min_abs_delta=0.2, max_abs_delta=0.4),
    DeltaBucketSpec(name="d40_60", min_abs_delta=0.4, max_abs_delta=0.6),
    DeltaBucketSpec(name="d60_80", min_abs_delta=0.6, max_abs_delta=0.8),
    DeltaBucketSpec(name="d80_100", min_abs_delta=0.8, max_abs_delta=1.0, max_inclusive=True),
)


SNAPSHOT_DERIVED_ARTIFACTS: Final[tuple[str, ...]] = ("iv_surface", "exposure")
INTRADAY_DERIVED_ARTIFACTS: Final[tuple[str, ...]] = ("intraday_flow",)
CANDLE_DERIVED_ARTIFACTS: Final[tuple[str, ...]] = ("levels",)
POSITION_DERIVED_ARTIFACTS: Final[tuple[str, ...]] = ("scenarios",)


IV_SURFACE_TENOR_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "tenor_target_dte",
    "expiry",
    "dte",
    "tenor_gap_dte",
    "atm_strike",
    "atm_iv",
    "atm_mark",
    "straddle_mark",
    "expected_move_pct",
    "skew_25d_pp",
    "skew_10d_pp",
    "contracts_used",
    "warnings",
)

IV_SURFACE_DELTA_BUCKET_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "tenor_target_dte",
    "expiry",
    "option_type",
    "delta_bucket",
    "avg_iv",
    "median_iv",
    "n_contracts",
    "warnings",
)

EXPOSURE_STRIKE_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "expiry",
    "strike",
    "call_oi",
    "put_oi",
    "call_gex",
    "put_gex",
    "net_gex",
)

EXPOSURE_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "spot",
    "flip_strike",
    "total_call_gex",
    "total_put_gex",
    "total_net_gex",
    "warnings",
)

INTRADAY_FLOW_CONTRACT_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "market_date",
    "source",
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "delta_bucket",
    "buy_volume",
    "sell_volume",
    "unknown_volume",
    "buy_notional",
    "sell_notional",
    "net_notional",
    "trade_count",
    "unknown_trade_share",
    "quote_coverage_pct",
    "warnings",
)

INTRADAY_FLOW_TIME_BUCKET_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "market_date",
    "bucket_start_utc",
    "bucket_minutes",
    "contract_symbol",
    "expiry",
    "option_type",
    "strike",
    "delta_bucket",
    "buy_notional",
    "sell_notional",
    "net_notional",
    "trade_count",
    "unknown_trade_share",
)

LEVELS_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "spot",
    "prev_close",
    "session_open",
    "gap_pct",
    "prior_high",
    "prior_low",
    "rolling_high",
    "rolling_low",
    "rs_ratio",
    "beta_20d",
    "corr_20d",
    "warnings",
)

LEVELS_ANCHORED_VWAP_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "anchor_id",
    "anchor_type",
    "anchor_ts_utc",
    "anchor_price",
    "anchored_vwap",
    "distance_from_spot_pct",
)

LEVELS_VOLUME_PROFILE_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "price_bin_low",
    "price_bin_high",
    "volume",
    "volume_pct",
    "is_poc",
    "is_hvn",
    "is_lvn",
)

SCENARIO_SUMMARY_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "contract_symbol",
    "option_type",
    "side",
    "contracts",
    "spot",
    "strike",
    "expiry",
    "mark",
    "iv",
    "intrinsic",
    "extrinsic",
    "theta_burn_dollars_day",
    "theta_burn_pct_premium_day",
    "warnings",
)

SCENARIO_GRID_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "as_of",
    "contract_symbol",
    "spot_change_pct",
    "iv_change_pp",
    "days_forward",
    "scenario_spot",
    "scenario_iv",
    "days_to_expiry",
    "theoretical_price",
    "pnl_per_contract",
    "pnl_position",
)


def classify_abs_delta_bucket(abs_delta: float | None) -> DeltaBucketName | None:
    value = _coerce_number(abs_delta)
    if value is None:
        return None
    if value < 0.0 or value > 1.0:
        return None
    for spec in DELTA_BUCKET_SPECS:
        lower_ok = value >= spec.min_abs_delta
        upper_ok = value <= spec.max_abs_delta if spec.max_inclusive else value < spec.max_abs_delta
        if lower_ok and upper_ok:
            return spec.name
    return None


def classify_delta_bucket(delta: float | None) -> DeltaBucketName | None:
    value = _coerce_number(delta)
    if value is None:
        return None
    return classify_abs_delta_bucket(abs(value))


def _coerce_number(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


__all__ = [
    "CANDLE_DERIVED_ARTIFACTS",
    "DELTA_BUCKET_ORDER",
    "DELTA_BUCKET_SPECS",
    "DeltaBucketName",
    "DeltaBucketSpec",
    "EXPOSURE_STRIKE_FIELDS",
    "EXPOSURE_SUMMARY_FIELDS",
    "INTRADAY_DERIVED_ARTIFACTS",
    "INTRADAY_FLOW_CONTRACT_FIELDS",
    "INTRADAY_FLOW_TIME_BUCKET_FIELDS",
    "IV_SURFACE_DELTA_BUCKET_FIELDS",
    "IV_SURFACE_TENOR_FIELDS",
    "IV_SURFACE_TENOR_TARGETS_DTE",
    "LEVELS_ANCHORED_VWAP_FIELDS",
    "LEVELS_SUMMARY_FIELDS",
    "LEVELS_VOLUME_PROFILE_FIELDS",
    "POSITION_DERIVED_ARTIFACTS",
    "SCENARIO_GRID_FIELDS",
    "SCENARIO_SUMMARY_FIELDS",
    "SIGNED_EXPOSURE_CONVENTION",
    "SNAPSHOT_DERIVED_ARTIFACTS",
    "SignedExposureConvention",
    "classify_abs_delta_bucket",
    "classify_delta_bucket",
]
