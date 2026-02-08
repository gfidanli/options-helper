from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from options_helper.analysis.strategy_modeling_contracts import parse_strategy_trade_simulations
from options_helper.schemas.strategy_modeling_contracts import SegmentDimension, StrategyTradeSimulation

_EPSILON = 1e-12

_SEGMENT_DIMENSIONS: tuple[SegmentDimension, ...] = (
    "symbol",
    "direction",
    "extension_bucket",
    "rsi_regime",
    "rsi_divergence",
    "volatility_regime",
    "bars_since_swing_bucket",
)


@dataclass(frozen=True)
class StrategySegmentationConfig:
    min_sample_threshold: int = 20
    include_confidence_intervals: bool = True
    confidence_z_score: float = 1.96
    unknown_segment_value: str = "unknown"

    def __post_init__(self) -> None:
        if int(self.min_sample_threshold) < 1:
            raise ValueError("min_sample_threshold must be >= 1")
        if float(self.confidence_z_score) <= 0.0 or not math.isfinite(float(self.confidence_z_score)):
            raise ValueError("confidence_z_score must be finite and > 0")
        if not str(self.unknown_segment_value).strip():
            raise ValueError("unknown_segment_value must be non-empty")


@dataclass(frozen=True)
class StrategySegmentPerformance:
    segment_dimension: str
    segment_value: str
    trade_count: int
    sample_size: int
    min_sample_threshold: int
    is_reliable: bool
    win_rate: float | None
    avg_realized_r: float | None
    expectancy_r: float | None
    profit_factor: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    confidence_interval_label: str | None = None


@dataclass(frozen=True)
class StrategySegmentReconciliation:
    segment_dimension: SegmentDimension
    base_trade_count: int
    slice_trade_count: int
    is_reconciled: bool


@dataclass(frozen=True)
class StrategySegmentationResult:
    base_trade_count: int
    overall: StrategySegmentPerformance
    segments: tuple[StrategySegmentPerformance, ...]
    reconciliation: tuple[StrategySegmentReconciliation, ...]

    @property
    def all_dimensions_reconciled(self) -> bool:
        return all(item.is_reconciled for item in self.reconciliation)


def aggregate_strategy_segmentation(
    trades: Iterable[Mapping[str, Any] | StrategyTradeSimulation],
    *,
    segment_context: Iterable[Mapping[str, Any]] | None = None,
    config: StrategySegmentationConfig | None = None,
) -> StrategySegmentationResult:
    cfg = config or StrategySegmentationConfig()
    parsed_trades = parse_strategy_trade_simulations(trades)
    base_trades = [trade for trade in parsed_trades if _is_closed_trade(trade)]
    sorted_trades = sorted(base_trades, key=_trade_sort_key)
    lookup = _build_segment_lookup(segment_context or (), unknown_value=cfg.unknown_segment_value)

    grouped: dict[SegmentDimension, dict[str, list[StrategyTradeSimulation]]] = {
        dimension: {} for dimension in _SEGMENT_DIMENSIONS
    }
    for trade in sorted_trades:
        context = _resolve_trade_context(
            trade=trade,
            lookup=lookup,
            unknown_value=cfg.unknown_segment_value,
        )
        for dimension in _SEGMENT_DIMENSIONS:
            value = context[dimension]
            grouped[dimension].setdefault(value, []).append(trade)

    overall = _compute_segment_performance(
        segment_dimension="overall",
        segment_value="all",
        trades=sorted_trades,
        config=cfg,
    )

    segment_rows: list[StrategySegmentPerformance] = []
    reconciliation: list[StrategySegmentReconciliation] = []
    base_count = len(sorted_trades)

    for dimension in _SEGMENT_DIMENSIONS:
        slices = grouped[dimension]
        slice_trade_count = 0
        for value in sorted(slices):
            trade_rows = slices[value]
            summary = _compute_segment_performance(
                segment_dimension=dimension,
                segment_value=value,
                trades=trade_rows,
                config=cfg,
            )
            segment_rows.append(summary)
            slice_trade_count += summary.sample_size
        reconciliation.append(
            StrategySegmentReconciliation(
                segment_dimension=dimension,
                base_trade_count=base_count,
                slice_trade_count=slice_trade_count,
                is_reconciled=slice_trade_count == base_count,
            )
        )

    return StrategySegmentationResult(
        base_trade_count=base_count,
        overall=overall,
        segments=tuple(segment_rows),
        reconciliation=tuple(reconciliation),
    )


def required_strategy_segment_dimensions() -> tuple[SegmentDimension, ...]:
    return _SEGMENT_DIMENSIONS


def _compute_segment_performance(
    *,
    segment_dimension: str,
    segment_value: str,
    trades: list[StrategyTradeSimulation],
    config: StrategySegmentationConfig,
) -> StrategySegmentPerformance:
    sample_size = len(trades)
    realized_values = [value for value in (_finite_float(trade.realized_r) for trade in trades) if value is not None]

    winners = sum(1 for value in realized_values if value > 0.0)
    win_rate = (winners / sample_size) if sample_size > 0 else None
    avg_realized_r = mean(realized_values) if realized_values else None
    expectancy_r = avg_realized_r

    gross_profit = sum(value for value in realized_values if value > 0.0)
    gross_loss = abs(sum(value for value in realized_values if value < 0.0))
    profit_factor: float | None = None
    if realized_values and gross_loss > _EPSILON:
        profit_factor = gross_profit / gross_loss

    sharpe_ratio = _sharpe_from_values(realized_values)
    max_drawdown_pct = _max_drawdown_pct(realized_values)

    ci_low: float | None = None
    ci_high: float | None = None
    ci_label: str | None = None
    if config.include_confidence_intervals and sample_size > 0:
        ci_low, ci_high = _wilson_interval(
            successes=winners,
            sample_size=sample_size,
            z_score=float(config.confidence_z_score),
        )
        ci_label = _confidence_interval_label(float(config.confidence_z_score))

    return StrategySegmentPerformance(
        segment_dimension=segment_dimension,
        segment_value=segment_value,
        trade_count=sample_size,
        sample_size=sample_size,
        min_sample_threshold=int(config.min_sample_threshold),
        is_reliable=sample_size >= int(config.min_sample_threshold),
        win_rate=win_rate,
        avg_realized_r=avg_realized_r,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
        confidence_interval_label=ci_label,
    )


def _build_segment_lookup(
    rows: Iterable[Mapping[str, Any]],
    *,
    unknown_value: str,
) -> dict[str, dict[str, dict[SegmentDimension, str]]]:
    by_trade_id: dict[str, dict[SegmentDimension, str]] = {}
    by_event_id: dict[str, dict[SegmentDimension, str]] = {}

    for row in rows:
        payload = dict(row)
        trade_id = _normalize_identifier(payload.get("trade_id"))
        event_id = _normalize_identifier(payload.get("event_id"))
        if trade_id is None and event_id is None:
            raise ValueError("segment_context rows must include trade_id or event_id")

        parsed = _extract_segment_values(payload, unknown_value=unknown_value)
        if trade_id is not None:
            _upsert_context(by_trade_id, key=trade_id, values=parsed)
        if event_id is not None:
            _upsert_context(by_event_id, key=event_id, values=parsed)

    return {"trade_id": by_trade_id, "event_id": by_event_id}


def _upsert_context(
    bucket: dict[str, dict[SegmentDimension, str]],
    *,
    key: str,
    values: dict[SegmentDimension, str],
) -> None:
    existing = bucket.get(key)
    if existing is None:
        bucket[key] = values
        return
    if existing != values:
        raise ValueError(f"Conflicting segment context rows for identifier: {key}")


def _extract_segment_values(
    payload: Mapping[str, Any],
    *,
    unknown_value: str,
) -> dict[SegmentDimension, str]:
    out: dict[SegmentDimension, str] = {}
    for dimension in _SEGMENT_DIMENSIONS:
        value = _value_from_payload(payload, dimension=dimension)
        if value is not None:
            out[dimension] = _normalize_segment_value(value, unknown_value=unknown_value)
    return out


def _value_from_payload(payload: Mapping[str, Any], *, dimension: SegmentDimension) -> Any:
    if dimension == "symbol":
        if "symbol" in payload:
            return payload.get("symbol")
        return payload.get("ticker")
    if dimension == "volatility_regime":
        if "volatility_regime" in payload:
            return payload.get("volatility_regime")
        return payload.get("realized_vol_regime")
    return payload.get(dimension)


def _resolve_trade_context(
    *,
    trade: StrategyTradeSimulation,
    lookup: dict[str, dict[str, dict[SegmentDimension, str]]],
    unknown_value: str,
) -> dict[SegmentDimension, str]:
    out: dict[SegmentDimension, str] = {
        "symbol": str(trade.symbol).strip().upper() or unknown_value,
        "direction": str(trade.direction).strip().lower() or unknown_value,
        "extension_bucket": unknown_value,
        "rsi_regime": unknown_value,
        "rsi_divergence": unknown_value,
        "volatility_regime": unknown_value,
        "bars_since_swing_bucket": unknown_value,
    }

    event_values = lookup["event_id"].get(str(trade.event_id))
    if event_values:
        out.update(event_values)

    trade_values = lookup["trade_id"].get(str(trade.trade_id))
    if trade_values:
        out.update(trade_values)

    out["symbol"] = str(trade.symbol).strip().upper() or unknown_value
    out["direction"] = str(trade.direction).strip().lower() or unknown_value
    return out


def _trade_sort_key(trade: StrategyTradeSimulation) -> tuple[Any, ...]:
    return (
        trade.entry_ts,
        trade.signal_confirmed_ts,
        trade.signal_ts,
        trade.symbol,
        trade.trade_id,
    )


def _is_closed_trade(trade: StrategyTradeSimulation) -> bool:
    return (
        str(trade.status) == "closed"
        and trade.reject_code is None
        and trade.exit_ts is not None
        and trade.exit_price is not None
    )


def _normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_segment_value(value: Any, *, unknown_value: str) -> str:
    if value is None:
        return unknown_value
    text = str(value).strip()
    if not text:
        return unknown_value
    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return unknown_value
    return text


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(number):
        return None
    return number


def _sharpe_from_values(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    sigma = pstdev(values)
    if sigma <= _EPSILON:
        return None
    return (mean(values) / sigma) * math.sqrt(float(len(values)))


def _max_drawdown_pct(values: list[float]) -> float | None:
    if not values:
        return None
    equity = 1.0
    peak = equity
    max_drawdown = 0.0
    for value in values:
        equity += float(value)
        peak = max(peak, equity)
        if peak > _EPSILON:
            drawdown = (equity / peak) - 1.0
            max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _wilson_interval(
    *,
    successes: int,
    sample_size: int,
    z_score: float,
) -> tuple[float, float]:
    if sample_size <= 0:
        return (0.0, 0.0)
    n = float(sample_size)
    p = max(0.0, min(1.0, float(successes) / n))
    z2 = z_score * z_score
    denominator = 1.0 + (z2 / n)
    center = (p + (z2 / (2.0 * n))) / denominator
    margin = (z_score / denominator) * math.sqrt(((p * (1.0 - p)) + (z2 / (4.0 * n))) / n)
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (low, high)


def _confidence_interval_label(z_score: float) -> str:
    confidence = math.erf(float(z_score) / math.sqrt(2.0))
    percentage = max(0.0, min(100.0, confidence * 100.0))
    return f"wilson_{percentage:.1f}%"


__all__ = [
    "StrategySegmentationConfig",
    "StrategySegmentationResult",
    "StrategySegmentPerformance",
    "StrategySegmentReconciliation",
    "aggregate_strategy_segmentation",
    "required_strategy_segment_dimensions",
]
