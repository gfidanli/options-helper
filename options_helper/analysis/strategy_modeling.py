from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd

from options_helper.analysis.strategy_features import (
    StrategyFeatureConfig,
    compute_strategy_features,
    label_bars_since_swing_bucket,
    parse_strategy_feature_config,
)
from options_helper.analysis.strategy_metrics import (
    StrategyMetricsResult,
    compute_strategy_metrics,
    compute_strategy_target_hit_rates,
    serialize_strategy_metrics_result,
)
from options_helper.analysis.strategy_modeling_contracts import parse_strategy_trade_simulations
from options_helper.analysis.strategy_modeling_filters import apply_entry_filters
from options_helper.analysis.strategy_modeling_policy import parse_strategy_modeling_policy_config
from options_helper.analysis.strategy_portfolio import (
    StrategyPortfolioLedgerResult,
    StrategyPortfolioLedgerRow,
    StrategyPortfolioTargetSubset,
    build_strategy_portfolio_ledger,
    select_portfolio_trade_subset,
)
from options_helper.analysis.strategy_signals import build_strategy_signal_events
from options_helper.analysis.strategy_simulator import (
    StrategyRTarget,
    simulate_strategy_trade_paths,
)
from options_helper.data.strategy_modeling_io import (
    AdjustedDataFallbackMode,
    DailySourceMode,
    StrategyModelingDailyLoadResult,
    StrategyModelingIntradayLoadResult,
    StrategyModelingIntradayPreflightResult,
    StrategyModelingUniverseLoadResult,
    build_required_intraday_sessions,
    list_strategy_modeling_universe,
    load_daily_ohlc_history,
    load_required_intraday_bars,
    normalize_symbol,
)
from options_helper.schemas.strategy_modeling_contracts import (
    SegmentDimension,
    StrategyPortfolioMetrics,
    StrategyRLadderStat,
    StrategySegmentRecord,
    StrategySignalEvent,
    StrategyTradeSimulation,
)
from options_helper.schemas.strategy_modeling_filters import StrategyEntryFilterConfig
from options_helper.schemas.strategy_modeling_policy import StrategyModelingPolicyConfig

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


def parse_strategy_segmentation_config(
    overrides: Mapping[str, Any] | None = None,
) -> StrategySegmentationConfig:
    return StrategySegmentationConfig(**dict(overrides or {}))


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


StrategyUniverseLoader = Callable[..., StrategyModelingUniverseLoadResult]
StrategyDailyLoader = Callable[..., StrategyModelingDailyLoadResult]
StrategyRequiredSessionsBuilder = Callable[..., dict[str, list[date]]]
StrategyIntradayLoader = Callable[..., StrategyModelingIntradayLoadResult]
StrategyFeatureComputer = Callable[..., pd.DataFrame]
StrategySignalBuilder = Callable[..., list[StrategySignalEvent]]
StrategyTradeSimulator = Callable[..., list[StrategyTradeSimulation]]
StrategyPortfolioBuilder = Callable[..., StrategyPortfolioLedgerResult]
StrategyMetricsComputer = Callable[..., StrategyMetricsResult]
StrategySegmentationAggregator = Callable[..., StrategySegmentationResult]


@dataclass(frozen=True)
class StrategyModelingRequest:
    strategy: str
    symbols: Sequence[str] | None = None
    start_date: date | None = None
    end_date: date | None = None
    database_path: str | Path | None = None
    intraday_dir: Path = Path("data/intraday")
    intraday_timeframe: str = "1Min"
    daily_interval: str = "1d"
    signal_timeframe: str | None = "1d"
    policy: StrategyModelingPolicyConfig | Mapping[str, Any] | None = None
    feature_config: StrategyFeatureConfig | Mapping[str, Any] | None = None
    segmentation_config: StrategySegmentationConfig | Mapping[str, Any] | None = None
    filter_config: StrategyEntryFilterConfig | Mapping[str, Any] | None = None
    adjusted_data_fallback_mode: AdjustedDataFallbackMode = "warn_and_skip_symbol"
    signal_kwargs: Mapping[str, Any] | None = None
    starting_capital: float = 10_000.0
    max_concurrent_positions: int | None = None
    max_hold_bars: int | None = None
    target_ladder: Sequence[StrategyRTarget] | None = None
    block_on_missing_intraday_coverage: bool = True

    def __post_init__(self) -> None:
        if not str(self.strategy).strip():
            raise ValueError("strategy must be non-empty")
        if not str(self.intraday_timeframe).strip():
            raise ValueError("intraday_timeframe must be non-empty")
        if not str(self.daily_interval).strip():
            raise ValueError("daily_interval must be non-empty")
        if not math.isfinite(float(self.starting_capital)) or float(self.starting_capital) <= 0.0:
            raise ValueError("starting_capital must be > 0")


@dataclass(frozen=True)
class StrategyModelingRunResult:
    strategy: str
    policy: StrategyModelingPolicyConfig
    feature_config: StrategyFeatureConfig
    segmentation_config: StrategySegmentationConfig
    requested_symbols: tuple[str, ...]
    universe_symbols: tuple[str, ...]
    modeled_symbols: tuple[str, ...]
    skipped_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    source_by_symbol: dict[str, DailySourceMode]
    required_sessions_by_symbol: dict[str, tuple[date, ...]]
    intraday_preflight: StrategyModelingIntradayPreflightResult
    signal_events: tuple[StrategySignalEvent, ...]
    trade_simulations: tuple[StrategyTradeSimulation, ...]
    portfolio_ledger: tuple[StrategyPortfolioLedgerRow, ...]
    equity_curve: tuple[Any, ...]
    accepted_trade_ids: tuple[str, ...]
    skipped_trade_ids: tuple[str, ...]
    portfolio_metrics: StrategyPortfolioMetrics
    target_hit_rates: tuple[StrategyRLadderStat, ...]
    expectancy_dollars: float | None
    segmentation: StrategySegmentationResult
    segment_records: tuple[StrategySegmentRecord, ...]
    segment_context: tuple[dict[str, str], ...]
    filter_metadata: dict[str, Any] = field(default_factory=dict)
    filter_summary: dict[str, Any] = field(default_factory=dict)
    directional_metrics: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_intraday_blocked(self) -> bool:
        return self.intraday_preflight.is_blocked


@dataclass(frozen=True)
class StrategyModelingService:
    list_universe_loader: StrategyUniverseLoader
    daily_loader: StrategyDailyLoader
    required_sessions_builder: StrategyRequiredSessionsBuilder
    intraday_loader: StrategyIntradayLoader
    feature_computer: StrategyFeatureComputer
    signal_builder: StrategySignalBuilder
    trade_simulator: StrategyTradeSimulator
    portfolio_builder: StrategyPortfolioBuilder
    metrics_computer: StrategyMetricsComputer
    segmentation_aggregator: StrategySegmentationAggregator

    def run(self, request: StrategyModelingRequest) -> StrategyModelingRunResult:
        strategy = str(request.strategy).strip().lower()
        policy = _resolve_policy_config(request.policy)
        feature_config = _resolve_feature_config(request.feature_config)
        segmentation_config = _resolve_segmentation_config(request.segmentation_config)
        filter_config = _resolve_filter_config(request.filter_config)
        signal_kwargs = dict(request.signal_kwargs or {})

        universe = self.list_universe_loader(database_path=request.database_path)
        requested_symbols = _resolve_requested_symbols(
            request.symbols,
            universe_symbols=universe.symbols,
        )

        daily = self.daily_loader(
            requested_symbols,
            database_path=request.database_path,
            policy=policy,
            adjusted_data_fallback_mode=request.adjusted_data_fallback_mode,
            start_date=request.start_date,
            end_date=request.end_date,
            interval=request.daily_interval,
        )

        required_sessions = _normalize_required_sessions(
            self.required_sessions_builder(
                daily.candles_by_symbol,
                start_date=request.start_date,
                end_date=request.end_date,
            )
        )
        intraday = self.intraday_loader(
            required_sessions,
            intraday_dir=request.intraday_dir,
            timeframe=request.intraday_timeframe,
            require_intraday_bars=policy.require_intraday_bars,
        )

        events: list[StrategySignalEvent] = []
        segment_context: list[dict[str, str]] = []
        daily_ohlc_by_symbol: dict[str, pd.DataFrame] = {}
        daily_features_by_symbol: dict[str, pd.DataFrame] = {}

        for symbol in sorted(daily.candles_by_symbol):
            ohlc = _daily_candles_to_ohlc_frame(daily.candles_by_symbol[symbol])
            if ohlc.empty:
                continue

            features = self.feature_computer(ohlc, config=feature_config)
            daily_ohlc_by_symbol[symbol] = ohlc
            daily_features_by_symbol[symbol] = features
            symbol_events = self.signal_builder(
                strategy,
                ohlc,
                symbol=symbol,
                timeframe=request.signal_timeframe,
                **signal_kwargs,
            )
            events.extend(symbol_events)
            segment_context.extend(
                _build_event_segment_context_rows(
                    symbol=symbol,
                    events=symbol_events,
                    features=features,
                    feature_config=feature_config,
                    unknown_value=segmentation_config.unknown_segment_value,
                )
            )

        filtered_events, filter_summary, filter_metadata = apply_entry_filters(
            events,
            filter_config=filter_config,
            feature_config=feature_config,
            daily_features_by_symbol=daily_features_by_symbol,
            daily_ohlc_by_symbol=daily_ohlc_by_symbol,
            intraday_bars_by_symbol=intraday.bars_by_symbol,
        )
        sorted_events = tuple(filtered_events)
        kept_event_ids = {event.event_id for event in sorted_events}
        sorted_segment_context = tuple(
            sorted(
                (row for row in segment_context if row["event_id"] in kept_event_ids),
                key=lambda row: (row["event_id"], row["ticker"]),
            )
        )

        block_simulation = (
            bool(request.block_on_missing_intraday_coverage)
            and bool(policy.require_intraday_bars)
            and intraday.preflight.is_blocked
        )
        target_ladder = tuple(request.target_ladder) if request.target_ladder is not None else None
        if block_simulation:
            trade_simulations: list[StrategyTradeSimulation] = []
        else:
            trade_simulations = self.trade_simulator(
                sorted_events,
                intraday.bars_by_symbol,
                policy=policy,
                max_hold_bars=request.max_hold_bars,
                target_ladder=target_ladder,
            )
        sorted_trades = tuple(sorted(trade_simulations, key=_trade_sort_key))
        portfolio_subset = _select_portfolio_target_subset(
            sorted_trades,
            target_ladder=target_ladder,
        )
        subset_ids = set(portfolio_subset.trade_ids)
        portfolio_trades = tuple(trade for trade in sorted_trades if trade.trade_id in subset_ids)

        portfolio = self.portfolio_builder(
            portfolio_trades,
            starting_capital=float(request.starting_capital),
            policy=policy,
            max_concurrent_positions=request.max_concurrent_positions,
        )
        portfolio_metrics_result = self.metrics_computer(
            portfolio_trades,
            portfolio.equity_curve,
            starting_capital=float(request.starting_capital),
        )
        target_hit_rates = tuple(compute_strategy_target_hit_rates(sorted_trades))
        directional_metrics = _build_directional_metrics(
            all_trades=sorted_trades,
            portfolio_trades=portfolio_trades,
            portfolio_subset=portfolio_subset,
            portfolio_builder=self.portfolio_builder,
            metrics_computer=self.metrics_computer,
            policy=policy,
            starting_capital=float(request.starting_capital),
            max_concurrent_positions=request.max_concurrent_positions,
        )
        segmentation = self.segmentation_aggregator(
            portfolio_trades,
            segment_context=sorted_segment_context,
            config=segmentation_config,
        )
        segment_records = _segment_records_from_segmentation(segmentation)

        notes = tuple(
            [
                *list(universe.notes),
                *list(daily.notes),
                *list(intraday.notes),
            ]
        )

        return StrategyModelingRunResult(
            strategy=strategy,
            policy=policy,
            feature_config=feature_config,
            segmentation_config=segmentation_config,
            requested_symbols=tuple(requested_symbols),
            universe_symbols=tuple(universe.symbols),
            modeled_symbols=tuple(sorted(daily.candles_by_symbol)),
            skipped_symbols=tuple(daily.skipped_symbols),
            missing_symbols=tuple(daily.missing_symbols),
            source_by_symbol=dict(daily.source_by_symbol),
            required_sessions_by_symbol=required_sessions,
            intraday_preflight=intraday.preflight,
            signal_events=sorted_events,
            trade_simulations=sorted_trades,
            portfolio_ledger=tuple(portfolio.ledger),
            equity_curve=tuple(portfolio.equity_curve),
            accepted_trade_ids=tuple(portfolio.accepted_trade_ids),
            skipped_trade_ids=tuple(portfolio.skipped_trade_ids),
            portfolio_metrics=portfolio_metrics_result.portfolio_metrics,
            target_hit_rates=target_hit_rates,
            expectancy_dollars=portfolio_metrics_result.expectancy_dollars,
            segmentation=segmentation,
            segment_records=segment_records,
            segment_context=sorted_segment_context,
            filter_metadata=filter_metadata,
            filter_summary=filter_summary,
            directional_metrics=directional_metrics,
            notes=notes,
        )


def build_strategy_modeling_service(
    *,
    list_universe_loader: StrategyUniverseLoader | None = None,
    daily_loader: StrategyDailyLoader | None = None,
    required_sessions_builder: StrategyRequiredSessionsBuilder | None = None,
    intraday_loader: StrategyIntradayLoader | None = None,
    feature_computer: StrategyFeatureComputer | None = None,
    signal_builder: StrategySignalBuilder | None = None,
    trade_simulator: StrategyTradeSimulator | None = None,
    portfolio_builder: StrategyPortfolioBuilder | None = None,
    metrics_computer: StrategyMetricsComputer | None = None,
    segmentation_aggregator: StrategySegmentationAggregator | None = None,
) -> StrategyModelingService:
    return StrategyModelingService(
        list_universe_loader=list_universe_loader or list_strategy_modeling_universe,
        daily_loader=daily_loader or load_daily_ohlc_history,
        required_sessions_builder=required_sessions_builder or build_required_intraday_sessions,
        intraday_loader=intraday_loader or load_required_intraday_bars,
        feature_computer=feature_computer or compute_strategy_features,
        signal_builder=signal_builder or build_strategy_signal_events,
        trade_simulator=trade_simulator or simulate_strategy_trade_paths,
        portfolio_builder=portfolio_builder or build_strategy_portfolio_ledger,
        metrics_computer=metrics_computer or compute_strategy_metrics,
        segmentation_aggregator=segmentation_aggregator or aggregate_strategy_segmentation,
    )


def _resolve_policy_config(
    value: StrategyModelingPolicyConfig | Mapping[str, Any] | None,
) -> StrategyModelingPolicyConfig:
    if value is None:
        return StrategyModelingPolicyConfig()
    if isinstance(value, StrategyModelingPolicyConfig):
        return value
    return parse_strategy_modeling_policy_config(value)


def _resolve_feature_config(
    value: StrategyFeatureConfig | Mapping[str, Any] | None,
) -> StrategyFeatureConfig:
    if value is None:
        return StrategyFeatureConfig()
    if isinstance(value, StrategyFeatureConfig):
        return value
    return parse_strategy_feature_config(value)


def _resolve_segmentation_config(
    value: StrategySegmentationConfig | Mapping[str, Any] | None,
) -> StrategySegmentationConfig:
    if value is None:
        return StrategySegmentationConfig()
    if isinstance(value, StrategySegmentationConfig):
        return value
    return parse_strategy_segmentation_config(value)


def _resolve_filter_config(
    value: StrategyEntryFilterConfig | Mapping[str, Any] | None,
) -> StrategyEntryFilterConfig:
    if value is None:
        return StrategyEntryFilterConfig()
    if isinstance(value, StrategyEntryFilterConfig):
        return value
    return StrategyEntryFilterConfig.model_validate(dict(value))


def _resolve_requested_symbols(
    requested: Sequence[str] | None,
    *,
    universe_symbols: Sequence[str],
) -> list[str]:
    if requested:
        seed = requested
    else:
        seed = universe_symbols

    out: list[str] = []
    seen: set[str] = set()
    for value in seed:
        symbol = normalize_symbol(value)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        out.append(symbol)
    return out


def _normalize_required_sessions(
    sessions: Mapping[str, Sequence[date]],
) -> dict[str, tuple[date, ...]]:
    out: dict[str, tuple[date, ...]] = {}
    for symbol in sorted(sessions):
        out[symbol] = tuple(sorted(set(sessions[symbol])))
    return out


def _daily_candles_to_ohlc_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close"])

    out = frame.copy()
    ts = pd.to_datetime(out.get("ts"), errors="coerce")
    opened = pd.to_numeric(out.get("open"), errors="coerce")
    high = pd.to_numeric(out.get("high"), errors="coerce")
    low = pd.to_numeric(out.get("low"), errors="coerce")
    close = pd.to_numeric(out.get("close"), errors="coerce")

    normalized = pd.DataFrame(
        {
            "Open": opened.to_numpy(),
            "High": high.to_numpy(),
            "Low": low.to_numpy(),
            "Close": close.to_numpy(),
        },
        index=pd.DatetimeIndex(ts),
    )
    normalized = normalized.dropna(subset=["Open", "High", "Low", "Close"])
    normalized = normalized.loc[~normalized.index.isna()]
    normalized = normalized.sort_index(kind="stable")
    normalized = normalized.loc[~normalized.index.duplicated(keep="first")]
    return normalized


def _build_event_segment_context_rows(
    *,
    symbol: str,
    events: Sequence[StrategySignalEvent],
    features: pd.DataFrame,
    feature_config: StrategyFeatureConfig,
    unknown_value: str,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for event in sorted(events, key=_signal_event_sort_key):
        feature_row = _feature_row_for_event(features, event.signal_ts)
        note_map = _notes_to_map(event.notes)

        bars_bucket = _string_or_none(feature_row.get("bars_since_swing_bucket"))
        if bars_bucket is None:
            bars_since_swing = _finite_float(note_map.get("bars_since_swing"))
            bars_bucket = label_bars_since_swing_bucket(
                bars_since_swing,
                boundaries=feature_config.bars_since_swing_boundaries,
            )

        volatility_regime = _first_non_null(
            feature_row.get("volatility_regime"),
            feature_row.get("realized_vol_regime"),
        )

        out.append(
            {
                "event_id": str(event.event_id),
                "ticker": normalize_symbol(symbol),
                "extension_bucket": _normalize_segment_value(
                    feature_row.get("extension_bucket"),
                    unknown_value=unknown_value,
                ),
                "rsi_regime": _normalize_segment_value(
                    feature_row.get("rsi_regime"),
                    unknown_value=unknown_value,
                ),
                "rsi_divergence": _normalize_segment_value(
                    feature_row.get("rsi_divergence"),
                    unknown_value=unknown_value,
                ),
                "volatility_regime": _normalize_segment_value(
                    volatility_regime,
                    unknown_value=unknown_value,
                ),
                "bars_since_swing_bucket": _normalize_segment_value(
                    bars_bucket,
                    unknown_value=unknown_value,
                ),
            }
        )
    return out


def _feature_row_for_event(features: pd.DataFrame, signal_ts: object) -> Mapping[str, Any]:
    if features is None or features.empty:
        return {}

    try:
        ts = pd.Timestamp(signal_ts)
    except Exception:  # noqa: BLE001
        return {}

    candidates = [ts]
    index = features.index
    if isinstance(index, pd.DatetimeIndex):
        if ts.tzinfo is not None and index.tz is None:
            candidates.append(ts.tz_localize(None))
        if ts.tzinfo is None and index.tz is not None:
            candidates.append(ts.tz_localize(index.tz))
        if ts.tzinfo is not None and index.tz is not None:
            candidates.append(ts.tz_convert(index.tz))

    for candidate in candidates:
        if candidate in features.index:
            row = features.loc[candidate]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            if isinstance(row, pd.Series):
                return row.to_dict()
    return {}


def _notes_to_map(notes: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for note in notes:
        if "=" not in note:
            continue
        key, value = str(note).split("=", 1)
        key_text = str(key).strip()
        value_text = str(value).strip()
        if not key_text:
            continue
        out[key_text] = value_text
    return out


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _first_non_null(*values: object) -> object | None:
    for value in values:
        if value is not None and str(value).strip().lower() not in {"nan", "none", "null", ""}:
            return value
    return None


def _signal_event_sort_key(event: StrategySignalEvent) -> tuple[Any, ...]:
    return (
        event.signal_confirmed_ts,
        event.signal_ts,
        event.entry_ts,
        event.symbol,
        event.event_id,
    )


def _build_default_filter_summary(
    *,
    base_event_count: int,
    kept_event_count: int,
) -> dict[str, Any]:
    rejected_count = max(0, int(base_event_count) - int(kept_event_count))
    return {
        "base_event_count": int(base_event_count),
        "kept_event_count": int(kept_event_count),
        "rejected_event_count": rejected_count,
        "reject_counts": {},
    }


def _select_portfolio_target_subset(
    trades: Sequence[StrategyTradeSimulation],
    *,
    target_ladder: Sequence[StrategyRTarget] | None,
) -> StrategyPortfolioTargetSubset:
    preferred_target_label: str | None = None
    preferred_target_r: float | None = None
    if target_ladder:
        first = target_ladder[0]
        preferred_target_label = str(first.label).strip() or None
        preferred_target_r = float(first.target_r)
    return select_portfolio_trade_subset(
        trades,
        preferred_target_label=preferred_target_label,
        preferred_target_r=preferred_target_r,
    )


def _build_directional_metrics(
    *,
    all_trades: Sequence[StrategyTradeSimulation],
    portfolio_trades: Sequence[StrategyTradeSimulation],
    portfolio_subset: StrategyPortfolioTargetSubset,
    portfolio_builder: StrategyPortfolioBuilder,
    metrics_computer: StrategyMetricsComputer,
    policy: StrategyModelingPolicyConfig,
    starting_capital: float,
    max_concurrent_positions: int | None,
) -> dict[str, Any]:
    combined_trades = tuple(portfolio_trades)
    long_trades = tuple(trade for trade in combined_trades if str(trade.direction).strip().lower() == "long")
    short_trades = tuple(trade for trade in combined_trades if str(trade.direction).strip().lower() == "short")

    closed_combined = sum(1 for trade in combined_trades if _is_closed_trade(trade))
    closed_long = sum(1 for trade in long_trades if _is_closed_trade(trade))
    closed_short = sum(1 for trade in short_trades if _is_closed_trade(trade))

    combined_payload = _run_directional_counterfactual(
        combined_trades,
        portfolio_builder=portfolio_builder,
        metrics_computer=metrics_computer,
        policy=policy,
        starting_capital=starting_capital,
        max_concurrent_positions=max_concurrent_positions,
    )
    long_payload = _run_directional_counterfactual(
        long_trades,
        portfolio_builder=portfolio_builder,
        metrics_computer=metrics_computer,
        policy=policy,
        starting_capital=starting_capital,
        max_concurrent_positions=max_concurrent_positions,
    )
    short_payload = _run_directional_counterfactual(
        short_trades,
        portfolio_builder=portfolio_builder,
        metrics_computer=metrics_computer,
        policy=policy,
        starting_capital=starting_capital,
        max_concurrent_positions=max_concurrent_positions,
    )

    return {
        "counts": {
            "all_simulated_trade_count": int(len(all_trades)),
            "portfolio_subset_trade_count": int(len(combined_trades)),
            "portfolio_subset_closed_trade_count": int(closed_combined),
            "portfolio_subset_long_trade_count": int(len(long_trades)),
            "portfolio_subset_long_closed_trade_count": int(closed_long),
            "portfolio_subset_short_trade_count": int(len(short_trades)),
            "portfolio_subset_short_closed_trade_count": int(closed_short),
        },
        "portfolio_target": {
            "target_label": portfolio_subset.target_label,
            "target_r": portfolio_subset.target_r,
            "selection_source": portfolio_subset.selection_source,
        },
        "combined": {
            **combined_payload,
        },
        "long_only": {
            **long_payload,
        },
        "short_only": {
            **short_payload,
        },
    }


def _run_directional_counterfactual(
    trades: Sequence[StrategyTradeSimulation],
    *,
    portfolio_builder: StrategyPortfolioBuilder,
    metrics_computer: StrategyMetricsComputer,
    policy: StrategyModelingPolicyConfig,
    starting_capital: float,
    max_concurrent_positions: int | None,
) -> dict[str, Any]:
    portfolio = portfolio_builder(
        trades,
        starting_capital=starting_capital,
        policy=policy,
        max_concurrent_positions=max_concurrent_positions,
    )
    metrics = metrics_computer(
        trades,
        portfolio.equity_curve,
        starting_capital=starting_capital,
    )
    payload = serialize_strategy_metrics_result(metrics)
    payload.update(
        {
            "simulated_trade_count": int(len(trades)),
            "closed_trade_count": int(sum(1 for trade in trades if _is_closed_trade(trade))),
            "accepted_trade_count": int(len(portfolio.accepted_trade_ids)),
            "skipped_trade_count": int(len(portfolio.skipped_trade_ids)),
        }
    )
    return payload


def _segment_records_from_segmentation(
    segmentation: StrategySegmentationResult,
) -> tuple[StrategySegmentRecord, ...]:
    out: list[StrategySegmentRecord] = []
    for row in segmentation.segments:
        out.append(
            StrategySegmentRecord(
                segment_dimension=row.segment_dimension,  # type: ignore[arg-type]
                segment_value=row.segment_value,
                trade_count=row.trade_count,
                win_rate=row.win_rate,
                avg_realized_r=row.avg_realized_r,
                expectancy_r=row.expectancy_r,
                profit_factor=row.profit_factor,
                sharpe_ratio=row.sharpe_ratio,
                max_drawdown_pct=row.max_drawdown_pct,
            )
        )
    out.sort(key=lambda item: (item.segment_dimension, item.segment_value))
    return tuple(out)


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
    "StrategyModelingRequest",
    "StrategyModelingRunResult",
    "StrategyModelingService",
    "StrategySegmentationConfig",
    "StrategySegmentationResult",
    "StrategySegmentPerformance",
    "StrategySegmentReconciliation",
    "aggregate_strategy_segmentation",
    "build_strategy_modeling_service",
    "parse_strategy_segmentation_config",
    "required_strategy_segment_dimensions",
]
