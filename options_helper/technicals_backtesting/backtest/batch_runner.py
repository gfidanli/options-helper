from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Literal, Mapping, Sequence

import pandas as pd


ProgressState = Literal["started", "completed", "failed"]
Clock = Callable[[], float]
LoadOhlc = Callable[[str], pd.DataFrame]
ComputeFeatures = Callable[[str, pd.DataFrame], pd.DataFrame]
SelectStrategyFeatures = Callable[[str, pd.DataFrame], pd.DataFrame]
RunStrategyBacktest = Callable[[str, pd.DataFrame], object]


@dataclass(frozen=True)
class BatchProgressEvent:
    stage: str
    state: ProgressState
    symbol: str | None = None
    elapsed_seconds: float | None = None
    detail: str | None = None


ProgressCallback = Callable[[BatchProgressEvent], None]


@dataclass(frozen=True)
class SymbolBacktestOutcome:
    symbol: str
    ok: bool
    stats: dict[str, object] | None
    equity_curve: pd.DataFrame | None
    trades: pd.DataFrame | None
    warnings: tuple[str, ...]
    error: str | None
    stage_timings: dict[str, float]


@dataclass(frozen=True)
class BatchBacktestResult:
    symbols: tuple[str, ...]
    outcomes: tuple[SymbolBacktestOutcome, ...]
    stage_timings: dict[str, float]
    progress_events: tuple[BatchProgressEvent, ...]

    @property
    def success_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.ok)

    @property
    def failure_count(self) -> int:
        return sum(1 for outcome in self.outcomes if not outcome.ok)


@dataclass(frozen=True)
class _ParsedStatsPayload:
    stats: dict[str, object]
    equity_curve: pd.DataFrame | None
    trades: pd.DataFrame | None
    warnings: tuple[str, ...]


def normalize_batch_symbols(symbols: str | Sequence[str]) -> tuple[str, ...]:
    items = [symbols] if isinstance(symbols, str) else list(symbols)
    normalized: list[str] = []
    for item in items:
        for token in str(item).split(","):
            symbol = token.strip().upper()
            if symbol:
                normalized.append(symbol)
    if not normalized:
        raise ValueError("Provide at least one symbol.")
    return tuple(normalized)


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _emit_progress(
    *,
    events: list[BatchProgressEvent],
    progress_callback: ProgressCallback | None,
    stage: str,
    state: ProgressState,
    symbol: str | None = None,
    elapsed_seconds: float | None = None,
    detail: str | None = None,
) -> None:
    event = BatchProgressEvent(
        stage=stage,
        state=state,
        symbol=symbol,
        elapsed_seconds=elapsed_seconds,
        detail=detail,
    )
    events.append(event)
    if progress_callback is not None:
        try:
            progress_callback(event)
        except Exception:  # noqa: BLE001
            return


def _run_stage(
    *,
    stage: str,
    symbol: str,
    fn: Callable[[], object],
    clock: Clock,
    stage_timings: dict[str, float],
    events: list[BatchProgressEvent],
    progress_callback: ProgressCallback | None,
) -> object:
    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage=stage,
        state="started",
        symbol=symbol,
    )
    started = clock()
    try:
        output = fn()
    except Exception as exc:  # noqa: BLE001
        elapsed = clock() - started
        stage_timings[stage] = stage_timings.get(stage, 0.0) + elapsed
        _emit_progress(
            events=events,
            progress_callback=progress_callback,
            stage=stage,
            state="failed",
            symbol=symbol,
            elapsed_seconds=elapsed,
            detail=_format_exception(exc),
        )
        raise
    elapsed = clock() - started
    stage_timings[stage] = stage_timings.get(stage, 0.0) + elapsed
    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage=stage,
        state="completed",
        symbol=symbol,
        elapsed_seconds=elapsed,
    )
    return output


def _parse_stats_mapping(raw_stats: object) -> dict[str, object]:
    if isinstance(raw_stats, pd.Series):
        return {str(key): value for key, value in raw_stats.to_dict().items()}
    if isinstance(raw_stats, Mapping):
        return {str(key): value for key, value in raw_stats.items()}
    raise ValueError(
        f"run_strategy_backtest returned unsupported payload type: {type(raw_stats).__name__}"
    )


def _extract_frame(
    *,
    payload: Mapping[str, object],
    key: str,
    warnings: list[str],
) -> pd.DataFrame | None:
    value = payload.get(key)
    label = key.lstrip("_")
    if value is None:
        warnings.append(f"missing_{label}")
        return None
    if not isinstance(value, pd.DataFrame):
        warnings.append(f"invalid_{label}_type:{type(value).__name__}")
        return None
    return value.copy()


def _parse_stats_payload(raw_stats: object) -> _ParsedStatsPayload:
    payload = _parse_stats_mapping(raw_stats)
    stats = {key: value for key, value in payload.items() if not key.startswith("_")}
    warnings: list[str] = []
    equity_curve = _extract_frame(payload=payload, key="_equity_curve", warnings=warnings)
    trades = _extract_frame(payload=payload, key="_trades", warnings=warnings)
    if not stats:
        warnings.append("empty_stats")
    return _ParsedStatsPayload(
        stats=stats,
        equity_curve=equity_curve,
        trades=trades,
        warnings=tuple(warnings),
    )


def _execute_symbol_stages(
    *,
    symbol: str,
    load_ohlc: LoadOhlc,
    compute_features: ComputeFeatures,
    select_strategy_features: SelectStrategyFeatures,
    run_strategy_backtest: RunStrategyBacktest,
    clock: Clock,
    stage_timings: dict[str, float],
    events: list[BatchProgressEvent],
    progress_callback: ProgressCallback | None,
) -> _ParsedStatsPayload:
    ohlc = _run_stage(
        stage="load_ohlc",
        symbol=symbol,
        fn=lambda: load_ohlc(symbol),
        clock=clock,
        stage_timings=stage_timings,
        events=events,
        progress_callback=progress_callback,
    )
    features = _run_stage(
        stage="compute_features",
        symbol=symbol,
        fn=lambda: compute_features(symbol, ohlc),
        clock=clock,
        stage_timings=stage_timings,
        events=events,
        progress_callback=progress_callback,
    )
    strategy_features = _run_stage(
        stage="select_strategy_features",
        symbol=symbol,
        fn=lambda: select_strategy_features(symbol, features),
        clock=clock,
        stage_timings=stage_timings,
        events=events,
        progress_callback=progress_callback,
    )
    raw_stats = _run_stage(
        stage="run_backtest",
        symbol=symbol,
        fn=lambda: run_strategy_backtest(symbol, strategy_features),
        clock=clock,
        stage_timings=stage_timings,
        events=events,
        progress_callback=progress_callback,
    )
    return _parse_stats_payload(raw_stats)


def _build_failed_symbol_outcome(
    *,
    symbol: str,
    error: str,
    exc_type: str,
    stage_timings: dict[str, float],
) -> SymbolBacktestOutcome:
    return SymbolBacktestOutcome(
        symbol=symbol,
        ok=False,
        stats=None,
        equity_curve=None,
        trades=None,
        warnings=(f"symbol_failed:{exc_type}",),
        error=error,
        stage_timings=stage_timings,
    )


def _run_symbol_backtest(
    *,
    symbol: str,
    load_ohlc: LoadOhlc,
    compute_features: ComputeFeatures,
    select_strategy_features: SelectStrategyFeatures,
    run_strategy_backtest: RunStrategyBacktest,
    clock: Clock,
    events: list[BatchProgressEvent],
    progress_callback: ProgressCallback | None,
) -> SymbolBacktestOutcome:
    stage_timings: dict[str, float] = {}
    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage="symbol",
        state="started",
        symbol=symbol,
    )
    symbol_started = clock()
    try:
        parsed = _execute_symbol_stages(
            symbol=symbol,
            load_ohlc=load_ohlc,
            compute_features=compute_features,
            select_strategy_features=select_strategy_features,
            run_strategy_backtest=run_strategy_backtest,
            clock=clock,
            stage_timings=stage_timings,
            events=events,
            progress_callback=progress_callback,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = clock() - symbol_started
        stage_timings["symbol_total"] = elapsed
        error = _format_exception(exc)
        _emit_progress(
            events=events,
            progress_callback=progress_callback,
            stage="symbol",
            state="failed",
            symbol=symbol,
            elapsed_seconds=elapsed,
            detail=error,
        )
        return _build_failed_symbol_outcome(
            symbol=symbol,
            error=error,
            exc_type=type(exc).__name__,
            stage_timings=stage_timings,
        )

    elapsed = clock() - symbol_started
    stage_timings["symbol_total"] = elapsed
    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage="symbol",
        state="completed",
        symbol=symbol,
        elapsed_seconds=elapsed,
    )
    return SymbolBacktestOutcome(
        symbol=symbol,
        ok=True,
        stats=parsed.stats,
        equity_curve=parsed.equity_curve,
        trades=parsed.trades,
        warnings=parsed.warnings,
        error=None,
        stage_timings=stage_timings,
    )


def _merge_stage_timings(
    *,
    aggregate: dict[str, float],
    stage_timings: Mapping[str, float],
) -> None:
    for stage, elapsed in stage_timings.items():
        aggregate[stage] = aggregate.get(stage, 0.0) + float(elapsed)


def run_backtest_batch(
    *,
    symbols: str | Sequence[str],
    load_ohlc: LoadOhlc,
    compute_features: ComputeFeatures,
    select_strategy_features: SelectStrategyFeatures,
    run_strategy_backtest: RunStrategyBacktest,
    progress_callback: ProgressCallback | None = None,
    clock: Clock | None = None,
) -> BatchBacktestResult:
    resolved_symbols = normalize_batch_symbols(symbols)
    timer = clock or time.perf_counter
    events: list[BatchProgressEvent] = []
    stage_timings: dict[str, float] = {}
    outcomes: list[SymbolBacktestOutcome] = []

    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage="batch",
        state="started",
    )
    batch_started = timer()
    for symbol in resolved_symbols:
        outcome = _run_symbol_backtest(
            symbol=symbol,
            load_ohlc=load_ohlc,
            compute_features=compute_features,
            select_strategy_features=select_strategy_features,
            run_strategy_backtest=run_strategy_backtest,
            clock=timer,
            events=events,
            progress_callback=progress_callback,
        )
        outcomes.append(outcome)
        _merge_stage_timings(aggregate=stage_timings, stage_timings=outcome.stage_timings)

    batch_elapsed = timer() - batch_started
    stage_timings["batch_total"] = batch_elapsed
    failed_count = sum(1 for outcome in outcomes if not outcome.ok)
    _emit_progress(
        events=events,
        progress_callback=progress_callback,
        stage="batch",
        state="completed",
        elapsed_seconds=batch_elapsed,
        detail=f"symbols={len(resolved_symbols)} failed={failed_count}",
    )
    return BatchBacktestResult(
        symbols=resolved_symbols,
        outcomes=tuple(outcomes),
        stage_timings=stage_timings,
        progress_events=tuple(events),
    )


__all__ = [
    "BatchBacktestResult",
    "BatchProgressEvent",
    "ProgressCallback",
    "SymbolBacktestOutcome",
    "normalize_batch_symbols",
    "run_backtest_batch",
]
