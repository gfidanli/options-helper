from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

from rich.console import Console
import typer

from .strategy_model_helpers_legacy import _intraday_coverage_block_message

_SERVICE_STAGE_ATTRS = (
    "list_universe_loader",
    "daily_loader",
    "required_sessions_builder",
    "intraday_loader",
    "feature_computer",
    "signal_builder",
    "trade_simulator",
    "portfolio_builder",
    "metrics_computer",
    "segmentation_aggregator",
)


@dataclass
class StageTracker:
    console: Console
    show_progress: bool
    timings: dict[str, float] = field(default_factory=dict)

    def format_seconds(self, value: float) -> str:
        return f"{value:.2f}s"

    def add_timing(self, stage_name: str, elapsed: float) -> None:
        self.timings[stage_name] = self.timings.get(stage_name, 0.0) + elapsed

    def _safe_detail(
        self,
        detail_builder: Callable[[Any, tuple[Any, ...], dict[str, Any]], str] | None,
        output: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        if detail_builder is None:
            return ""
        try:
            detail = detail_builder(output, args, kwargs)
        except Exception:  # noqa: BLE001
            return ""
        if not detail:
            return ""
        return f" | {detail}"

    def wrap_stage(
        self,
        stage_name: str,
        fn: Callable[..., Any],
        detail_builder: Callable[[Any, tuple[Any, ...], dict[str, Any]], str] | None = None,
    ) -> Callable[..., Any]:
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.show_progress:
                self.console.print(f"[cyan]... {stage_name}[/cyan]")
            started = time.perf_counter()
            output = fn(*args, **kwargs)
            elapsed = time.perf_counter() - started
            self.add_timing(stage_name, elapsed)
            if self.show_progress:
                detail = self._safe_detail(detail_builder, output, args, kwargs)
                self.console.print(
                    f"[green]OK  {stage_name} ({self.format_seconds(elapsed)}){detail}[/green]"
                )
            return output

        return _wrapped


def _detail_universe(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    return f"symbols={len(tuple(getattr(out, 'symbols', ()) or ())) or 0}"


def _detail_daily(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    return (
        f"loaded={len(getattr(out, 'candles_by_symbol', {}) or {})} "
        f"skipped={len(tuple(getattr(out, 'skipped_symbols', ()) or ())) or 0} "
        f"missing={len(tuple(getattr(out, 'missing_symbols', ()) or ())) or 0}"
    )


def _detail_required_sessions(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    return f"symbols={len(out or {})} sessions={sum(len(tuple(v or ())) for v in (out or {}).values())}"


def _detail_intraday(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    blocked = len(tuple(getattr(getattr(out, 'preflight', None), 'blocked_symbols', ()) or ())) or 0
    return f"symbols={len(getattr(out, 'bars_by_symbol', {}) or {})} blocked={blocked}"


def _detail_signals(out: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    symbol = str(kwargs.get("symbol", "")).upper()
    return f"symbol={symbol} events={len(tuple(out or ())) or 0}"


def _detail_trades(out: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    events = len(tuple(args[0] or ())) if args else 0
    targets = len(tuple(kwargs.get("target_ladder") or ())) or 0
    trades = len(tuple(out or ())) or 0
    return f"events={events} targets={targets} trades={trades}"


def _detail_portfolio(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    accepted = len(tuple(getattr(out, "accepted_trade_ids", ()) or ())) or 0
    skipped = len(tuple(getattr(out, "skipped_trade_ids", ()) or ())) or 0
    return f"accepted={accepted} skipped={skipped}"


def _detail_segments(out: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
    return f"segments={len(tuple(getattr(out, 'segments', ()) or ())) or 0}"


def instrument_service(*, service: Any, tracker: StageTracker) -> Any:
    if not tracker.show_progress or not all(hasattr(service, attr) for attr in _SERVICE_STAGE_ATTRS):
        return service

    from options_helper.analysis.strategy_modeling import StrategyModelingService

    return StrategyModelingService(
        list_universe_loader=tracker.wrap_stage("Loading universe", service.list_universe_loader, _detail_universe),
        daily_loader=tracker.wrap_stage("Loading daily candles", service.daily_loader, _detail_daily),
        required_sessions_builder=tracker.wrap_stage(
            "Building required sessions",
            service.required_sessions_builder,
            _detail_required_sessions,
        ),
        intraday_loader=tracker.wrap_stage("Loading intraday bars", service.intraday_loader, _detail_intraday),
        feature_computer=tracker.wrap_stage("Computing features", service.feature_computer),
        signal_builder=tracker.wrap_stage("Building signals", service.signal_builder, _detail_signals),
        trade_simulator=tracker.wrap_stage("Simulating trades", service.trade_simulator, _detail_trades),
        portfolio_builder=tracker.wrap_stage("Building portfolio ledger", service.portfolio_builder, _detail_portfolio),
        metrics_computer=tracker.wrap_stage("Computing metrics", service.metrics_computer),
        segmentation_aggregator=tracker.wrap_stage(
            "Building segmentation",
            service.segmentation_aggregator,
            _detail_segments,
        ),
    )


def run_strategy_model(
    *,
    service: Any,
    request: Any,
    strategy: str,
    symbol_count: int,
    intraday_timeframe: str,
    tracker: StageTracker,
) -> tuple[Any, float]:
    if tracker.show_progress:
        tracker.console.print(
            (
                f"[cyan]Starting strategy-model run: strategy={strategy} "
                f"symbols={symbol_count} timeframe={intraday_timeframe}[/cyan]"
            )
        )
    started = time.perf_counter()
    run_result = service.run(request)
    run_elapsed = time.perf_counter() - started
    if tracker.show_progress:
        tracker.console.print(f"[green]Run complete in {tracker.format_seconds(run_elapsed)}[/green]")

    block_message = _intraday_coverage_block_message(getattr(run_result, "intraday_preflight", None))
    if block_message is not None:
        raise typer.BadParameter(block_message)
    return run_result, run_elapsed


def emit_stage_timings(*, tracker: StageTracker) -> None:
    if not tracker.show_progress or not tracker.timings:
        return
    tracker.console.print("Stage timings:")
    for name in sorted(tracker.timings, key=tracker.timings.get, reverse=True):
        tracker.console.print(f"  - {name}: {tracker.format_seconds(tracker.timings[name])}")


__all__ = [
    "StageTracker",
    "instrument_service",
    "run_strategy_model",
    "emit_stage_timings",
]
