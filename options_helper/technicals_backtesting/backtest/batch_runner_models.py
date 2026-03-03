from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

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
