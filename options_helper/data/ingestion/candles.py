from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import threading
import time

from options_helper.data.candles import CandleStore
from options_helper.data.ingestion.tuning import EndpointStats, build_endpoint_stats


@dataclass(frozen=True)
class CandleIngestResult:
    symbol: str
    status: str
    last_date: date | None
    error: str | None


@dataclass(frozen=True)
class CandleIngestSummary:
    calls: int
    ok: int
    empty: int
    error: int
    endpoint_stats: EndpointStats


@dataclass(frozen=True)
class CandleIngestOutput:
    results: list[CandleIngestResult]
    summary: CandleIngestSummary


class _RequestRateLimiter:
    def __init__(self, max_requests_per_second: float | None) -> None:
        if max_requests_per_second is None or max_requests_per_second <= 0:
            self._min_interval_seconds = 0.0
        else:
            self._min_interval_seconds = 1.0 / float(max_requests_per_second)
        self._next_allowed_at = 0.0
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                remaining = self._next_allowed_at - now
                if remaining <= 0:
                    self._next_allowed_at = now + self._min_interval_seconds
                    return
            time.sleep(remaining)


def _looks_like_timeout(error: Exception) -> bool:
    message = str(error).lower()
    return "timeout" in message or "timed out" in message


def _looks_like_429(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def ingest_candles_with_summary(
    store: CandleStore,
    symbols: list[str],
    *,
    period: str = "max",
    best_effort: bool = True,
    concurrency: int = 1,
    max_requests_per_second: float | None = None,
) -> CandleIngestOutput:
    normalized_symbols: list[str] = []
    for symbol in symbols:
        sym = symbol.strip().upper()
        if sym:
            normalized_symbols.append(sym)

    results_by_symbol: dict[str, CandleIngestResult] = {}
    latencies_ms: list[float] = []
    timeout_count = 0
    rate_limit_429 = 0
    error_count = 0
    limiter = _RequestRateLimiter(max_requests_per_second)
    lock = threading.Lock()

    def _ingest_symbol(sym: str) -> CandleIngestResult:
        nonlocal timeout_count, rate_limit_429, error_count
        limiter.wait_turn()
        started = time.perf_counter()
        try:
            history = store.get_daily_history(sym, period=period)
            if history is None or history.empty:
                result = CandleIngestResult(symbol=sym, status="empty", last_date=None, error=None)
            else:
                last_dt = history.index.max()
                result = CandleIngestResult(symbol=sym, status="ok", last_date=last_dt.date(), error=None)
        except Exception as exc:  # noqa: BLE001
            result = CandleIngestResult(symbol=sym, status="error", last_date=None, error=str(exc))
            with lock:
                error_count += 1
                if _looks_like_timeout(exc):
                    timeout_count += 1
                if _looks_like_429(exc):
                    rate_limit_429 += 1
            if not best_effort:
                raise
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with lock:
                latencies_ms.append(elapsed_ms)
        return result

    max_workers = max(1, int(concurrency or 1))
    if max_workers == 1:
        for sym in normalized_symbols:
            result = _ingest_symbol(sym)
            results_by_symbol[sym] = result
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_ingest_symbol, sym): sym for sym in normalized_symbols}
            for future in as_completed(futures):
                sym = futures[future]
                result = future.result()
                results_by_symbol[sym] = result

    ordered_results = [results_by_symbol[sym] for sym in normalized_symbols if sym in results_by_symbol]
    ok = sum(1 for item in ordered_results if item.status == "ok")
    empty = sum(1 for item in ordered_results if item.status == "empty")
    errors = sum(1 for item in ordered_results if item.status == "error")
    endpoint_stats = build_endpoint_stats(
        calls=len(ordered_results),
        retries=0,
        rate_limit_429=rate_limit_429,
        timeout_count=timeout_count,
        error_count=errors,
        latencies_ms=latencies_ms,
    )

    return CandleIngestOutput(
        results=ordered_results,
        summary=CandleIngestSummary(
            calls=len(ordered_results),
            ok=ok,
            empty=empty,
            error=errors,
            endpoint_stats=endpoint_stats,
        ),
    )


def ingest_candles(
    store: CandleStore,
    symbols: list[str],
    *,
    period: str = "max",
    best_effort: bool = True,
    concurrency: int = 1,
    max_requests_per_second: float | None = None,
) -> list[CandleIngestResult]:
    output = ingest_candles_with_summary(
        store,
        symbols,
        period=period,
        best_effort=best_effort,
        concurrency=concurrency,
        max_requests_per_second=max_requests_per_second,
    )
    return output.results
