from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
import threading
import time as time_mod
import pandas as pd
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.ingestion.options_bars_backfill_planner_legacy import (
    BarsBatchPlan,
    BarsFetchPlan,
)
from options_helper.data.ingestion.options_bars_helpers import (
    error_status as _error_status,
    looks_like_429 as _looks_like_429,
    looks_like_timeout as _looks_like_timeout,
)
from options_helper.data.option_bars import OptionBarsStore
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
                now = time_mod.monotonic()
                remaining = self._next_allowed_at - now
                if remaining <= 0:
                    self._next_allowed_at = now + self._min_interval_seconds
                    return
            time_mod.sleep(remaining)
@dataclass
class BackfillExecutionResult:
    ok_contracts: int
    error_contracts: int
    bars_rows: int
    endpoint_calls: int
    endpoint_errors: int
    endpoint_timeout_count: int
    endpoint_rate_limit_429: int
    endpoint_latencies_ms: list[float]
    endpoint_split_count: int
    endpoint_fallback_count: int
@dataclass
class _BackfillExecutor:
    client: AlpacaClient
    store: OptionBarsStore
    provider: str
    page_limit: int | None
    bars_max_requests_per_second: float | None
    bars_concurrency: int
    bars_batch_mode: str
    bars_batch_size: int
    bars_write_batch_size: int
    fail_fast: bool
    plan_by_symbol: dict[str, BarsFetchPlan]
    rate_limiter: _RequestRateLimiter = field(init=False)
    max_workers: int = field(init=False)
    batch_mode: str = field(init=False)
    batch_size: int = field(init=False)
    write_batch_size: int = field(init=False)
    pending_bars: list[pd.DataFrame] = field(default_factory=list)
    pending_success_symbols: list[str] = field(default_factory=list)
    pending_success_rows: dict[str, int] = field(default_factory=dict)
    pending_success_start: dict[str, datetime] = field(default_factory=dict)
    pending_success_end: dict[str, datetime] = field(default_factory=dict)
    pending_success_count: int = 0
    pending_errors: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    pending_error_count: int = 0
    ok_contracts: int = 0
    error_contracts: int = 0
    bars_rows: int = 0
    endpoint_calls: int = 0
    endpoint_errors: int = 0
    endpoint_timeout_count: int = 0
    endpoint_rate_limit_429: int = 0
    endpoint_latencies_ms: list[float] = field(default_factory=list)
    endpoint_split_count: int = 0
    endpoint_fallback_count: int = 0
    endpoint_lock: threading.Lock = field(default_factory=threading.Lock)
    def __post_init__(self) -> None:
        self.rate_limiter = _RequestRateLimiter(self.bars_max_requests_per_second)
        self.max_workers = 1 if self.fail_fast else max(1, int(self.bars_concurrency))
        self.batch_mode = str(self.bars_batch_mode or "adaptive").strip().lower()
        if self.batch_mode not in {"adaptive", "per-contract"}:
            self.batch_mode = "adaptive"
        self.batch_size = max(1, int(self.bars_batch_size))
        self.write_batch_size = max(1, int(self.bars_write_batch_size))
    def run(self, fetch_plans: list[BarsFetchPlan]) -> BackfillExecutionResult:
        if self.batch_mode == "per-contract":
            self._run_per_contract(fetch_plans)
        else:
            self._run_adaptive(fetch_plans)
        self._flush_buffers()
        return BackfillExecutionResult(
            ok_contracts=self.ok_contracts,
            error_contracts=self.error_contracts,
            bars_rows=self.bars_rows,
            endpoint_calls=self.endpoint_calls,
            endpoint_errors=self.endpoint_errors,
            endpoint_timeout_count=self.endpoint_timeout_count,
            endpoint_rate_limit_429=self.endpoint_rate_limit_429,
            endpoint_latencies_ms=self.endpoint_latencies_ms,
            endpoint_split_count=self.endpoint_split_count,
            endpoint_fallback_count=self.endpoint_fallback_count,
        )
    def _fetch_bars(self, symbols: list[str], *, fetch_start: date, fetch_end: date) -> pd.DataFrame:
        self.rate_limiter.wait_turn()
        started = time_mod.perf_counter()
        try:
            return self.client.get_option_bars_daily_full(
                symbols,
                start=fetch_start,
                end=fetch_end,
                interval="1d",
                chunk_size=max(1, len(symbols)),
                page_limit=self.page_limit,
            )
        except Exception as exc:  # noqa: BLE001
            with self.endpoint_lock:
                self.endpoint_errors += 1
                if _looks_like_timeout(exc):
                    self.endpoint_timeout_count += 1
                if _looks_like_429(exc):
                    self.endpoint_rate_limit_429 += 1
            raise
        finally:
            elapsed_ms = (time_mod.perf_counter() - started) * 1000.0
            with self.endpoint_lock:
                self.endpoint_calls += 1
                self.endpoint_latencies_ms.append(elapsed_ms)
    def _run_per_contract(self, fetch_plans: list[BarsFetchPlan]) -> None:
        if self.max_workers <= 1:
            for plan in fetch_plans:
                item_plan, df_bars, exc = self._fetch_single_plan(plan)
                self._process_outcome(item_plan, df_bars, exc)
                if self.fail_fast and exc is not None:
                    self._flush_buffers()
                    raise exc
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._fetch_single_plan, plan): plan for plan in fetch_plans}
            for future in as_completed(futures):
                item_plan, df_bars, exc = future.result()
                self._process_outcome(item_plan, df_bars, exc)
    def _run_adaptive(self, fetch_plans: list[BarsFetchPlan]) -> None:
        batch_plans = self._build_batch_plans(fetch_plans)
        if self.max_workers <= 1:
            for batch_plan in batch_plans:
                outcomes = self._resolve_adaptive_batch(batch_plan)
                for item_plan, df_bars, exc in outcomes:
                    self._process_outcome(item_plan, df_bars, exc)
                    if self.fail_fast and exc is not None:
                        self._flush_buffers()
                        raise exc
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._resolve_adaptive_batch, batch_plan): batch_plan for batch_plan in batch_plans}
            for future in as_completed(futures):
                for item_plan, df_bars, exc in future.result():
                    self._process_outcome(item_plan, df_bars, exc)
    def _build_batch_plans(self, fetch_plans: list[BarsFetchPlan]) -> list[BarsBatchPlan]:
        grouped_symbols: dict[tuple[date, date], list[str]] = {}
        for plan in fetch_plans:
            grouped_symbols.setdefault((plan.fetch_start, plan.fetch_end), []).append(plan.symbol)
        batch_plans: list[BarsBatchPlan] = []
        for (fetch_start, fetch_end), symbols in grouped_symbols.items():
            ordered = sorted({sym for sym in symbols if sym})
            for index in range(0, len(ordered), self.batch_size):
                chunk = ordered[index : index + self.batch_size]
                if chunk:
                    batch_plans.append(BarsBatchPlan(symbols=tuple(chunk), fetch_start=fetch_start, fetch_end=fetch_end))
        return batch_plans
    def _resolve_adaptive_batch(self, batch_plan: BarsBatchPlan) -> list[tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]]:
        symbols = [sym for sym in batch_plan.symbols if sym]
        if not symbols:
            return []
        try:
            df_bars = self._fetch_bars(symbols, fetch_start=batch_plan.fetch_start, fetch_end=batch_plan.fetch_end)
        except Exception as exc:  # noqa: BLE001
            return self._adaptive_failure(symbols, batch_plan, exc)
        if df_bars is None or df_bars.empty:
            return self._adaptive_empty(symbols, batch_plan)
        return self._adaptive_rows(symbols, batch_plan, df_bars)
    def _adaptive_failure(self, symbols: list[str], batch_plan: BarsBatchPlan, exc: Exception) -> list[tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]]:
        if len(symbols) == 1:
            plan = self.plan_by_symbol.get(symbols[0])
            return [] if plan is None else [(plan, None, exc)]
        self._increment_split(fallback=False)
        left_symbols, right_symbols = self._split_batch(symbols)
        left = self._resolve_adaptive_batch(BarsBatchPlan(tuple(left_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        right = self._resolve_adaptive_batch(BarsBatchPlan(tuple(right_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        return left + right
    def _adaptive_empty(self, symbols: list[str], batch_plan: BarsBatchPlan) -> list[tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]]:
        if len(symbols) == 1:
            plan = self.plan_by_symbol.get(symbols[0])
            return [] if plan is None else [(plan, pd.DataFrame(), None)]
        self._increment_split(fallback=True)
        left_symbols, right_symbols = self._split_batch(symbols)
        left = self._resolve_adaptive_batch(BarsBatchPlan(tuple(left_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        right = self._resolve_adaptive_batch(BarsBatchPlan(tuple(right_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        return left + right
    def _adaptive_rows(self, symbols: list[str], batch_plan: BarsBatchPlan, df_bars: pd.DataFrame) -> list[tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]]:
        normalized = df_bars.copy()
        if "contractSymbol" not in normalized.columns:
            normalized["contractSymbol"] = pd.NA
        normalized["contractSymbol"] = normalized["contractSymbol"].map(lambda value: str(value).strip().upper() if value is not None else None)
        normalized = normalized.dropna(subset=["contractSymbol"]).copy()
        if not normalized.empty:
            normalized = normalized[normalized["contractSymbol"].isin(symbols)].copy()
        present_symbols = sorted({str(value).strip().upper() for value in normalized["contractSymbol"].tolist()}) if not normalized.empty else []
        outcomes: list[tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]] = []
        for sym in present_symbols:
            plan = self.plan_by_symbol.get(sym)
            if plan is not None:
                outcomes.append((plan, normalized[normalized["contractSymbol"] == sym].copy(), None))
        missing_symbols = [sym for sym in symbols if sym not in set(present_symbols)]
        if not missing_symbols:
            return outcomes
        if len(symbols) == 1:
            plan = self.plan_by_symbol.get(symbols[0])
            return outcomes if plan is None else outcomes + [(plan, pd.DataFrame(), None)]
        self._increment_split(fallback=True)
        left_symbols, right_symbols = self._split_batch(missing_symbols)
        left = self._resolve_adaptive_batch(BarsBatchPlan(tuple(left_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        right = self._resolve_adaptive_batch(BarsBatchPlan(tuple(right_symbols), batch_plan.fetch_start, batch_plan.fetch_end))
        return outcomes + left + right
    def _split_batch(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        midpoint = max(1, len(symbols) // 2)
        return symbols[:midpoint], symbols[midpoint:]
    def _increment_split(self, *, fallback: bool) -> None:
        with self.endpoint_lock:
            self.endpoint_split_count += 1
            if fallback:
                self.endpoint_fallback_count += 1
    def _fetch_single_plan(self, plan: BarsFetchPlan) -> tuple[BarsFetchPlan, pd.DataFrame | None, Exception | None]:
        try:
            df_bars = self._fetch_bars([plan.symbol], fetch_start=plan.fetch_start, fetch_end=plan.fetch_end)
            return plan, df_bars, None
        except Exception as exc:  # noqa: BLE001
            return plan, None, exc
    def _process_outcome(self, plan: BarsFetchPlan, df_bars: pd.DataFrame | None, exc: Exception | None) -> None:
        if exc is not None:
            self._queue_error(symbol=plan.symbol, status=_error_status(exc), error_text=str(exc))
            self.error_contracts += 1
            return
        self._record_success(plan, df_bars)
    def _record_success(self, plan: BarsFetchPlan, df_bars: pd.DataFrame | None) -> None:
        if df_bars is None or df_bars.empty:
            if plan.has_coverage:
                self._queue_success(symbol=plan.symbol, rows=0, start_ts=None, end_ts=None)
                self.ok_contracts += 1
            else:
                self._queue_error(symbol=plan.symbol, status="not_found", error_text="no bars returned")
                self.error_contracts += 1
            return
        self.bars_rows += int(len(df_bars))
        normalized = df_bars.copy()
        normalized["contractSymbol"] = normalized["contractSymbol"].map(lambda v: str(v).strip().upper() if v is not None else None)
        normalized = normalized.dropna(subset=["contractSymbol"])
        rows = normalized[normalized["contractSymbol"] == plan.symbol]
        if not rows.empty:
            self._queue_success(symbol=plan.symbol, rows=int(len(rows)), start_ts=rows["ts"].min(), end_ts=rows["ts"].max(), bars_df=df_bars)
            self.ok_contracts += 1
            return
        if plan.has_coverage:
            self._queue_success(symbol=plan.symbol, rows=0, start_ts=None, end_ts=None, bars_df=df_bars)
            self.ok_contracts += 1
            return
        self._queue_error(symbol=plan.symbol, status="not_found", error_text="no bars returned")
        self.error_contracts += 1
    def _flush_success_buffers(self) -> None:
        if self.pending_bars:
            merged = pd.concat(self.pending_bars, ignore_index=True)
            self.store.upsert_bars(merged, interval="1d", provider=self.provider)
            self.pending_bars.clear()
        if not self.pending_success_symbols:
            return
        self.store.mark_meta_success(
            self.pending_success_symbols,
            interval="1d",
            provider=self.provider,
            rows=self.pending_success_rows or None,
            start_ts=self.pending_success_start or None,
            end_ts=self.pending_success_end or None,
        )
        self.pending_success_symbols.clear()
        self.pending_success_rows.clear()
        self.pending_success_start.clear()
        self.pending_success_end.clear()
        self.pending_success_count = 0
    def _flush_error_buffers(self) -> None:
        if not self.pending_errors:
            return
        for (status, error_text), symbols in self.pending_errors.items():
            if symbols:
                self.store.mark_meta_error(symbols, interval="1d", provider=self.provider, error=error_text, status=status)
        self.pending_errors.clear()
        self.pending_error_count = 0
    def _flush_buffers(self) -> None:
        apply_write_batch = getattr(self.store, "apply_write_batch", None)
        if callable(apply_write_batch):
            self._flush_via_apply_batch(apply_write_batch)
            return
        self._flush_success_buffers()
        self._flush_error_buffers()
    def _flush_via_apply_batch(self, apply_write_batch) -> None:  # noqa: ANN001
        merged = pd.concat(self.pending_bars, ignore_index=True) if self.pending_bars else None
        error_groups = [(symbols, status, error_text) for (status, error_text), symbols in self.pending_errors.items()]
        if merged is not None or self.pending_success_symbols or error_groups:
            apply_write_batch(
                bars_df=merged,
                interval="1d",
                provider=self.provider,
                success_symbols=self.pending_success_symbols,
                success_rows=self.pending_success_rows or None,
                success_start_ts=self.pending_success_start or None,
                success_end_ts=self.pending_success_end or None,
                error_groups=error_groups,
            )
        self.pending_bars.clear()
        self.pending_success_symbols.clear()
        self.pending_success_rows.clear()
        self.pending_success_start.clear()
        self.pending_success_end.clear()
        self.pending_success_count = 0
        self.pending_errors.clear()
        self.pending_error_count = 0
    def _queue_success(
        self,
        *,
        symbol: str,
        rows: int,
        start_ts: datetime | None,
        end_ts: datetime | None,
        bars_df: pd.DataFrame | None = None,
    ) -> None:
        if bars_df is not None and not bars_df.empty:
            self.pending_bars.append(bars_df)
        self.pending_success_symbols.append(symbol)
        self.pending_success_rows[symbol] = int(rows)
        if start_ts is not None:
            self.pending_success_start[symbol] = start_ts
        if end_ts is not None:
            self.pending_success_end[symbol] = end_ts
        self.pending_success_count += 1
        if self.pending_success_count >= self.write_batch_size:
            self._flush_buffers()
    def _queue_error(self, *, symbol: str, status: str, error_text: str) -> None:
        key = (str(status or "error").strip().lower() or "error", str(error_text))
        self.pending_errors.setdefault(key, []).append(symbol)
        self.pending_error_count += 1
        if self.pending_error_count >= self.write_batch_size:
            self._flush_buffers()
def execute_backfill_plans(
    *,
    client: AlpacaClient,
    store: OptionBarsStore,
    provider: str,
    page_limit: int | None,
    bars_concurrency: int,
    bars_max_requests_per_second: float | None,
    bars_batch_mode: str,
    bars_batch_size: int,
    bars_write_batch_size: int,
    fail_fast: bool,
    fetch_plans: list[BarsFetchPlan],
) -> BackfillExecutionResult:
    executor = _BackfillExecutor(
        client=client,
        store=store,
        provider=provider,
        page_limit=page_limit,
        bars_max_requests_per_second=bars_max_requests_per_second,
        bars_concurrency=bars_concurrency,
        bars_batch_mode=bars_batch_mode,
        bars_batch_size=bars_batch_size,
        bars_write_batch_size=bars_write_batch_size,
        fail_fast=fail_fast,
        plan_by_symbol={fetch_plan.symbol: fetch_plan for fetch_plan in fetch_plans},
    )
    return executor.run(fetch_plans)
__all__ = ["BackfillExecutionResult", "execute_backfill_plans"]
