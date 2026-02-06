from __future__ import annotations

from datetime import date
import time
from typing import Any

import pandas as pd

from options_helper.data.ingestion.options_bars import backfill_option_bars


class _FakeClient:
    def __init__(self, *, sleep_seconds: float = 0.0) -> None:
        self.sleep_seconds = sleep_seconds
        self.calls: list[dict[str, Any]] = []
        self.call_starts: list[float] = []

    def get_option_bars_daily_full(
        self,
        symbols: list[str],
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str = "1d",  # noqa: ARG002
        feed: str | None = None,  # noqa: ARG002
        chunk_size: int = 200,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        self.call_starts.append(time.monotonic())
        self.calls.append({"symbols": list(symbols), "start": start, "end": end})
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)
        rows: list[dict[str, object]] = []
        for sym in symbols:
            rows.append(
                {
                    "contractSymbol": sym,
                    "ts": pd.Timestamp("2026-01-02"),
                    "open": 1.0,
                    "high": 1.1,
                    "low": 0.9,
                    "close": 1.0,
                    "volume": 10,
                    "vwap": 1.0,
                    "trade_count": 2,
                }
            )
        return pd.DataFrame(rows)


class _FakeStore:
    def __init__(self) -> None:
        self.success_symbols: list[str] = []
        self.error_symbols: list[str] = []
        self.upserts: list[pd.DataFrame] = []

    def upsert_bars(
        self,
        df: pd.DataFrame,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.upserts.append(df.copy())

    def mark_meta_success(
        self,
        contract_symbols,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        status: str = "ok",  # noqa: ARG002
        rows=None,  # noqa: ANN001,ARG002
        start_ts=None,  # noqa: ANN001,ARG002
        end_ts=None,  # noqa: ANN001,ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.success_symbols.extend([str(sym).strip().upper() for sym in contract_symbols])

    def mark_meta_error(
        self,
        contract_symbols,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        error,  # noqa: ANN001,ARG002
        status: str = "error",  # noqa: ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.error_symbols.extend([str(sym).strip().upper() for sym in contract_symbols])

    def coverage(
        self,
        contract_symbol: str,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
    ) -> dict[str, Any]:
        del contract_symbol
        return {}


class _FakeStoreWithBatchCounters(_FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.upsert_calls = 0
        self.success_calls = 0
        self.error_calls = 0

    def upsert_bars(
        self,
        df: pd.DataFrame,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.upsert_calls += 1
        super().upsert_bars(df, interval=interval, provider=provider, updated_at=updated_at)

    def mark_meta_success(
        self,
        contract_symbols,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        status: str = "ok",  # noqa: ARG002
        rows=None,  # noqa: ANN001,ARG002
        start_ts=None,  # noqa: ANN001,ARG002
        end_ts=None,  # noqa: ANN001,ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.success_calls += 1
        super().mark_meta_success(
            contract_symbols,
            interval=interval,
            provider=provider,
            status=status,
            rows=rows,
            start_ts=start_ts,
            end_ts=end_ts,
            updated_at=updated_at,
        )

    def mark_meta_error(
        self,
        contract_symbols,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
        error,  # noqa: ANN001,ARG002
        status: str = "error",  # noqa: ARG002
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.error_calls += 1
        super().mark_meta_error(
            contract_symbols,
            interval=interval,
            provider=provider,
            error=error,
            status=status,
            updated_at=updated_at,
        )


class _FakeStoreWithApplyBatch(_FakeStoreWithBatchCounters):
    def __init__(self) -> None:
        super().__init__()
        self.apply_calls = 0

    def apply_write_batch(
        self,
        *,
        bars_df: pd.DataFrame | None,
        interval: str = "1d",
        provider: str,
        success_symbols=None,  # noqa: ANN001
        success_rows=None,  # noqa: ANN001
        success_start_ts=None,  # noqa: ANN001
        success_end_ts=None,  # noqa: ANN001
        error_groups=None,  # noqa: ANN001
        updated_at=None,  # noqa: ANN001,ARG002
    ) -> None:
        self.apply_calls += 1
        if bars_df is not None and not bars_df.empty:
            self.upsert_bars(bars_df, interval=interval, provider=provider)
        if success_symbols:
            self.mark_meta_success(
                success_symbols,
                interval=interval,
                provider=provider,
                rows=success_rows,
                start_ts=success_start_ts,
                end_ts=success_end_ts,
            )
        for symbols, status, error_text in error_groups or []:
            self.mark_meta_error(
                symbols,
                interval=interval,
                provider=provider,
                status=status,
                error=error_text,
            )


class _FakeStoreWithBulkCoverage(_FakeStore):
    def __init__(self) -> None:
        super().__init__()
        self.coverage_calls = 0
        self.bulk_calls = 0

    def coverage(
        self,
        contract_symbol: str,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
    ) -> dict[str, Any]:
        del contract_symbol
        self.coverage_calls += 1
        return {}

    def coverage_bulk(
        self,
        contract_symbols,
        *,
        interval: str = "1d",  # noqa: ARG002
        provider: str,  # noqa: ARG002
    ) -> dict[str, dict[str, Any]]:
        self.bulk_calls += 1
        return {str(sym).strip().upper(): {} for sym in contract_symbols}


class _AlwaysErrorClient(_FakeClient):
    def get_option_bars_daily_full(
        self,
        symbols: list[str],
        *,
        start,  # noqa: ANN001
        end,  # noqa: ANN001
        interval: str = "1d",  # noqa: ARG002
        feed: str | None = None,  # noqa: ARG002
        chunk_size: int = 200,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
        page_limit: int | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        self.call_starts.append(time.monotonic())
        self.calls.append({"symbols": list(symbols), "start": start, "end": end})
        raise RuntimeError("boom")


def _contracts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contractSymbol": [
                "AAA260117C00100000",
                "BBB260117C00100000",
                "CCC260117P00100000",
            ],
            "expiry_date": [date(2026, 1, 17), date(2026, 1, 17), date(2026, 1, 17)],
        }
    )


def test_backfill_option_bars_parallel_records_success() -> None:
    client = _FakeClient(sleep_seconds=0.05)
    store = _FakeStore()

    summary = backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=3,
        bars_max_requests_per_second=None,
        resume=False,
        dry_run=False,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert summary.requests_attempted == 3
    assert summary.ok_contracts == 3
    assert summary.error_contracts == 0
    assert set(store.success_symbols) == {
        "AAA260117C00100000",
        "BBB260117C00100000",
        "CCC260117P00100000",
    }
    assert store.error_symbols == []
    assert len(client.calls) == 3
    assert all(len(call["symbols"]) == 1 for call in client.calls)


def test_backfill_option_bars_throttle_limits_request_rate() -> None:
    client = _FakeClient(sleep_seconds=0.0)
    store = _FakeStore()

    backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=3,
        bars_max_requests_per_second=5.0,
        resume=False,
        dry_run=False,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert len(client.call_starts) == 3
    elapsed = max(client.call_starts) - min(client.call_starts)
    assert elapsed >= 0.25


def test_backfill_option_bars_uses_bulk_coverage_when_available() -> None:
    client = _FakeClient(sleep_seconds=0.0)
    store = _FakeStoreWithBulkCoverage()

    summary = backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=1,
        bars_max_requests_per_second=None,
        resume=True,
        dry_run=True,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert summary.requests_attempted == 3
    assert store.bulk_calls == 1
    assert store.coverage_calls == 0


def test_backfill_option_bars_batches_writes() -> None:
    client = _FakeClient(sleep_seconds=0.0)
    store = _FakeStoreWithBatchCounters()

    summary = backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=1,
        bars_max_requests_per_second=None,
        bars_write_batch_size=2,
        resume=False,
        dry_run=False,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert summary.requests_attempted == 3
    assert summary.ok_contracts == 3
    assert store.upsert_calls == 2
    assert store.success_calls == 2


def test_backfill_option_bars_batches_error_meta_writes() -> None:
    client = _AlwaysErrorClient(sleep_seconds=0.0)
    store = _FakeStoreWithBatchCounters()

    summary = backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=1,
        bars_max_requests_per_second=None,
        bars_write_batch_size=2,
        resume=False,
        dry_run=False,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert summary.requests_attempted == 3
    assert summary.ok_contracts == 0
    assert summary.error_contracts == 3
    assert store.upsert_calls == 0
    assert store.success_calls == 0
    assert store.error_calls == 2
    assert set(store.error_symbols) == {
        "AAA260117C00100000",
        "BBB260117C00100000",
        "CCC260117P00100000",
    }


def test_backfill_option_bars_uses_apply_write_batch_when_available() -> None:
    client = _FakeClient(sleep_seconds=0.0)
    store = _FakeStoreWithApplyBatch()

    summary = backfill_option_bars(
        client,  # type: ignore[arg-type]
        store,  # type: ignore[arg-type]
        _contracts_frame(),
        provider="alpaca",
        lookback_years=1,
        bars_concurrency=1,
        bars_max_requests_per_second=None,
        bars_write_batch_size=2,
        resume=False,
        dry_run=False,
        fail_fast=False,
        today=date(2026, 1, 20),
    )

    assert summary.requests_attempted == 3
    assert summary.ok_contracts == 3
    assert summary.error_contracts == 0
    assert store.apply_calls == 2
    assert store.upsert_calls == 2
    assert store.success_calls == 2
