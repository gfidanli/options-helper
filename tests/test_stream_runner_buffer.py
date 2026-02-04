from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.streaming.normalizers import NormalizedEvent
from options_helper.data.streaming.runner import CaptureBuffer, compute_backoff_seconds


def test_compute_backoff_seconds_caps() -> None:
    assert compute_backoff_seconds(0, base_seconds=1.0, cap_seconds=3.0) == 0.0
    assert compute_backoff_seconds(1, base_seconds=1.0, cap_seconds=3.0) == 1.0
    assert compute_backoff_seconds(2, base_seconds=1.0, cap_seconds=3.0) == 2.0
    assert compute_backoff_seconds(3, base_seconds=1.0, cap_seconds=3.0) == 3.0
    assert compute_backoff_seconds(4, base_seconds=1.0, cap_seconds=3.0) == 3.0


def test_capture_buffer_partitions_by_market_day(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    market_tz = ZoneInfo("America/New_York")
    buffer = CaptureBuffer(store, market_tz=market_tz, base_meta={"provider": "alpaca"})

    # 2026-02-04T01:00Z is 2026-02-03 20:00 in New York -> day=2026-02-03
    ts1 = datetime(2026, 2, 4, 1, 0, tzinfo=timezone.utc)
    # 2026-02-04T05:00Z is 2026-02-04 00:00 in New York -> day=2026-02-04
    ts2 = datetime(2026, 2, 4, 5, 0, tzinfo=timezone.utc)

    buffer.ingest(NormalizedEvent(dataset="stock_trades", symbol="AAPL", row={"timestamp": ts1, "price": 1.0}))
    buffer.ingest(NormalizedEvent(dataset="stock_trades", symbol="AAPL", row={"timestamp": ts2, "price": 2.0}))

    assert buffer.writer_count == 2
    buffer.flush_all()

    p1 = store.partition_path("stocks", "trades", "tick", "AAPL", date(2026, 2, 3))
    p2 = store.partition_path("stocks", "trades", "tick", "AAPL", date(2026, 2, 4))
    assert p1.exists()
    assert p2.exists()


def test_capture_buffer_writes_option_quotes(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path)
    buffer = CaptureBuffer(store, market_tz=ZoneInfo("America/New_York"))
    contract = "AAPL260320C00150000"
    ts = datetime(2026, 2, 3, 14, 30, tzinfo=timezone.utc)
    buffer.ingest(
        NormalizedEvent(dataset="option_quotes", symbol=contract, row={"timestamp": ts, "bid_price": 1.0})
    )
    buffer.flush_all()

    path = store.partition_path("options", "quotes", "tick", contract, date(2026, 2, 3))
    assert path.exists()
