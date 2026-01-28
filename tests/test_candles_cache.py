from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_helper.data.candles import CandleStore


def _df(d0: date, days: int) -> pd.DataFrame:
    idx = pd.to_datetime([d0 + timedelta(days=i) for i in range(days)])
    return pd.DataFrame(
        {
            "Open": range(days),
            "High": range(days),
            "Low": range(days),
            "Close": range(days),
            "Volume": range(days),
        },
        index=idx,
    )


def test_initial_download_writes_cache(tmp_path) -> None:
    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        assert start is not None
        return _df(start, 5)

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0)
    today = date(2026, 1, 27)
    out = store.get_daily_history("UROY", period="10d", today=today)

    assert len(calls) == 1
    sym, start, end = calls[0]
    assert sym == "UROY"
    assert end is None
    assert start == today - timedelta(days=10)
    assert not out.empty
    assert (tmp_path / "UROY.csv").exists()


def test_refresh_tail_fetches_recent_window(tmp_path) -> None:
    # Seed cache with a few days of data.
    cached_start = date(2026, 1, 10)
    cached = _df(cached_start, 10)  # last cached date = 1/19

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        assert start is not None
        # Return overlapping + newer data.
        return _df(start, 15)

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=5)
    store.save("UROY", cached)

    out = store.get_daily_history("UROY", period="10d", today=date(2026, 1, 27))

    assert len(calls) == 1
    _, start, end = calls[0]
    assert end is None
    assert start == date(2026, 1, 19) - timedelta(days=5)
    assert out.index.max().date() == (start + timedelta(days=14))


def test_backfill_earlier_history_when_period_expands(tmp_path) -> None:
    cached_start = date(2026, 1, 20)
    cached = _df(cached_start, 5)

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        assert start is not None
        assert end is not None
        return _df(start, (end - start).days)

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0)
    store.save("UROY", cached)

    today = date(2026, 1, 27)
    out = store.get_daily_history("UROY", period="20d", today=today)

    assert len(calls) == 1
    _, start, end = calls[0]
    assert start == today - timedelta(days=20)
    assert end == cached_start
    assert out.index.min().date() == start


def test_load_normalizes_mixed_tz_index(tmp_path) -> None:
    # Simulate a cached CSV with mixed tz offsets in the index.
    path = tmp_path / "UROY.csv"
    path.write_text(
        "Date,Open,High,Low,Close,Volume\n"
        "2026-01-01 00:00:00-05:00,1,1,1,1,1\n"
        "2026-01-02 00:00:00+00:00,2,2,2,2,2\n",
        encoding="utf-8",
    )

    store = CandleStore(tmp_path, backfill_days=0)
    df = store.load("UROY")
    assert isinstance(df.index, pd.DatetimeIndex)
    # Should be tz-naive after normalization.
    assert df.index.tz is None
    # Resampling should work.
    _ = df["Close"].resample("W-FRI").last()
