from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_helper.data.candles import CandleStore
from options_helper.data.market_types import DataFetchError


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


def test_provider_history_fetch_used(tmp_path) -> None:
    calls: list[tuple[str, date | None, date | None, str, bool, bool]] = []

    class StubProvider:
        def get_history(
            self,
            symbol: str,
            *,
            start: date | None,
            end: date | None,
            interval: str,
            auto_adjust: bool,
            back_adjust: bool,
        ) -> pd.DataFrame:
            calls.append((symbol, start, end, interval, auto_adjust, back_adjust))
            assert start is not None
            return _df(start, 5)

    store = CandleStore(tmp_path, provider=StubProvider(), backfill_days=0)
    today = date(2026, 1, 27)
    out = store.get_daily_history("UROY", period="5d", today=today)

    assert len(calls) == 1
    sym, start, end, interval, auto_adjust, back_adjust = calls[0]
    assert sym == "UROY"
    assert end is None
    assert interval == "1d"
    assert auto_adjust is True
    assert back_adjust is False
    assert start == today - timedelta(days=5)
    assert not out.empty


def test_rate_limit_detection_unwraps_exception_chain(tmp_path) -> None:
    store = CandleStore(tmp_path, backfill_days=0)

    class YFRateLimitError(Exception):
        pass

    inner = YFRateLimitError("429 Too Many Requests")
    try:
        raise DataFetchError("Failed to fetch") from inner
    except DataFetchError as exc:
        assert store._is_rate_limit_error(exc)


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


def test_period_max_backfills_earlier_history_when_cache_exists(tmp_path) -> None:
    cached_start = date(2026, 1, 20)
    cached = _df(cached_start, 5)

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        assert start is not None
        assert end is not None
        # get_daily_history uses a far-past (epoch-safe) start for "max" backfills; keep the stub lightweight.
        assert start == date(1970, 1, 1)
        assert end == cached_start
        return _df(date(2026, 1, 1), 10)

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0)
    store.save("UROY", cached)

    out = store.get_daily_history("UROY", period="max", today=date(2026, 1, 27))

    assert len(calls) == 1
    assert out.index.min().date() == date(2026, 1, 1)
    assert out.index.max().date() == (cached_start + timedelta(days=4))


def test_period_max_backfill_runs_once_then_uses_meta_flag(tmp_path) -> None:
    cached_start = date(2026, 1, 20)
    cached = _df(cached_start, 5)

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        assert start == date(1970, 1, 1)
        assert end == cached_start
        return _df(date(2026, 1, 1), 10)

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0)
    store.save("UROY", cached)

    first = store.get_daily_history("UROY", period="max", today=date(2026, 1, 27))
    second = store.get_daily_history("UROY", period="max", today=date(2026, 1, 28))

    assert len(calls) == 1
    assert first.index.min().date() == date(2026, 1, 1)
    assert second.index.min().date() == date(2026, 1, 1)


def test_period_max_pre_1970_cache_skips_invalid_backfill_range(tmp_path) -> None:
    cached_start = date(1962, 1, 2)
    cached = _df(cached_start, 5)

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:
        calls.append((symbol, start, end))
        return pd.DataFrame()

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0)
    store.save("CVX", cached)

    out = store.get_daily_history("CVX", period="max", today=date(2026, 2, 7))

    # Existing cache already predates _MAX_BACKFILL_START (1970-01-01), so skip
    # bounded backfill instead of emitting an invalid start>end request.
    assert calls == []
    assert out.index.min().date() == cached_start
    meta = store.load_meta("CVX") or {}
    assert bool(meta.get("max_backfill_complete")) is True


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


def test_legacy_cache_upgrades_to_adjusted_without_fetch(tmp_path) -> None:
    # Legacy cache: CSV exists but no meta.json, and it contains Adj Close.
    idx = pd.to_datetime([date(2026, 1, 1), date(2026, 1, 2)])
    legacy = pd.DataFrame(
        {
            "Open": [100.0, 110.0],
            "High": [101.0, 111.0],
            "Low": [99.0, 109.0],
            "Close": [100.0, 110.0],
            "Adj Close": [50.0, 55.0],
            "Volume": [1, 2],
        },
        index=idx,
    )
    legacy.to_csv(tmp_path / "UROY.csv")

    calls: list[tuple[str, date | None, date | None]] = []

    def fetcher(symbol: str, start: date | None, end: date | None) -> pd.DataFrame:  # pragma: no cover
        calls.append((symbol, start, end))
        return pd.DataFrame()

    store = CandleStore(tmp_path, fetcher=fetcher, backfill_days=0, auto_adjust=True, back_adjust=False)
    out = store.get_daily_history("UROY", period="10d", today=date(2026, 1, 27))

    assert calls == []
    assert not out.empty
    assert "Adj Close" not in out.columns
    assert out["Close"].iloc[0] == 50.0
    assert (tmp_path / "UROY.meta.json").exists()
