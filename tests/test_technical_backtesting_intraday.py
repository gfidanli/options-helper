from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.technical_backtesting_intraday import load_intraday_candles


def _bars(rows: list[tuple[str, float, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )


def _save_partition(
    store: IntradayStore,
    *,
    symbol: str,
    day: date,
    rows: list[tuple[str, float, float, float, float, float]],
) -> None:
    store.save_partition("stocks", "bars", "1Min", symbol, day, _bars(rows))


def test_load_intraday_candles_resamples_ohlcv_deterministically(tmp_path: Path) -> None:
    store = IntradayStore(tmp_path / "intraday")
    day = date(2026, 2, 3)
    _save_partition(
        store,
        symbol="AAPL",
        day=day,
        rows=[
            ("2026-02-03T14:30:00Z", 10.0, 12.0, 9.0, 11.0, 100.0),
            ("2026-02-03T14:31:00Z", 11.0, 13.0, 10.0, 12.0, 150.0),
            ("2026-02-03T14:35:00Z", 12.0, 14.0, 11.0, 13.0, 50.0),
        ],
    )

    first = load_intraday_candles(
        symbol="AAPL",
        start_day=day,
        end_day=day,
        base_timeframe="1Min",
        target_interval="5Min",
        store=store,
    )
    second = load_intraday_candles(
        symbol="AAPL",
        start_day=day,
        end_day=day,
        base_timeframe="1Min",
        target_interval="5Min",
        store=store,
    )

    expected = pd.DataFrame(
        {
            "Open": [10.0, 12.0],
            "High": [13.0, 14.0],
            "Low": [9.0, 11.0],
            "Close": [12.0, 13.0],
            "Volume": [250.0, 50.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2026-02-03T14:30:00"),
                pd.Timestamp("2026-02-03T14:35:00"),
            ]
        ),
    )
    expected.index.name = "timestamp"

    pd.testing.assert_frame_equal(first.candles, expected, check_freq=False)
    pd.testing.assert_frame_equal(first.candles, second.candles, check_freq=False)
    assert first.candles.index.tz is None
    assert first.coverage.loaded_day_count == 1
    assert first.coverage.output_row_count == 2


def test_load_intraday_candles_warns_and_tracks_missing_and_empty_days(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = IntradayStore(tmp_path / "intraday")
    day_1 = date(2026, 2, 3)
    day_2 = date(2026, 2, 4)
    day_3 = date(2026, 2, 5)
    _save_partition(
        store,
        symbol="AAPL",
        day=day_1,
        rows=[("2026-02-03T14:30:00Z", 10.0, 10.5, 9.8, 10.2, 100.0)],
    )
    store.save_partition("stocks", "bars", "1Min", "AAPL", day_2, pd.DataFrame())

    caplog.set_level("WARNING", logger="options_helper.data.technical_backtesting_intraday")
    result = load_intraday_candles(
        symbol="AAPL",
        start_day=day_1,
        end_day=day_3,
        base_timeframe="1Min",
        target_interval="1Min",
        store=store,
    )

    coverage = result.coverage
    assert coverage.requested_day_count == 3
    assert coverage.loaded_day_count == 1
    assert coverage.empty_day_count == 1
    assert coverage.missing_day_count == 1
    assert coverage.loaded_days == (day_1,)
    assert coverage.empty_days == (day_2,)
    assert coverage.missing_days == (day_3,)
    assert any("Skipping empty intraday partition" in msg for msg in caplog.messages)
    assert any("Missing intraday partition" in msg for msg in caplog.messages)
    assert len(result.candles.index) == 1


def test_load_intraday_candles_rejects_target_interval_smaller_than_base(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be >= base_timeframe"):
        load_intraday_candles(
            symbol="AAPL",
            start_day=date(2026, 2, 3),
            end_day=date(2026, 2, 3),
            base_timeframe="5Min",
            target_interval="1Min",
            intraday_dir=tmp_path / "intraday",
        )


def test_load_intraday_candles_rejects_non_multiple_target_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        load_intraday_candles(
            symbol="AAPL",
            start_day=date(2026, 2, 3),
            end_day=date(2026, 2, 3),
            base_timeframe="5Min",
            target_interval="12Min",
            intraday_dir=tmp_path / "intraday",
        )
