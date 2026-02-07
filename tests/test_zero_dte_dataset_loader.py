from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.zero_dte_dataset import ZeroDTEIntradayDatasetLoader, build_us_equity_session
from options_helper.db.warehouse import DuckDBWarehouse


def _bars_frame(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [500.0 + idx for idx in range(len(timestamps))],
            "high": [500.1 + idx for idx in range(len(timestamps))],
            "low": [499.9 + idx for idx in range(len(timestamps))],
            "close": [500.05 + idx for idx in range(len(timestamps))],
            "volume": [1000 + idx for idx in range(len(timestamps))],
        }
    )


def test_zero_dte_loader_handles_missing_files_and_tables(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path / "intraday")
    warehouse = DuckDBWarehouse(tmp_path / "missing_tables.duckdb")
    loader = ZeroDTEIntradayDatasetLoader(intraday_store=store, warehouse=warehouse)

    dataset = loader.load_day(
        date(2026, 2, 3),
        decision_times=["10:30"],
        include_option_snapshot=True,
        include_option_bars=True,
        option_contract_symbols=["SPY260320P00500000"],
    )

    assert dataset.underlying_bars.empty
    assert len(dataset.state_rows) == 1
    assert str(dataset.state_rows.iloc[0]["status"]) == "no_underlying_data"
    assert dataset.option_snapshot.empty
    assert dataset.option_bars.empty
    assert any("Option snapshot store unavailable" in note for note in dataset.notes)
    assert any("DuckDB table missing: option_bars" in note for note in dataset.notes)


def test_zero_dte_loader_normalizes_timezone_and_builds_snapshots(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path / "intraday")
    target_day = date(2026, 2, 3)
    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "SPY",
        target_day,
        _bars_frame(["2026-02-03T14:30:00Z", "2026-02-03T14:31:00Z"]),
    )

    loader = ZeroDTEIntradayDatasetLoader(intraday_store=store)
    dataset = loader.load_day(target_day, decision_times=["09:30", "09:31"])

    first_market_ts = dataset.underlying_bars.iloc[0]["timestamp_market"]
    assert first_market_ts.strftime("%Y-%m-%d %H:%M %z") == "2026-02-03 09:30 -0500"

    assert list(dataset.state_rows["status"].astype(str)) == ["ok", "ok"]
    second_bar_market_ts = dataset.state_rows.iloc[1]["bar_ts_market"]
    assert second_bar_market_ts.strftime("%Y-%m-%d %H:%M %z") == "2026-02-03 09:31 -0500"


def test_zero_dte_loader_respects_half_day_sessions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path / "intraday")
    half_day = date(2026, 11, 27)  # Day after Thanksgiving.
    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "SPY",
        half_day,
        _bars_frame(["2026-11-27T14:30:00Z", "2026-11-27T17:59:00Z"]),
    )

    loader = ZeroDTEIntradayDatasetLoader(intraday_store=store)
    dataset = loader.load_day(half_day, decision_times=["12:59", "13:30"])

    assert dataset.session.is_half_day is True
    assert dataset.session.market_close is not None
    assert dataset.session.market_close.strftime("%H:%M") == "13:00"
    assert list(dataset.state_rows["status"].astype(str)) == ["ok", "outside_session"]


def test_zero_dte_loader_handles_dst_boundary_sessions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = IntradayStore(tmp_path / "intraday")
    loader = ZeroDTEIntradayDatasetLoader(intraday_store=store)

    # First trading day after DST starts (UTC-4).
    spring_day = date(2026, 3, 9)
    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "SPY",
        spring_day,
        _bars_frame(["2026-03-09T13:30:00Z"]),
    )

    # First trading day after DST ends (UTC-5).
    fall_day = date(2026, 11, 2)
    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "SPY",
        fall_day,
        _bars_frame(["2026-11-02T14:30:00Z"]),
    )

    spring_dataset = loader.load_day(spring_day, decision_times=["09:30"])
    fall_dataset = loader.load_day(fall_day, decision_times=["09:30"])

    spring_market_ts = spring_dataset.underlying_bars.iloc[0]["timestamp_market"]
    fall_market_ts = fall_dataset.underlying_bars.iloc[0]["timestamp_market"]

    assert spring_market_ts.strftime("%Y-%m-%d %H:%M %z") == "2026-03-09 09:30 -0400"
    assert fall_market_ts.strftime("%Y-%m-%d %H:%M %z") == "2026-11-02 09:30 -0500"

    spring_session = build_us_equity_session(spring_day, market_tz=loader.market_tz)
    fall_session = build_us_equity_session(fall_day, market_tz=loader.market_tz)

    assert spring_session.market_open is not None
    assert fall_session.market_open is not None
    assert spring_session.market_open.astimezone(loader.market_tz).strftime("%H:%M") == "09:30"
    assert fall_session.market_open.astimezone(loader.market_tz).strftime("%H:%M") == "09:30"
