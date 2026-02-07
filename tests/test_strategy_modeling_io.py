from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.strategy_modeling_io import (
    build_required_intraday_sessions,
    list_strategy_modeling_universe,
    load_daily_ohlc_history,
    load_required_intraday_bars,
    preflight_intraday_coverage,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_candles_db(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO candles_daily(
              symbol, interval, auto_adjust, back_adjust, ts,
              open, high, low, close, volume, vwap, trade_count
            ) VALUES
              ('AAPL', '1d', TRUE, FALSE, '2026-01-02 00:00:00', 100, 103, 99, 102, 1000000, 101.2, 12000),
              ('AAPL', '1d', TRUE, FALSE, '2026-01-05 00:00:00', 102, 104, 101, 103, 1200000, 102.5, 13000),
              ('MSFT', '1d', FALSE, FALSE, '2026-01-02 00:00:00', 200, 203, 199, 201, 900000, 200.8, 10000)
            """
        )


def _intraday_df(start_ts: str, *, rows: int = 2) -> pd.DataFrame:
    ts = pd.date_range(start_ts, periods=rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + i for i in range(rows)],
            "high": [100.2 + i for i in range(rows)],
            "low": [99.8 + i for i in range(rows)],
            "close": [100.1 + i for i in range(rows)],
            "volume": [1_000 + i for i in range(rows)],
            "vwap": [100.05 + i for i in range(rows)],
            "trade_count": [100 + i for i in range(rows)],
        }
    )


def test_list_strategy_modeling_universe_success(tmp_path: Path) -> None:
    db_path = tmp_path / "options.duckdb"
    _seed_candles_db(db_path)

    result = list_strategy_modeling_universe(database_path=db_path)

    assert result.database_exists is True
    assert result.symbols == ["AAPL", "MSFT"]
    assert result.notes == []


def test_list_strategy_modeling_universe_missing_table(tmp_path: Path) -> None:
    db_path = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
        conn.execute("INSERT INTO sample VALUES (1)")
    finally:
        conn.close()

    result = list_strategy_modeling_universe(database_path=db_path)

    assert result.symbols == []
    assert result.notes
    assert "candles_daily table not found" in result.notes[0]


def test_load_daily_ohlc_history_warn_and_skip_unadjusted_only_symbol(tmp_path: Path) -> None:
    db_path = tmp_path / "options.duckdb"
    _seed_candles_db(db_path)

    result = load_daily_ohlc_history(
        ["AAPL", "MSFT", "TSLA"],
        database_path=db_path,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
    )

    assert sorted(result.candles_by_symbol) == ["AAPL"]
    assert result.source_by_symbol == {"AAPL": "adjusted"}
    assert result.skipped_symbols == ["MSFT"]
    assert result.missing_symbols == ["TSLA"]
    assert any("MSFT: adjusted candles unavailable" in note for note in result.notes)
    assert any("TSLA: no daily candles found" in note for note in result.notes)


def test_load_daily_ohlc_history_explicit_unadjusted_fallback(tmp_path: Path) -> None:
    db_path = tmp_path / "options.duckdb"
    _seed_candles_db(db_path)

    result = load_daily_ohlc_history(
        ["MSFT"],
        database_path=db_path,
        adjusted_data_fallback_mode="use_unadjusted_ohlc",
    )

    assert sorted(result.candles_by_symbol) == ["MSFT"]
    assert result.source_by_symbol == {"MSFT": "unadjusted"}
    assert result.skipped_symbols == []
    assert result.missing_symbols == []
    assert any("using unadjusted fallback" in note for note in result.notes)


def test_preflight_intraday_coverage_blocks_when_required(tmp_path: Path) -> None:
    intraday_dir = tmp_path / "intraday"
    store = IntradayStore(intraday_dir)
    day_1 = date(2026, 1, 2)
    day_2 = date(2026, 1, 5)

    store.save_partition("stocks", "bars", "1Min", "AAPL", day_1, _intraday_df("2026-01-02 14:30:00+00:00"))

    preflight = preflight_intraday_coverage(
        {"AAPL": [day_1, day_2]},
        intraday_dir=intraday_dir,
        timeframe="1Min",
        require_intraday_bars=True,
    )

    assert preflight.is_blocked is True
    assert preflight.blocked_symbols == ["AAPL"]
    coverage = preflight.coverage_by_symbol["AAPL"]
    assert coverage.covered_days == (day_1,)
    assert coverage.missing_days == (day_2,)



def test_load_required_intraday_bars_honors_policy_gate(tmp_path: Path) -> None:
    intraday_dir = tmp_path / "intraday"
    store = IntradayStore(intraday_dir)
    day_1 = date(2026, 1, 2)
    day_2 = date(2026, 1, 5)

    store.save_partition("stocks", "bars", "1Min", "AAPL", day_1, _intraday_df("2026-01-02 14:30:00+00:00"))
    store.save_partition("stocks", "bars", "1Min", "AAPL", day_2, _intraday_df("2026-01-05 14:30:00+00:00"))

    required = {"AAPL": [day_1, day_2]}
    loaded = load_required_intraday_bars(
        required,
        intraday_dir=intraday_dir,
        timeframe="1Min",
        require_intraday_bars=True,
    )

    assert loaded.preflight.is_blocked is False
    bars = loaded.bars_by_symbol["AAPL"]
    assert len(bars) == 4
    assert bars["timestamp"].is_monotonic_increasing
    assert {pd.Timestamp(day_1), pd.Timestamp(day_2)} == set(bars["session_date"].unique())

    gated = load_required_intraday_bars(
        {"AAPL": [day_1, date(2026, 1, 6)]},
        intraday_dir=intraday_dir,
        timeframe="1Min",
        require_intraday_bars=True,
    )
    assert gated.preflight.is_blocked is True
    assert "AAPL" not in gated.bars_by_symbol

    permissive = load_required_intraday_bars(
        {"AAPL": [day_1, date(2026, 1, 6)]},
        intraday_dir=intraday_dir,
        timeframe="1Min",
        require_intraday_bars=False,
    )
    assert permissive.preflight.is_blocked is False
    assert "AAPL" in permissive.bars_by_symbol
    assert len(permissive.bars_by_symbol["AAPL"]) == 2


def test_build_required_intraday_sessions_from_daily_frames() -> None:
    candles = {
        "AAPL": pd.DataFrame(
            {
                "ts": pd.to_datetime([
                    "2026-01-02",
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                ]),
                "open": [100.0, 101.0, 101.0, 102.0],
                "high": [101.0, 102.0, 102.0, 103.0],
                "low": [99.0, 100.0, 100.0, 101.0],
                "close": [100.5, 101.5, 101.5, 102.5],
            }
        )
    }

    sessions = build_required_intraday_sessions(
        candles,
        start_date=date(2026, 1, 3),
        end_date=date(2026, 1, 5),
    )

    assert sessions == {"AAPL": [date(2026, 1, 5)]}
