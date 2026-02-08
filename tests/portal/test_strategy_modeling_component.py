from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.strategy_modeling_io import (
    IntradayCoverageBySymbol,
    StrategyModelingDailyLoadResult,
    StrategyModelingIntradayPreflightResult,
    StrategyModelingUniverseLoadResult,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_candles(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO candles_daily(
              symbol, interval, auto_adjust, back_adjust, ts,
              open, high, low, close, volume, vwap, trade_count
            ) VALUES
              ('AAPL', '1d', TRUE, FALSE, '2026-01-02 00:00:00', 100, 102, 99, 101, 1000000, 100.8, 100),
              ('AAPL', '1d', TRUE, FALSE, '2026-01-05 00:00:00', 101, 103, 100, 102, 1100000, 101.7, 110),
              ('MSFT', '1d', TRUE, FALSE, '2026-01-05 00:00:00', 200, 202, 198, 201, 900000, 200.5, 90)
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
            "volume": [1000 + i for i in range(rows)],
            "vwap": [100.05 + i for i in range(rows)],
            "trade_count": [100 + i for i in range(rows)],
        }
    )


def _clear_caches(page: object) -> None:
    page._list_strategy_modeling_symbols_cached.clear()
    page._load_strategy_modeling_payload_cached.clear()


def test_strategy_modeling_component_payload_shape_and_success(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import strategy_modeling_page as page

    db_path = tmp_path / "portal.duckdb"
    intraday_dir = tmp_path / "intraday"
    _seed_candles(db_path)

    store = IntradayStore(intraday_dir)
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 2), _intraday_df("2026-01-02 14:30:00+00:00"))
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 5), _intraday_df("2026-01-05 14:30:00+00:00"))

    _clear_caches(page)

    symbols, notes = page.list_strategy_modeling_symbols(database_path=db_path)
    assert notes == []
    assert symbols == ["AAPL", "MSFT"]

    payload = page.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=True,
    )

    expected_keys = {
        "status",
        "errors",
        "notes",
        "database_path",
        "intraday_dir",
        "start_date",
        "end_date",
        "intraday_timeframe",
        "require_intraday_bars",
        "adjusted_data_fallback_mode",
        "universe_symbols",
        "requested_symbols",
        "modeled_symbols",
        "skipped_symbols",
        "missing_symbols",
        "source_by_symbol",
        "daily_rows_by_symbol",
        "required_sessions_by_symbol",
        "intraday_preflight",
        "blocking",
        "has_data",
    }
    assert set(payload) == expected_keys
    assert payload["status"] == "ok"
    assert payload["errors"] == []
    assert payload["has_data"] is True
    assert payload["requested_symbols"] == ["AAPL"]
    assert payload["modeled_symbols"] == ["AAPL"]
    assert payload["required_sessions_by_symbol"]["AAPL"] == ["2026-01-02", "2026-01-05"]

    preflight = payload["intraday_preflight"]
    assert preflight["is_blocked"] is False
    assert preflight["blocked_symbols"] == []

    blocking = payload["blocking"]
    assert blocking["is_blocked"] is False
    assert blocking["reason"] is None
    assert blocking["missing_sessions_total"] == 0
    assert blocking["coverage_rows"][0]["symbol"] == "AAPL"


def test_strategy_modeling_component_handles_missing_db_and_table(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import strategy_modeling_page as page

    missing_db = tmp_path / "missing.duckdb"
    _clear_caches(page)

    symbols, notes = page.list_strategy_modeling_symbols(database_path=missing_db)
    assert symbols == []
    assert any("not found" in note.lower() for note in notes)

    missing_payload = page.load_strategy_modeling_data_payload(database_path=missing_db)
    assert missing_payload["status"] == "error"
    assert missing_payload["blocking"]["is_blocked"] is True
    assert any("not found" in item.lower() for item in missing_payload["errors"])

    tableless_db = tmp_path / "tableless.duckdb"
    conn = duckdb.connect(str(tableless_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
    finally:
        conn.close()

    _clear_caches(page)
    table_payload = page.load_strategy_modeling_data_payload(database_path=tableless_db)
    assert table_payload["status"] == "error"
    assert table_payload["blocking"]["is_blocked"] is True
    assert any("candles_daily table not found" in item for item in table_payload["errors"])


def test_strategy_modeling_component_handles_invalid_filters_and_fallback(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import strategy_modeling_page as page

    db_path = tmp_path / "portal.duckdb"
    _seed_candles(db_path)
    _clear_caches(page)

    invalid_dates = page.load_strategy_modeling_data_payload(
        database_path=db_path,
        start_date=date(2026, 1, 10),
        end_date=date(2026, 1, 5),
    )
    assert invalid_dates["status"] == "error"
    assert invalid_dates["blocking"]["is_blocked"] is True
    assert any("invalid date range" in item.lower() for item in invalid_dates["errors"])

    _clear_caches(page)
    invalid_symbols = page.load_strategy_modeling_data_payload(
        database_path=db_path,
        symbols=["XXXX"],
    )
    assert invalid_symbols["status"] == "error"
    assert invalid_symbols["blocking"]["is_blocked"] is True
    assert any("excluded all available symbols" in item.lower() for item in invalid_symbols["errors"])

    _clear_caches(page)
    invalid_fallback = page.load_strategy_modeling_data_payload(
        database_path=db_path,
        adjusted_data_fallback_mode="invalid",  # type: ignore[arg-type]
    )
    assert invalid_fallback["status"] == "error"
    assert any("invalid adjusted_data_fallback_mode" in item.lower() for item in invalid_fallback["errors"])


def test_strategy_modeling_component_intraday_preflight_blocking(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import strategy_modeling_page as page

    db_path = tmp_path / "portal.duckdb"
    intraday_dir = tmp_path / "intraday"
    _seed_candles(db_path)

    store = IntradayStore(intraday_dir)
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 2), _intraday_df("2026-01-02 14:30:00+00:00"))

    _clear_caches(page)
    blocked = page.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=True,
    )
    assert blocked["status"] == "ok"
    assert blocked["blocking"]["is_blocked"] is True
    assert blocked["blocking"]["reason"] == "intraday_coverage_missing"
    assert blocked["blocking"]["blocked_symbols"] == ["AAPL"]
    assert blocked["blocking"]["missing_sessions_total"] == 1

    _clear_caches(page)
    permissive = page.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=False,
    )
    assert permissive["status"] == "ok"
    assert permissive["blocking"]["is_blocked"] is False
    assert permissive["blocking"]["reason"] is None
    assert permissive["blocking"]["missing_sessions_total"] == 1


def test_strategy_modeling_component_uses_streamlit_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import strategy_modeling_page as page

    calls = {"universe": 0, "daily": 0, "sessions": 0, "preflight": 0}

    def _stub_universe(*, database_path: str | Path | None = None) -> StrategyModelingUniverseLoadResult:
        calls["universe"] += 1
        resolved = Path(database_path) if database_path is not None else Path("/tmp/stub.duckdb")
        return StrategyModelingUniverseLoadResult(
            symbols=["SPY"],
            notes=[],
            database_path=resolved,
            database_exists=True,
        )

    def _stub_daily(
        symbols: list[str],
        *,
        database_path: str | Path | None = None,
        **_: object,
    ) -> StrategyModelingDailyLoadResult:
        calls["daily"] += 1
        frame = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2026-01-02"]),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
            }
        )
        return StrategyModelingDailyLoadResult(
            candles_by_symbol={"SPY": frame},
            source_by_symbol={"SPY": "adjusted"},
            skipped_symbols=[],
            missing_symbols=[symbol for symbol in symbols if symbol != "SPY"],
            notes=[],
        )

    def _stub_sessions(
        daily_candles_by_symbol: dict[str, pd.DataFrame],
        *,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, list[date]]:
        _ = (daily_candles_by_symbol, start_date, end_date)
        calls["sessions"] += 1
        return {"SPY": [date(2026, 1, 2)]}

    def _stub_preflight(
        required_sessions_by_symbol: dict[str, list[date]],
        *,
        intraday_dir: Path = Path("data/intraday"),
        timeframe: str = "1Min",
        require_intraday_bars: bool = True,
    ) -> StrategyModelingIntradayPreflightResult:
        _ = (required_sessions_by_symbol, intraday_dir, timeframe)
        calls["preflight"] += 1
        coverage = IntradayCoverageBySymbol(
            symbol="SPY",
            required_days=(date(2026, 1, 2),),
            covered_days=(date(2026, 1, 2),),
            missing_days=(),
        )
        return StrategyModelingIntradayPreflightResult(
            require_intraday_bars=require_intraday_bars,
            coverage_by_symbol={"SPY": coverage},
            blocked_symbols=[],
            notes=[],
        )

    monkeypatch.setattr(page, "list_strategy_modeling_universe", _stub_universe)
    monkeypatch.setattr(page, "load_daily_ohlc_history", _stub_daily)
    monkeypatch.setattr(page, "build_required_intraday_sessions", _stub_sessions)
    monkeypatch.setattr(page, "preflight_intraday_coverage", _stub_preflight)

    _clear_caches(page)

    kwargs = {
        "symbols": ["SPY"],
        "database_path": tmp_path / "cache.duckdb",
        "intraday_dir": tmp_path / "intraday",
    }
    payload_1 = page.load_strategy_modeling_data_payload(**kwargs)
    payload_2 = page.load_strategy_modeling_data_payload(**kwargs)

    assert payload_1 == payload_2
    assert calls == {"universe": 1, "daily": 1, "sessions": 1, "preflight": 1}

    # list_strategy_modeling_symbols has its own cache seam.
    _clear_caches(page)
    symbols_1, _ = page.list_strategy_modeling_symbols(database_path=tmp_path / "cache.duckdb")
    symbols_2, _ = page.list_strategy_modeling_symbols(database_path=tmp_path / "cache.duckdb")
    assert symbols_1 == symbols_2 == ["SPY"]
    assert calls["universe"] == 2
