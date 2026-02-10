from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from apps.streamlit.components import strategy_modeling_trade_review as trade_review_component
from apps.streamlit.components.strategy_modeling_trade_drilldown import (
    extract_selected_rows,
    load_intraday_window,
    resample_for_chart,
    resample_ohlc,
    selected_trade_id_from_event,
    supported_chart_timeframes,
)
from apps.streamlit.components.strategy_modeling_trade_review import build_trade_review_tables
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


def test_strategy_modeling_trade_drilldown_selection_helpers() -> None:
    assert extract_selected_rows(None) == []
    assert extract_selected_rows({"selection": {"rows": [2, "1", -1, "bad", 2, True]}}) == [2, 1]
    assert extract_selected_rows(SimpleNamespace(selection=SimpleNamespace(rows=["0"]))) == [0]

    displayed = pd.DataFrame({"trade_id": ["T-001", "T-002"]})
    assert selected_trade_id_from_event({"selection": {"rows": [1]}}, displayed) == "T-002"
    assert selected_trade_id_from_event({"selection": {"rows": [3]}}, displayed) is None
    assert selected_trade_id_from_event({"selection": {"rows": [0]}}, pd.DataFrame({"id": ["T-001"]})) is None
    assert (
        selected_trade_id_from_event(
            {"selection": {"rows": [0]}},
            pd.DataFrame({"trade_id": [""]}),
        )
        is None
    )


def test_strategy_modeling_trade_drilldown_load_intraday_window(tmp_path: Path) -> None:
    intraday_dir = tmp_path / "intraday"
    store = IntradayStore(intraday_dir)

    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "AAPL",
        date(2026, 1, 2),
        pd.DataFrame(
            {
                "ts": [
                    "2026-01-02T14:31:00+00:00",
                    "2026-01-02T14:30:00+00:00",
                    "invalid",
                ],
                "open": ["101.0", "100.0", "x"],
                "high": ["101.5", "100.5", "y"],
                "low": ["100.5", "99.5", "z"],
                "close": ["101.2", "100.1", "w"],
                "volume": ["10", "8", "1"],
                "vwap": ["101.1", "100.0", "bad"],
                "trade_count": ["2", "1", "1"],
            }
        ),
    )
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 3), pd.DataFrame())
    store.save_partition(
        "stocks",
        "bars",
        "1Min",
        "AAPL",
        date(2026, 1, 4),
        pd.DataFrame(
            {
                "timestamp": ["2026-01-04T14:29:00+00:00", "2026-01-04T14:30:00+00:00"],
                "open": [200.0, 201.0],
                "high": [200.6, 201.6],
                "low": [199.6, 200.6],
                "close": [200.2, 201.2],
                "volume": [4, 5],
                "vwap": [200.1, 201.1],
                "trade_count": [1, 1],
            }
        ),
    )

    loaded = load_intraday_window(
        intraday_dir,
        "AAPL",
        "1Min",
        pd.Timestamp("2026-01-02T14:30:30+00:00"),
        pd.Timestamp("2026-01-04T14:29:30+00:00"),
    )
    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    assert len(loaded.index) == 2
    assert loaded["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02T14:31:00+00:00"),
        pd.Timestamp("2026-01-04T14:29:00+00:00"),
    ]
    assert loaded["timestamp"].dt.tz is not None
    assert loaded["open"].tolist() == [101.0, 200.0]


def test_strategy_modeling_trade_drilldown_resample_semantics() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02T14:30:00+00:00",
                    "2026-01-02T14:31:00+00:00",
                    "2026-01-02T14:35:00+00:00",
                ],
                utc=True,
            ),
            "open": [10.0, 20.0, 30.0],
            "high": [11.0, 21.0, 31.0],
            "low": [9.0, 19.0, 29.0],
            "close": [10.5, 20.5, 30.5],
            "volume": [100.0, 0.0, 0.0],
            "vwap": [10.0, 999.0, None],
            "trade_count": [2.0, 3.0, 4.0],
        }
    )

    resampled = resample_ohlc(frame, "5Min")
    assert resampled["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02T14:30:00+00:00"),
        pd.Timestamp("2026-01-02T14:35:00+00:00"),
    ]

    first = resampled.iloc[0]
    assert float(first["open"]) == pytest.approx(10.0)
    assert float(first["high"]) == pytest.approx(21.0)
    assert float(first["low"]) == pytest.approx(9.0)
    assert float(first["close"]) == pytest.approx(20.5)
    assert float(first["volume"]) == pytest.approx(100.0)
    assert float(first["trade_count"]) == pytest.approx(5.0)
    assert float(first["vwap"]) == pytest.approx(10.0)

    second = resampled.iloc[1]
    assert float(second["volume"]) == pytest.approx(0.0)
    assert float(second["trade_count"]) == pytest.approx(4.0)
    assert float(second["vwap"]) == pytest.approx(float(second["close"]))


def test_strategy_modeling_trade_drilldown_supported_timeframes() -> None:
    assert supported_chart_timeframes("1Min") == ["1Min", "5Min", "15Min", "30Min", "60Min"]
    assert supported_chart_timeframes("5Min") == ["5Min", "15Min", "30Min", "60Min"]
    assert supported_chart_timeframes("30Min") == ["30Min", "60Min"]
    assert supported_chart_timeframes("invalid") == ["1Min", "5Min", "15Min", "30Min", "60Min"]


def test_strategy_modeling_trade_drilldown_chart_guardrail_behavior() -> None:
    dense = _intraday_df("2026-01-02 14:30:00+00:00", rows=6001)
    auto = resample_for_chart(
        dense,
        base_timeframe="1Min",
        chart_timeframe="1Min",
        max_bars=5000,
    )
    assert auto.skipped is False
    assert auto.timeframe == "5Min"
    assert len(auto.bars.index) <= 5000
    assert auto.warning is not None
    assert "Auto-adjusted chart timeframe" in auto.warning

    sparse = _intraday_df("2026-01-02 14:30:00+00:00", rows=181)
    skipped = resample_for_chart(
        sparse,
        base_timeframe="60Min",
        chart_timeframe="60Min",
        max_bars=2,
    )
    assert skipped.skipped is True
    assert skipped.bars.empty
    assert skipped.warning is not None
    assert "Skipping chart" in skipped.warning


def test_strategy_modeling_trade_review_wrapper_delegates_to_shared_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_df = pd.DataFrame(
        {
            "trade_id": ["trade-1", "trade-2"],
            "symbol": ["SPY", "QQQ"],
            "status": ["closed", "closed"],
            "realized_r": [0.5, 1.5],
            "entry_ts": ["2026-01-02T14:30:00+00:00", "2026-01-03T14:30:00+00:00"],
            "reject_code": [None, None],
        },
        index=[11, 99],
    )

    calls: dict[str, object] = {}

    def _stub_rank(
        trade_rows: object,
        *,
        accepted_trade_ids: object,
        top_n: int = 20,
        metric: str = "realized_r",
    ) -> SimpleNamespace:
        calls["trade_rows"] = list(trade_rows)  # type: ignore[arg-type]
        calls["accepted_trade_ids"] = accepted_trade_ids
        calls["top_n"] = top_n
        calls["metric"] = metric
        return SimpleNamespace(
            scope="accepted_closed_trades",
            top_best_rows=(
                {
                    "realized_r": 1.5,
                    "rank": 1,
                    "trade_id": "trade-2",
                    "symbol": "QQQ",
                    "status": "closed",
                    "entry_ts": "2026-01-03T14:30:00+00:00",
                    "reject_code": None,
                    "note": "best",
                },
            ),
            top_worst_rows=(
                {
                    "rank": 1,
                    "trade_id": "trade-1",
                    "symbol": "SPY",
                    "status": "closed",
                    "realized_r": 0.5,
                    "entry_ts": "2026-01-02T14:30:00+00:00",
                    "reject_code": None,
                    "note": "worst",
                },
            ),
        )

    monkeypatch.setattr(trade_review_component, "rank_trades_for_review", _stub_rank)

    best_df, worst_df, scope_label = build_trade_review_tables(
        trade_df,
        accepted_trade_ids=("trade-2",),
        top_n=5,
    )

    assert calls == {
        "trade_rows": trade_df.to_dict(orient="records"),
        "accepted_trade_ids": ("trade-2",),
        "top_n": 5,
        "metric": "realized_r",
    }
    assert scope_label == "Accepted closed trades"

    expected_columns = [
        "rank",
        "trade_id",
        "symbol",
        "status",
        "realized_r",
        "entry_ts",
        "reject_code",
        "note",
    ]
    assert best_df.columns.tolist() == expected_columns
    assert worst_df.columns.tolist() == expected_columns
    assert best_df.index.tolist() == [0]
    assert worst_df.index.tolist() == [0]


def test_strategy_modeling_trade_review_wrapper_scope_labels_and_display_index() -> None:
    trade_df = pd.DataFrame(
        {
            "trade_id": ["trade-a", "trade-b", "trade-c", "trade-d", "trade-e"],
            "symbol": ["SPY", "SPY", "QQQ", "QQQ", "IWM"],
            "status": ["closed", "closed", "closed", "open", "closed"],
            "realized_r": [1.0, -1.5, 0.5, 5.0, 2.0],
            "entry_ts": [
                "2026-01-02T14:30:00+00:00",
                "2026-01-02T14:31:00+00:00",
                "2026-01-02T14:29:00+00:00",
                "2026-01-02T14:35:00+00:00",
                "2026-01-02T14:32:00+00:00",
            ],
            "reject_code": [None, None, None, None, "symbol_not_allowed"],
            "pnl": [100.0, -150.0, 50.0, 500.0, 200.0],
        },
        index=[10, 20, 30, 40, 50],
    )

    best_df, worst_df, fallback_scope = build_trade_review_tables(trade_df, accepted_trade_ids=None)

    source_columns = [str(column) for column in trade_df.columns if str(column) != "rank"]
    expected_columns = ["rank", *source_columns]
    assert best_df.columns.tolist() == expected_columns
    assert worst_df.columns.tolist() == expected_columns
    assert best_df["trade_id"].tolist() == ["trade-a", "trade-c", "trade-b"]
    assert worst_df["trade_id"].tolist() == ["trade-b", "trade-c", "trade-a"]
    assert best_df["rank"].tolist() == [1, 2, 3]
    assert worst_df["rank"].tolist() == [1, 2, 3]
    assert fallback_scope == "Closed non-rejected trades"
    assert best_df.index.tolist() == [0, 1, 2]
    assert worst_df.index.tolist() == [0, 1, 2]

    accepted_best_df, accepted_worst_df, accepted_scope = build_trade_review_tables(
        trade_df,
        accepted_trade_ids=(),
    )
    assert accepted_scope == "Accepted closed trades"
    assert accepted_best_df.empty
    assert accepted_worst_df.empty
    assert accepted_best_df.columns.tolist() == expected_columns
    assert accepted_worst_df.columns.tolist() == expected_columns
    assert accepted_best_df.index.equals(pd.RangeIndex(start=0, stop=0, step=1))
    assert accepted_worst_df.index.equals(pd.RangeIndex(start=0, stop=0, step=1))
