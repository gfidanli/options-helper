from __future__ import annotations

import runpy
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest

from options_helper.data.intraday_store import IntradayStore
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_PATH = REPO_ROOT / "apps" / "streamlit" / "pages" / "11_Strategy_Modeling.py"


def _require_component_module() -> Any:
    pytest.importorskip("streamlit")
    return pytest.importorskip("apps.streamlit.components.strategy_modeling_page")


def _clear_component_caches(component: Any) -> None:
    for cache_name in (
        "_list_strategy_modeling_symbols_cached",
        "_load_strategy_modeling_payload_cached",
        "_load_strategy_modeling_data_payload_cached",
    ):
        cache = getattr(component, cache_name, None)
        clear = getattr(cache, "clear", None)
        if callable(clear):
            clear()


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
    timestamps = pd.date_range(start_ts, periods=rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [100.0 + i for i in range(rows)],
            "high": [100.2 + i for i in range(rows)],
            "low": [99.8 + i for i in range(rows)],
            "close": [100.1 + i for i in range(rows)],
            "volume": [1_000 + i for i in range(rows)],
            "vwap": [100.05 + i for i in range(rows)],
            "trade_count": [100 + i for i in range(rows)],
        }
    )


def _blocked_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "errors": [],
        "notes": [],
        "blocking": {
            "is_blocked": True,
            "reason": "intraday_coverage_missing",
            "blocked_symbols": ["SPY"],
            "missing_sessions_total": 2,
            "coverage_rows": [
                {
                    "symbol": "SPY",
                    "required_count": 3,
                    "covered_count": 1,
                    "missing_count": 2,
                    "missing_days": ["2026-01-06", "2026-01-07"],
                }
            ],
        },
    }


def test_strategy_modeling_component_seeded_db_success_path(tmp_path: Path) -> None:
    component = _require_component_module()
    db_path = tmp_path / "portal.duckdb"
    intraday_dir = tmp_path / "intraday"
    _seed_candles(db_path)

    store = IntradayStore(intraday_dir)
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 2), _intraday_df("2026-01-02 14:30:00+00:00"))
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 5), _intraday_df("2026-01-05 14:30:00+00:00"))

    _clear_component_caches(component)
    symbols, notes = component.list_strategy_modeling_symbols(database_path=db_path)
    assert notes == []
    assert symbols == ["AAPL", "MSFT"]

    payload = component.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=True,
    )

    assert payload["status"] == "ok"
    assert payload["errors"] == []
    assert payload["has_data"] is True
    assert payload["requested_symbols"] == ["AAPL"]
    assert payload["modeled_symbols"] == ["AAPL"]
    assert payload["required_sessions_by_symbol"]["AAPL"] == ["2026-01-02", "2026-01-05"]
    assert payload["intraday_preflight"]["is_blocked"] is False
    assert payload["blocking"]["is_blocked"] is False
    assert payload["blocking"]["reason"] is None
    assert payload["blocking"]["missing_sessions_total"] == 0


def test_strategy_modeling_component_handles_missing_db_states(tmp_path: Path) -> None:
    component = _require_component_module()
    missing_db = tmp_path / "missing.duckdb"
    _clear_component_caches(component)

    symbols, notes = component.list_strategy_modeling_symbols(database_path=missing_db)
    assert symbols == []
    assert notes
    assert any("not found" in str(note).lower() for note in notes)

    missing_payload = component.load_strategy_modeling_data_payload(database_path=missing_db)
    assert missing_payload["status"] == "error"
    assert missing_payload["blocking"]["is_blocked"] is True
    assert any("not found" in str(item).lower() for item in missing_payload["errors"])

    tableless_db = tmp_path / "tableless.duckdb"
    conn = duckdb.connect(str(tableless_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
    finally:
        conn.close()

    _clear_component_caches(component)
    table_payload = component.load_strategy_modeling_data_payload(database_path=tableless_db)
    assert table_payload["status"] == "error"
    assert table_payload["blocking"]["is_blocked"] is True
    assert any(
        ("candles_daily" in str(item).lower()) and ("not found" in str(item).lower())
        for item in table_payload["errors"]
    )


def test_strategy_modeling_component_flags_intraday_coverage_blocking(tmp_path: Path) -> None:
    component = _require_component_module()
    db_path = tmp_path / "portal.duckdb"
    intraday_dir = tmp_path / "intraday"
    _seed_candles(db_path)

    store = IntradayStore(intraday_dir)
    store.save_partition("stocks", "bars", "1Min", "AAPL", date(2026, 1, 2), _intraday_df("2026-01-02 14:30:00+00:00"))

    _clear_component_caches(component)
    blocked_payload = component.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=True,
    )
    assert blocked_payload["status"] == "ok"
    assert blocked_payload["blocking"]["is_blocked"] is True
    assert blocked_payload["blocking"]["reason"] == "intraday_coverage_missing"
    assert blocked_payload["blocking"]["blocked_symbols"] == ["AAPL"]
    assert blocked_payload["blocking"]["missing_sessions_total"] == 1

    _clear_component_caches(component)
    permissive_payload = component.load_strategy_modeling_data_payload(
        symbols=["AAPL"],
        start_date=date(2026, 1, 2),
        end_date=date(2026, 1, 5),
        database_path=db_path,
        intraday_dir=intraday_dir,
        intraday_timeframe="1Min",
        require_intraday_bars=False,
    )
    assert permissive_payload["status"] == "ok"
    assert permissive_payload["blocking"]["is_blocked"] is False
    assert permissive_payload["blocking"]["reason"] is None
    assert permissive_payload["blocking"]["missing_sessions_total"] == 1


def test_strategy_modeling_page_disables_run_when_intraday_coverage_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("options_helper.analysis.strategy_modeling")
    if not PAGE_PATH.exists():
        pytest.skip("Strategy modeling page scaffold is not present in this workspace.")

    from apps.streamlit.components import strategy_modeling_page as component
    import options_helper.cli_deps as cli_deps
    import streamlit as st

    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _blocked_payload())

    def _forbid_service_build() -> object:
        raise AssertionError("service should not be constructed while intraday coverage is blocked")

    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", _forbid_service_build)

    warnings: list[str] = []
    tables: list[pd.DataFrame] = []
    run_button_disabled: dict[str, bool] = {}

    def _warning(body: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        warnings.append(str(body))

    def _dataframe(data: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        if isinstance(data, pd.DataFrame):
            tables.append(data.copy())

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        if label == "Run Strategy Modeling":
            run_button_disabled[label] = bool(kwargs.get("disabled", False))
        return False

    monkeypatch.setattr(st, "warning", _warning)
    monkeypatch.setattr(st, "dataframe", _dataframe)
    monkeypatch.setattr(st, "button", _button)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_blocked__")

    assert run_button_disabled.get("Run Strategy Modeling") is True
    assert any("missing required intraday coverage" in item.lower() for item in warnings)

    coverage_table = next((frame for frame in tables if "missing_count" in frame.columns), None)
    assert coverage_table is not None
    assert str(coverage_table.iloc[0]["symbol"]) == "SPY"
    assert int(coverage_table.iloc[0]["missing_count"]) == 2
