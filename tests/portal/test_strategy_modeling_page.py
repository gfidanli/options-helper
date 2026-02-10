from __future__ import annotations

import json
import runpy
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
import pandas as pd
from pandas.io.formats.style import Styler
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


def _ready_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "errors": [],
        "notes": [],
        "blocking": {
            "is_blocked": False,
            "reason": None,
            "blocked_symbols": [],
            "missing_sessions_total": 0,
            "coverage_rows": [],
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


def test_strategy_modeling_page_orb_controls_build_request_and_render_filter_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("options_helper.analysis.strategy_modeling")
    if not PAGE_PATH.exists():
        pytest.skip("Strategy modeling page scaffold is not present in this workspace.")

    from apps.streamlit.components import strategy_modeling_page as component
    import options_helper.cli_deps as cli_deps
    import streamlit as st

    class _StubService:
        def __init__(self) -> None:
            self.last_request = None

        def run(self, request):  # noqa: ANN001
            self.last_request = request
            return SimpleNamespace(
                portfolio_metrics=None,
                target_hit_rates=(),
                equity_curve=(),
                segment_records=(),
                trade_simulations=(),
                filter_summary={
                    "base_event_count": 8,
                    "kept_event_count": 3,
                    "rejected_event_count": 5,
                    "reject_counts": {
                        "shorts_disabled": 2,
                        "orb_breakout_missing": 1,
                        "atr_floor_failed": 0,
                    },
                },
                directional_metrics={
                    "combined": {"trade_count": 3, "portfolio_metrics": {"total_return_pct": 2.0}},
                    "long_only": {"trade_count": 2, "portfolio_metrics": {"total_return_pct": 4.5}},
                    "short_only": {"trade_count": 1, "portfolio_metrics": {"total_return_pct": -1.0}},
                },
            )

    stub = _StubService()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: stub)

    selectbox_options: dict[str, list[str]] = {}
    frames: list[pd.DataFrame] = []

    def _selectbox(label: str, *, options, index: int = 0, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        selectbox_options[label] = option_values
        selected_map = {
            "Strategy": "orb",
            "Intraday timeframe": "5Min",
            "Intraday source": "stocks_bars_local",
            "Gap policy": "fill_at_open",
            "ORB stop policy": "tighten",
        }
        selected = selected_map.get(label)
        if selected in option_values:
            return selected
        if option_values:
            return option_values[index]
        return None

    def _multiselect(label: str, *, options, default=None, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        selected_map = {
            "Tickers": ["SPY"],
            "Segment dimensions": ["symbol", "direction"],
            "Allowed volatility regimes": ["low", "normal"],
        }
        selected = selected_map.get(label)
        if selected is not None:
            return [item for item in selected if item in option_values]
        if default is None:
            return []
        return list(default)

    def _checkbox(label: str, value: bool = False, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "One open trade per symbol": True,
            "Allow short-direction entries": False,
            "Enable ORB confirmation gate": True,
            "Enable ATR stop floor": True,
            "Enable RSI extremes gate": True,
            "Enable EMA9 regime gate": True,
            "Enable volatility regime gate": True,
        }
        return selected_map.get(label, value)

    def _number_input(label: str, value=0, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Starting capital": 25000.0,
            "Risk per trade (%)": 2.5,
            "Max hold (bars)": 20,
            "ORB range (minutes)": 20,
            "ATR stop floor multiple": 0.8,
            "EMA9 slope lookback (bars)": 5,
        }
        return selected_map.get(label, value)

    def _text_input(label: str, value: str = "", **kwargs) -> str:  # noqa: ARG001
        selected_map = {
            "Segment values (comma-separated)": "",
            "ORB confirmation cutoff ET (HH:MM)": "10:15",
        }
        return selected_map.get(label, value)

    def _date_input(label: str, value=None, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Start date": date(2026, 1, 1),
            "End date": date(2026, 1, 31),
        }
        return selected_map.get(label, value)

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        if isinstance(data, pd.DataFrame):
            frames.append(data.copy())

    monkeypatch.setattr(st, "selectbox", _selectbox)
    monkeypatch.setattr(st, "multiselect", _multiselect)
    monkeypatch.setattr(st, "checkbox", _checkbox)
    monkeypatch.setattr(st, "number_input", _number_input)
    monkeypatch.setattr(st, "text_input", _text_input)
    monkeypatch.setattr(st, "date_input", _date_input)
    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_orb__")

    assert "Strategy" in selectbox_options
    assert "orb" in selectbox_options["Strategy"]
    assert stub.last_request is not None
    assert stub.last_request.strategy == "orb"
    assert tuple(stub.last_request.symbols) == ("SPY",)
    assert stub.last_request.filter_config.allow_shorts is False
    assert stub.last_request.filter_config.enable_orb_confirmation is True
    assert stub.last_request.filter_config.orb_range_minutes == 20
    assert stub.last_request.filter_config.orb_confirmation_cutoff_et == "10:15"
    assert stub.last_request.filter_config.orb_stop_policy == "tighten"
    assert stub.last_request.filter_config.enable_atr_stop_floor is True
    assert stub.last_request.filter_config.atr_stop_floor_multiple == 0.8
    assert stub.last_request.filter_config.enable_rsi_extremes is True
    assert stub.last_request.filter_config.enable_ema9_regime is True
    assert stub.last_request.filter_config.ema9_slope_lookback_bars == 5
    assert stub.last_request.filter_config.enable_volatility_regime is True
    assert stub.last_request.filter_config.allowed_volatility_regimes == ("low", "normal")

    summary_frame = next((frame for frame in frames if set(frame.columns) == {"metric", "value"}), None)
    assert summary_frame is not None
    assert set(summary_frame["metric"]) == {"base_event_count", "kept_event_count", "rejected_event_count"}

    directional_frame = next((frame for frame in frames if "directional_bucket" in frame.columns), None)
    assert directional_frame is not None
    assert set(directional_frame["directional_bucket"]) == {"combined", "long_only", "short_only"}


def test_strategy_modeling_page_segment_breakdowns_apply_column_color_styling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("options_helper.analysis.strategy_modeling")
    if not PAGE_PATH.exists():
        pytest.skip("Strategy modeling page scaffold is not present in this workspace.")

    from apps.streamlit.components import strategy_modeling_page as component
    import options_helper.cli_deps as cli_deps
    import streamlit as st

    class _StubService:
        def run(self, request):  # noqa: ANN001
            del request
            return SimpleNamespace(
                portfolio_metrics=None,
                target_hit_rates=(),
                equity_curve=(),
                segment_records=(
                    {
                        "segment_dimension": "symbol",
                        "segment_value": "AAPL",
                        "trade_count": 15,
                        "win_rate": 0.67,
                        "avg_realized_r": 0.12,
                    },
                    {
                        "segment_dimension": "symbol",
                        "segment_value": "MSFT",
                        "trade_count": 12,
                        "win_rate": 0.33,
                        "avg_realized_r": -0.08,
                    },
                ),
                trade_simulations=(),
                filter_summary=None,
                directional_metrics=None,
            )

    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["AAPL", "MSFT"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    segment_stylers: list[Styler] = []

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        if not isinstance(data, Styler):
            return
        if {"segment_dimension", "segment_value"}.issubset(set(data.data.columns)):
            segment_stylers.append(data)

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_segment_styling__")

    assert len(segment_stylers) == 1
    html = segment_stylers[0].to_html()
    assert "background-color: rgb(" in html
    assert "color: #111827" in html
    assert "outline: 2px solid" in html


def test_strategy_modeling_page_defaults_tickers_to_core_symbols(
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
    monkeypatch.setattr(
        component,
        "list_strategy_modeling_symbols",
        lambda **_: (["AAPL", "AMZN", "MSFT", "NVDA", "SPY"], []),
    )
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: None)

    captured_defaults: dict[str, list[str]] = {}

    def _multiselect(label: str, *, options, default=None, **kwargs):  # noqa: ANN001,ARG001
        values = list(default or [])
        if label == "Tickers":
            captured_defaults[label] = values
        return values

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return False

    monkeypatch.setattr(st, "multiselect", _multiselect)
    monkeypatch.setattr(st, "button", _button)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_default_tickers__")

    assert captured_defaults.get("Tickers") == ["SPY", "AAPL", "AMZN", "NVDA"]


def test_strategy_modeling_page_can_export_reports_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("options_helper.analysis.strategy_modeling")
    if not PAGE_PATH.exists():
        pytest.skip("Strategy modeling page scaffold is not present in this workspace.")

    from apps.streamlit.components import strategy_modeling_page as component
    import options_helper.cli_deps as cli_deps
    import streamlit as st

    class _StubService:
        def run(self, request):  # noqa: ANN001
            del request
            return SimpleNamespace(
                strategy="sfp",
                requested_symbols=("SPY",),
                modeled_symbols=("SPY",),
                signal_events=("evt-1",),
                accepted_trade_ids=("trade-1",),
                skipped_trade_ids=(),
                portfolio_metrics={
                    "starting_capital": 25000.0,
                    "ending_capital": 25250.0,
                    "total_return_pct": 1.0,
                    "trade_count": 1,
                    "win_rate": 100.0,
                    "expectancy_r": 1.0,
                },
                target_hit_rates=(
                    {"target_label": "1.0R", "target_r": 1.0, "trade_count": 1, "hit_count": 1, "hit_rate": 1.0},
                ),
                equity_curve=(),
                segment_records=(
                    {
                        "segment_dimension": "symbol",
                        "segment_value": "SPY",
                        "trade_count": 1,
                        "win_rate": 1.0,
                        "avg_realized_r": 1.0,
                    },
                ),
                trade_simulations=(
                    {
                        "trade_id": "trade-1",
                        "event_id": "evt-1",
                        "symbol": "SPY",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": "2026-01-31T15:30:00+00:00",
                        "entry_price": 100.0,
                        "stop_price": 99.0,
                        "target_price": 101.0,
                        "exit_ts": "2026-01-31T16:00:00+00:00",
                        "exit_price": 101.0,
                        "exit_reason": "target_hit",
                        "initial_risk": 1.0,
                        "realized_r": 1.0,
                    },
                ),
                filter_summary={},
                directional_metrics={},
            )

    export_root = tmp_path / "strategy_modeling_reports"

    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    def _text_input(label: str, value: str = "", **kwargs) -> str:  # noqa: ARG001
        selected_map = {
            "Segment values (comma-separated)": "",
            "ORB confirmation cutoff ET (HH:MM)": "10:30",
            "Export reports dir": str(export_root),
            "Export output timezone": "America/Chicago",
        }
        return selected_map.get(label, value)

    def _date_input(label: str, value=None, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Start date": date(2026, 1, 1),
            "End date": date(2026, 1, 31),
        }
        return selected_map.get(label, value)

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label in {"Run Strategy Modeling", "Export Reports"}

    monkeypatch.setattr(st, "text_input", _text_input)
    monkeypatch.setattr(st, "date_input", _date_input)
    monkeypatch.setattr(st, "button", _button)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_export_reports__")

    run_dir = export_root / "sfp" / "2026-01-31"
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (run_dir / "llm_analysis_prompt.md").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "r_ladder.csv").exists()
    assert (run_dir / "segments.csv").exists()

    summary_payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["strategy"] == "sfp"
    assert summary_payload["requested_symbols"] == ["SPY"]
    assert summary_payload["summary"]["trade_count"] == 1
