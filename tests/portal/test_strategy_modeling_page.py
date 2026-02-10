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
from typer.testing import CliRunner

from options_helper.data.intraday_store import IntradayStore
from options_helper.data.strategy_modeling_profiles import (
    load_strategy_modeling_profile,
    save_strategy_modeling_profile,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.strategy_modeling_profile import StrategyModelingProfile


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


def _profile_fixture(**updates: object) -> StrategyModelingProfile:
    payload = {
        "strategy": "orb",
        "symbols": ["SPY"],
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 1, 31),
        "intraday_timeframe": "5Min",
        "intraday_source": "stocks_bars_local",
        "starting_capital": 25_000.0,
        "risk_per_trade_pct": 2.5,
        "gap_fill_policy": "fill_at_open",
        "max_hold_bars": 20,
        "one_open_per_symbol": True,
        "r_ladder_min_tenths": 12,
        "r_ladder_max_tenths": 16,
        "r_ladder_step_tenths": 2,
        "allow_shorts": False,
        "enable_orb_confirmation": True,
        "orb_range_minutes": 20,
        "orb_confirmation_cutoff_et": "10:15",
        "orb_stop_policy": "tighten",
        "enable_atr_stop_floor": True,
        "atr_stop_floor_multiple": 0.8,
        "enable_rsi_extremes": True,
        "enable_ema9_regime": True,
        "ema9_slope_lookback_bars": 5,
        "enable_volatility_regime": True,
        "allowed_volatility_regimes": ["low", "normal"],
        "ma_fast_window": 21,
        "ma_slow_window": 55,
        "ma_trend_window": 200,
        "ma_fast_type": "ema",
        "ma_slow_type": "sma",
        "ma_trend_type": "sma",
        "trend_slope_lookback_bars": 4,
        "atr_window": 10,
        "atr_stop_multiple": 1.7,
    }
    payload.update(updates)
    return StrategyModelingProfile.model_validate(payload)


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

    st.session_state.clear()
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
    assert "ma_crossover" in selectbox_options["Strategy"]
    assert "trend_following" in selectbox_options["Strategy"]
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


def test_strategy_modeling_page_ma_crossover_controls_build_signal_kwargs(
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
                filter_summary=None,
                directional_metrics=None,
            )

    stub = _StubService()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: stub)

    def _selectbox(label: str, *, options, index: int = 0, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        selected_map = {
            "Strategy": "ma_crossover",
            "Intraday timeframe": "5Min",
            "Intraday source": "stocks_bars_local",
            "Gap policy": "fill_at_open",
            "ORB stop policy": "base",
            "MA fast type": "ema",
            "MA slow type": "sma",
            "MA trend type": "sma",
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
            "Allowed volatility regimes": ["low", "normal", "high"],
        }
        selected = selected_map.get(label)
        if selected is not None:
            return [item for item in selected if item in option_values]
        if default is None:
            return []
        return list(default)

    def _number_input(label: str, value=0, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Starting capital": 25000.0,
            "Risk per trade (%)": 2.0,
            "Max hold (bars)": 20,
            "MA fast window": 21,
            "MA slow window": 55,
            "MA trend window": 200,
            "Trend slope lookback (bars)": 3,
            "ATR window": 10,
            "ATR stop multiple": 1.7,
            "ORB range (minutes)": 15,
            "ATR stop floor multiple": 0.5,
            "EMA9 slope lookback (bars)": 3,
        }
        return selected_map.get(label, value)

    def _checkbox(label: str, value: bool = False, **kwargs):  # noqa: ANN001,ARG001
        return value

    def _text_input(label: str, value: str = "", **kwargs) -> str:  # noqa: ARG001
        selected_map = {
            "Segment values (comma-separated)": "",
            "ORB confirmation cutoff ET (HH:MM)": "10:30",
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

    monkeypatch.setattr(st, "selectbox", _selectbox)
    monkeypatch.setattr(st, "multiselect", _multiselect)
    monkeypatch.setattr(st, "checkbox", _checkbox)
    monkeypatch.setattr(st, "number_input", _number_input)
    monkeypatch.setattr(st, "text_input", _text_input)
    monkeypatch.setattr(st, "date_input", _date_input)
    monkeypatch.setattr(st, "button", _button)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_ma_crossover__")

    assert stub.last_request is not None
    assert stub.last_request.strategy == "ma_crossover"
    assert stub.last_request.signal_kwargs == {
        "fast_window": 21,
        "slow_window": 55,
        "fast_type": "ema",
        "slow_type": "sma",
        "atr_window": 10,
        "atr_stop_multiple": 1.7,
    }


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

    st.session_state.clear()
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


def test_strategy_modeling_page_can_save_profile_and_cli_can_load_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("streamlit")
    pytest.importorskip("options_helper.analysis.strategy_modeling")
    if not PAGE_PATH.exists():
        pytest.skip("Strategy modeling page scaffold is not present in this workspace.")

    from apps.streamlit.components import strategy_modeling_page as component
    from options_helper.cli import app
    import options_helper.cli_deps as cli_deps
    import streamlit as st

    st.session_state.clear()
    profile_path = tmp_path / "strategy_profiles.json"

    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: None)

    def _selectbox(label: str, *, options, index: int = 0, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        selected_map = {
            "Saved profiles": "",
            "Strategy": "orb",
            "Intraday timeframe": "5Min",
            "Intraday source": "stocks_bars_local",
            "Gap policy": "fill_at_open",
            "MA fast type": "ema",
            "MA slow type": "sma",
            "MA trend type": "sma",
            "ORB stop policy": "tighten",
        }
        selected = selected_map.get(label)
        if selected in option_values:
            return selected
        if key and key in st.session_state and st.session_state[key] in option_values:
            return st.session_state[key]
        if option_values:
            return option_values[index]
        return None

    def _multiselect(label: str, *, options, default=None, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        selected_map = {
            "Tickers": ["SPY"],
            "Segment dimensions": ["symbol", "direction"],
            "Allowed volatility regimes": ["low", "normal"],
        }
        selected = selected_map.get(label)
        if selected is not None:
            return [item for item in selected if item in option_values]
        if key and key in st.session_state:
            return list(st.session_state[key] or [])
        return list(default or [])

    def _checkbox(label: str, value: bool = False, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Overwrite existing profile": False,
            "One open trade per symbol": True,
            "Allow short-direction entries": False,
            "Enable ORB confirmation gate": True,
            "Enable ATR stop floor": True,
            "Enable RSI extremes gate": True,
            "Enable EMA9 regime gate": True,
            "Enable volatility regime gate": True,
        }
        return selected_map.get(label, value)

    def _number_input(label: str, value=0, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Starting capital": 25_000.0,
            "Risk per trade (%)": 2.5,
            "Max hold (bars)": 20,
            "R ladder min (tenths)": 12,
            "R ladder max (tenths)": 16,
            "R ladder step (tenths)": 2,
            "MA fast window": 21,
            "MA slow window": 55,
            "MA trend window": 200,
            "Trend slope lookback (bars)": 4,
            "ATR window": 10,
            "ATR stop multiple": 1.7,
            "ORB range (minutes)": 20,
            "ATR stop floor multiple": 0.8,
            "EMA9 slope lookback (bars)": 5,
        }
        if label in selected_map:
            return selected_map[label]
        if key and key in st.session_state:
            return st.session_state[key]
        return value

    def _text_input(label: str, value: str = "", key: str | None = None, **kwargs) -> str:  # noqa: ARG001
        selected_map = {
            "Profile store path": str(profile_path),
            "Profile name": "portal_orb",
            "Segment values (comma-separated)": "",
            "ORB confirmation cutoff ET (HH:MM)": "10:15",
            "Export reports dir": str(tmp_path / "reports"),
            "Export output timezone": "America/Chicago",
        }
        selected = selected_map.get(label, value)
        if key is not None:
            st.session_state[key] = selected
        return selected

    def _date_input(label: str, value=None, **kwargs):  # noqa: ANN001,ARG001
        selected_map = {
            "Start date": date(2026, 1, 1),
            "End date": date(2026, 1, 31),
        }
        return selected_map.get(label, value)

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Save Profile"

    monkeypatch.setattr(st, "selectbox", _selectbox)
    monkeypatch.setattr(st, "multiselect", _multiselect)
    monkeypatch.setattr(st, "checkbox", _checkbox)
    monkeypatch.setattr(st, "number_input", _number_input)
    monkeypatch.setattr(st, "text_input", _text_input)
    monkeypatch.setattr(st, "date_input", _date_input)
    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "rerun", lambda: None)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_save_profile__")

    saved_profile = load_strategy_modeling_profile(profile_path, "portal_orb")
    assert saved_profile.strategy == "orb"
    assert saved_profile.symbols == ("SPY",)
    assert saved_profile.intraday_timeframe == "5Min"
    assert saved_profile.r_ladder_min_tenths == 12
    assert saved_profile.r_ladder_max_tenths == 16
    assert saved_profile.r_ladder_step_tenths == 2

    class _CliStubService:
        def __init__(self) -> None:
            self.last_request = None

        def list_universe_loader(self, *, database_path=None):  # noqa: ANN001
            del database_path
            return SimpleNamespace(symbols=["SPY"], notes=[])

        def run(self, request):  # noqa: ANN001
            self.last_request = request
            return SimpleNamespace(
                strategy=request.strategy,
                as_of=request.end_date,
                requested_symbols=tuple(request.symbols or ()),
                modeled_symbols=tuple(request.symbols or ()),
                signal_events=(),
                trade_simulations=(),
                accepted_trade_ids=(),
                skipped_trade_ids=(),
                intraday_preflight=SimpleNamespace(
                    blocked_symbols=[],
                    coverage_by_symbol={},
                    notes=[],
                ),
                portfolio_metrics=None,
                target_hit_rates=(),
                segment_records=(),
                filter_summary={},
                directional_metrics={},
            )

    cli_stub = _CliStubService()
    monkeypatch.setattr(
        "options_helper.commands.technicals.cli_deps.build_strategy_modeling_service",
        lambda: cli_stub,
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "strategy-model",
            "--profile-path",
            str(profile_path),
            "--profile",
            "portal_orb",
        ],
    )
    assert result.exit_code == 0, result.output
    assert cli_stub.last_request is not None
    assert cli_stub.last_request.strategy == "orb"
    assert tuple(cli_stub.last_request.symbols) == ("SPY",)
    assert cli_stub.last_request.intraday_timeframe == "5Min"


def test_strategy_modeling_page_load_profile_populates_request_values(
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
                filter_summary={},
                directional_metrics={},
            )

    st.session_state.clear()
    profile_path = tmp_path / "strategy_profiles.json"
    st.session_state["strategy_modeling_profile_path"] = str(profile_path)
    loaded = _profile_fixture(
        strategy="trend_following",
        intraday_timeframe="5Min",
        intraday_source="stocks_bars_local",
        starting_capital=30_000.0,
        risk_per_trade_pct=3.0,
        max_hold_bars=15,
        r_ladder_min_tenths=11,
        r_ladder_max_tenths=13,
        r_ladder_step_tenths=1,
    )
    save_strategy_modeling_profile(profile_path, "cli_saved", loaded, overwrite=False)

    stub = _StubService()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: stub)

    active_buttons: set[str] = {"Load Profile"}

    def _selectbox(label: str, *, options, index: int = 0, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        if key and key in st.session_state and st.session_state[key] in option_values:
            return st.session_state[key]
        selected_map = {"Saved profiles": "cli_saved"}
        selected = selected_map.get(label)
        if selected in option_values:
            return selected
        if option_values:
            return option_values[index]
        return None

    def _multiselect(label: str, *, options, default=None, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        if key and key in st.session_state:
            return [item for item in list(st.session_state[key] or []) if item in option_values]
        if default is None:
            return []
        return [item for item in list(default) if item in option_values]

    def _checkbox(label: str, value: bool = False, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        if key and key in st.session_state:
            return bool(st.session_state[key])
        return value

    def _number_input(label: str, value=0, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        if key and key in st.session_state:
            return st.session_state[key]
        return value

    def _text_input(label: str, value: str = "", key: str | None = None, **kwargs) -> str:  # noqa: ARG001
        selected_map = {
            "Profile store path": str(profile_path),
            "Profile name": "cli_saved",
            "Segment values (comma-separated)": "",
            "ORB confirmation cutoff ET (HH:MM)": "10:30",
            "Export reports dir": str(tmp_path / "reports"),
            "Export output timezone": "America/Chicago",
        }
        selected = selected_map.get(label, value)
        if key is not None:
            st.session_state[key] = selected
        return selected

    def _date_input(label: str, value=None, key: str | None = None, **kwargs):  # noqa: ANN001,ARG001
        if key and key in st.session_state:
            return st.session_state[key]
        selected_map = {"Start date": date(2026, 1, 1), "End date": date(2026, 1, 31)}
        return selected_map.get(label, value)

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label in active_buttons

    monkeypatch.setattr(st, "selectbox", _selectbox)
    monkeypatch.setattr(st, "multiselect", _multiselect)
    monkeypatch.setattr(st, "checkbox", _checkbox)
    monkeypatch.setattr(st, "number_input", _number_input)
    monkeypatch.setattr(st, "text_input", _text_input)
    monkeypatch.setattr(st, "date_input", _date_input)
    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "rerun", lambda: None)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_load_profile_step1__")

    active_buttons.clear()
    active_buttons.add("Run Strategy Modeling")
    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_load_profile_step2__")

    assert stub.last_request is not None
    assert stub.last_request.strategy == "trend_following"
    assert tuple(stub.last_request.symbols) == ("SPY",)
    assert stub.last_request.intraday_timeframe == "5Min"
    assert stub.last_request.starting_capital == 30_000.0
    assert stub.last_request.max_hold_bars == 15
    assert stub.last_request.policy["risk_per_trade_pct"] == 3.0
    assert [target.label for target in stub.last_request.target_ladder] == ["1.1R", "1.2R", "1.3R"]


def test_strategy_modeling_page_trade_review_selection_precedence_and_drilldown_fallback(
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
                segment_records=(),
                filter_summary={},
                directional_metrics={},
                accepted_trade_ids=("best-1", "worst-1", "log-1"),
                trade_simulations=(
                    {
                        "trade_id": "best-1",
                        "symbol": "BEST",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": "2026-01-31T15:30:00+00:00",
                        "entry_price": 100.0,
                        "stop_price": 99.0,
                        "target_price": 101.5,
                        "exit_ts": "2026-01-31T16:30:00+00:00",
                        "exit_price": 101.0,
                        "realized_r": 1.0,
                    },
                    {
                        "trade_id": "worst-1",
                        "symbol": "WORST",
                        "direction": "short",
                        "status": "closed",
                        "entry_ts": "2026-01-30T15:30:00+00:00",
                        "entry_price": 100.0,
                        "stop_price": 101.0,
                        "target_price": 98.5,
                        "exit_ts": "2026-01-30T16:00:00+00:00",
                        "exit_price": 101.0,
                        "realized_r": -1.0,
                    },
                    {
                        "trade_id": "log-1",
                        "symbol": "LOG",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": "2026-01-29T15:30:00+00:00",
                        "entry_price": 50.0,
                        "stop_price": 49.0,
                        "target_price": 52.0,
                        "exit_ts": "2026-01-29T16:30:00+00:00",
                        "exit_price": 51.0,
                        "realized_r": 1.0,
                    },
                ),
            )

    st.session_state.clear()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["BEST", "WORST", "LOG"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    def _stub_review_tables(
        trade_df: pd.DataFrame,
        accepted_trade_ids: object,
        *,
        top_n: int = 20,
    ) -> tuple[pd.DataFrame, pd.DataFrame, str]:
        del trade_df, accepted_trade_ids, top_n
        return (
            pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "trade_id": "best-1",
                        "symbol": "BEST",
                        "realized_r": 1.0,
                        "entry_ts": "2026-01-31T15:30:00+00:00",
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "rank": 1,
                        "trade_id": "worst-1",
                        "symbol": "WORST",
                        "realized_r": -1.0,
                        "entry_ts": "2026-01-30T15:30:00+00:00",
                    },
                ]
            ),
            "Accepted closed trades",
        )

    load_calls: list[tuple[str, str]] = []

    def _stub_load_intraday_window(
        store_root: object,
        symbol: str,
        timeframe: str,
        start_ts: object,
        end_ts: object,
    ) -> pd.DataFrame:
        del store_root, start_ts, end_ts
        load_calls.append((symbol, timeframe))
        if timeframe != "5Min":
            return pd.DataFrame()
        ts = pd.date_range("2026-01-31 14:30:00+00:00", periods=80, freq="5min", tz="UTC")
        base = pd.Series(range(len(ts)), dtype=float)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "open": 100.0 + base,
                "high": 100.5 + base,
                "low": 99.5 + base,
                "close": 100.2 + base,
                "volume": 1_000.0,
                "vwap": 100.1 + base,
                "trade_count": 100.0,
            }
        )

    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_review.build_trade_review_tables",
        _stub_review_tables,
    )
    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_drilldown.load_intraday_window",
        _stub_load_intraday_window,
    )

    chart_timeframe_options: list[str] = []
    dataframe_kwargs_by_key: dict[str, dict[str, object]] = {}
    warnings: list[str] = []

    def _selectbox(label: str, *, options, index: int = 0, **kwargs):  # noqa: ANN001,ARG001
        option_values = list(options)
        if label == "Chart timeframe":
            chart_timeframe_options[:] = option_values
            if "15Min" in option_values:
                return "15Min"
        if option_values:
            return option_values[index]
        return None

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        del data
        key = str(kwargs.get("key") or "")
        if key:
            dataframe_kwargs_by_key[key] = {
                "on_select": kwargs.get("on_select"),
                "selection_mode": kwargs.get("selection_mode"),
            }
        events = {
            "strategy_modeling_trade_review_top_best": {"selection": {"rows": [0]}},
            "strategy_modeling_trade_review_top_worst": {"selection": {"rows": [0]}},
            "strategy_modeling_trade_review_full_log": {"selection": {"rows": [1]}},
        }
        return events.get(key)

    def _warning(body: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        warnings.append(str(body))

    monkeypatch.setattr(st, "selectbox", _selectbox)
    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)
    monkeypatch.setattr(st, "warning", _warning)
    monkeypatch.setattr(st, "altair_chart", lambda *args, **kwargs: None)  # noqa: ARG005
    monkeypatch.setattr(st, "line_chart", lambda *args, **kwargs: None)  # noqa: ARG005

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_trade_precedence__")

    for key in (
        "strategy_modeling_trade_review_top_best",
        "strategy_modeling_trade_review_top_worst",
        "strategy_modeling_trade_review_full_log",
    ):
        assert dataframe_kwargs_by_key[key]["on_select"] == "rerun"
        assert dataframe_kwargs_by_key[key]["selection_mode"] == "single-row"
    assert load_calls
    assert load_calls[0][0] == "BEST"
    assert any(timeframe == "1Min" for _, timeframe in load_calls)
    assert any(timeframe == "5Min" for _, timeframe in load_calls)
    assert chart_timeframe_options == ["5Min", "15Min", "30Min", "60Min"]
    assert any("Using `5Min` base bars" in item for item in warnings)


def test_strategy_modeling_page_trade_drilldown_warns_when_no_selection(
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
                segment_records=(),
                filter_summary={},
                directional_metrics={},
                accepted_trade_ids=("trade-1",),
                trade_simulations=(
                    {
                        "trade_id": "trade-1",
                        "symbol": "SPY",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": "2026-01-31T15:30:00+00:00",
                        "entry_price": 100.0,
                        "stop_price": 99.0,
                        "target_price": 101.0,
                        "exit_ts": "2026-01-31T16:00:00+00:00",
                        "exit_price": 101.0,
                        "realized_r": 1.0,
                    },
                ),
            )

    st.session_state.clear()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_review.build_trade_review_tables",
        lambda *args, **kwargs: (  # noqa: ARG005
            pd.DataFrame([{"rank": 1, "trade_id": "trade-1"}]),
            pd.DataFrame([{"rank": 1, "trade_id": "trade-1"}]),
            "Accepted closed trades",
        ),
    )

    warnings: list[str] = []

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        del data
        return None

    def _warning(body: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        warnings.append(str(body))

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)
    monkeypatch.setattr(st, "warning", _warning)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_trade_drilldown_no_selection__")

    assert any("Select a row from Top 20 Best Trades" in item for item in warnings)


def test_strategy_modeling_page_trade_drilldown_warns_for_missing_timestamps(
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
                segment_records=(),
                filter_summary={},
                directional_metrics={},
                accepted_trade_ids=("trade-missing-ts",),
                trade_simulations=(
                    {
                        "trade_id": "trade-missing-ts",
                        "symbol": "SPY",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": None,
                        "entry_price": 100.0,
                        "stop_price": 99.0,
                        "target_price": 101.0,
                        "exit_ts": None,
                        "exit_price": 101.0,
                        "realized_r": 1.0,
                    },
                ),
            )

    st.session_state.clear()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_review.build_trade_review_tables",
        lambda *args, **kwargs: (  # noqa: ARG005
            pd.DataFrame([{"rank": 1, "trade_id": "trade-missing-ts"}]),
            pd.DataFrame(),
            "Accepted closed trades",
        ),
    )
    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_drilldown.load_intraday_window",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("loader should not run without timestamps")),
    )

    warnings: list[str] = []

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        del data
        events = {
            "strategy_modeling_trade_review_top_best": {"selection": {"rows": [0]}},
        }
        return events.get(str(kwargs.get("key") or ""))

    def _warning(body: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        warnings.append(str(body))

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)
    monkeypatch.setattr(st, "warning", _warning)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_trade_drilldown_missing_ts__")

    assert any("missing entry/exit timestamps" in item for item in warnings)


def test_strategy_modeling_page_trade_drilldown_warns_for_missing_bars(
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
                segment_records=(),
                filter_summary={},
                directional_metrics={},
                accepted_trade_ids=("trade-1",),
                trade_simulations=(
                    {
                        "trade_id": "trade-1",
                        "symbol": "SPY",
                        "direction": "long",
                        "status": "closed",
                        "entry_ts": "2026-01-31T15:30:00+00:00",
                        "entry_price": 100.0,
                        "stop_price": 99.0,
                        "target_price": 101.0,
                        "exit_ts": "2026-01-31T16:00:00+00:00",
                        "exit_price": 101.0,
                        "realized_r": 1.0,
                    },
                ),
            )

    st.session_state.clear()
    _clear_component_caches(component)
    monkeypatch.setattr(component, "list_strategy_modeling_symbols", lambda **_: (["SPY"], []))
    monkeypatch.setattr(component, "load_strategy_modeling_data_payload", lambda **_: _ready_payload())
    monkeypatch.setattr(cli_deps, "build_strategy_modeling_service", lambda: _StubService())

    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_review.build_trade_review_tables",
        lambda *args, **kwargs: (  # noqa: ARG005
            pd.DataFrame([{"rank": 1, "trade_id": "trade-1"}]),
            pd.DataFrame(),
            "Accepted closed trades",
        ),
    )
    monkeypatch.setattr(
        "apps.streamlit.components.strategy_modeling_trade_drilldown.load_intraday_window",
        lambda *args, **kwargs: pd.DataFrame(),  # noqa: ARG005
    )

    warnings: list[str] = []

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        return label == "Run Strategy Modeling"

    def _dataframe(data: object, *args: object, **kwargs: object) -> object:  # noqa: ARG001
        del data
        events = {
            "strategy_modeling_trade_review_top_best": {"selection": {"rows": [0]}},
        }
        return events.get(str(kwargs.get("key") or ""))

    def _warning(body: object, *args: object, **kwargs: object) -> None:  # noqa: ARG001
        warnings.append(str(body))

    monkeypatch.setattr(st, "button", _button)
    monkeypatch.setattr(st, "dataframe", _dataframe)
    monkeypatch.setattr(st, "warning", _warning)

    runpy.run_path(str(PAGE_PATH), run_name="__strategy_modeling_page_trade_drilldown_missing_bars__")

    assert any("No intraday bars found for selected trade/context window" in item for item in warnings)


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

    button_help: dict[str, str] = {}

    def _button(label: str, *args: object, **kwargs: object) -> bool:  # noqa: ARG001
        help_text = kwargs.get("help")
        if isinstance(help_text, str):
            button_help[label] = help_text
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
    assert "top_20_best_trades.csv" in button_help.get("Export Reports", "")
    assert "top_20_worst_trades.csv" in button_help.get("Export Reports", "")
