from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.storage_runtime import (
    reset_default_duckdb_path,
    reset_default_storage_backend,
    set_default_duckdb_path,
    set_default_storage_backend,
)
from options_helper.data.store_factory import close_warehouses, get_research_metrics_store
from options_helper.data.stores_duckdb import DuckDBResearchMetricsStore
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.schemas.research_metrics_contracts import (
    EXPOSURE_STRIKE_FIELDS,
    INTRADAY_FLOW_CONTRACT_FIELDS,
    IV_SURFACE_DELTA_BUCKET_FIELDS,
    IV_SURFACE_TENOR_FIELDS,
)


def test_duckdb_research_metrics_store_roundtrip(tmp_path) -> None:
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)
    store = DuckDBResearchMetricsStore(root_dir=tmp_path / "research", warehouse=wh)

    tenor_df = pd.DataFrame(
        [
            {
                "symbol": "aapl",
                "as_of": "2026-02-05",
                "tenor_target_dte": 7,
                "expiry": "2026-02-12",
                "dte": 7,
                "tenor_gap_dte": 0,
                "atm_strike": 190.0,
                "atm_iv": 0.25,
                "atm_mark": 4.2,
                "straddle_mark": 8.3,
                "expected_move_pct": 0.043,
                "skew_25d_pp": -1.0,
                "skew_10d_pp": -2.0,
                "contracts_used": 48,
                "warnings": [],
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "tenor_target_dte": 7,
                "expiry": "2026-02-12",
                "dte": 7,
                "tenor_gap_dte": 0,
                "atm_strike": 190.0,
                "atm_iv": 0.27,
                "atm_mark": 4.4,
                "straddle_mark": 8.8,
                "expected_move_pct": 0.046,
                "skew_25d_pp": -0.8,
                "skew_10d_pp": -1.5,
                "contracts_used": 50,
                "warnings": ["updated"],
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "tenor_target_dte": 30,
                "expiry": "2026-03-06",
                "dte": 29,
                "tenor_gap_dte": -1,
                "atm_strike": 190.0,
                "atm_iv": 0.24,
                "atm_mark": 7.0,
                "straddle_mark": 14.1,
                "expected_move_pct": 0.074,
                "skew_25d_pp": -0.2,
                "skew_10d_pp": -0.7,
                "contracts_used": 60,
                "warnings": ["tenor_gap"],
            },
        ]
    )
    assert store.upsert_iv_surface_tenor(tenor_df, provider="alpaca") == 2

    loaded_tenor = store.load_iv_surface_tenor(symbol="AAPL", as_of="2026-02-05", provider="alpaca")
    assert list(loaded_tenor.columns) == list(IV_SURFACE_TENOR_FIELDS) + ["provider", "updated_at"]
    assert list(loaded_tenor["tenor_target_dte"]) == [7, 30]
    assert float(loaded_tenor.iloc[0]["atm_iv"]) == 0.27
    assert loaded_tenor.iloc[0]["warnings"] == ["updated"]

    loaded_tenor_latest = store.load_iv_surface_tenor(symbol="AAPL", provider="alpaca")
    assert len(loaded_tenor_latest) == 2

    delta_df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "tenor_target_dte": 7,
                "expiry": "2026-02-12",
                "option_type": "call",
                "delta_bucket": "d20_40",
                "avg_iv": 0.22,
                "median_iv": 0.21,
                "n_contracts": 5,
                "warnings": [],
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "tenor_target_dte": 7,
                "expiry": "2026-02-12",
                "option_type": "call",
                "delta_bucket": "d20_40",
                "avg_iv": 0.23,
                "median_iv": 0.22,
                "n_contracts": 6,
                "warnings": ["refresh"],
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "tenor_target_dte": 7,
                "expiry": "2026-02-12",
                "option_type": "put",
                "delta_bucket": "d60_80",
                "avg_iv": 0.29,
                "median_iv": 0.28,
                "n_contracts": 4,
                "warnings": [],
            },
        ]
    )
    assert store.upsert_iv_surface_delta_buckets(delta_df, provider="alpaca") == 2

    loaded_delta = store.load_iv_surface_delta_buckets(symbol="AAPL", as_of=date(2026, 2, 5), provider="alpaca")
    assert list(loaded_delta.columns) == list(IV_SURFACE_DELTA_BUCKET_FIELDS) + ["provider", "updated_at"]
    assert len(loaded_delta) == 2
    call_row = loaded_delta[loaded_delta["option_type"] == "call"].iloc[0]
    assert float(call_row["avg_iv"]) == 0.23
    assert call_row["warnings"] == ["refresh"]

    exposure_df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "expiry": "2026-02-12",
                "strike": 185,
                "call_oi": 1000,
                "put_oi": 900,
                "call_gex": 4000.0,
                "put_gex": 3200.0,
                "net_gex": 800.0,
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "expiry": "2026-02-12",
                "strike": 185,
                "call_oi": 1100,
                "put_oi": 950,
                "call_gex": 4200.0,
                "put_gex": 3300.0,
                "net_gex": 900.0,
            },
            {
                "symbol": "AAPL",
                "as_of": "2026-02-05",
                "expiry": "2026-03-06",
                "strike": 190,
                "call_oi": 1400,
                "put_oi": 700,
                "call_gex": 6100.0,
                "put_gex": 2500.0,
                "net_gex": 3600.0,
            },
        ]
    )
    assert store.upsert_dealer_exposure_strikes(exposure_df, provider="alpaca") == 2

    loaded_exposure = store.load_dealer_exposure_strikes(symbol="AAPL", as_of="2026-02-05", provider="alpaca")
    assert list(loaded_exposure.columns) == list(EXPOSURE_STRIKE_FIELDS) + ["provider", "updated_at"]
    assert len(loaded_exposure) == 2
    updated_exposure = loaded_exposure[loaded_exposure["strike"] == 185.0].iloc[0]
    assert float(updated_exposure["net_gex"]) == 900.0

    intraday_df = pd.DataFrame(
        [
            {
                "symbol": "AAPL",
                "market_date": "2026-02-05",
                "source": "alpaca_stream",
                "contract_symbol": "AAPL260212C00190000",
                "expiry": "2026-02-12",
                "option_type": "call",
                "strike": 190,
                "delta_bucket": "d40_60",
                "buy_volume": 35,
                "sell_volume": 20,
                "unknown_volume": 5,
                "buy_notional": 10200,
                "sell_notional": 6700,
                "net_notional": 3500,
                "trade_count": 12,
                "unknown_trade_share": 0.25,
                "quote_coverage_pct": 0.75,
                "warnings": [],
            },
            {
                "symbol": "AAPL",
                "market_date": "2026-02-05",
                "source": "alpaca_stream",
                "contract_symbol": "AAPL260212C00190000",
                "expiry": "2026-02-12",
                "option_type": "call",
                "strike": 190,
                "delta_bucket": "d40_60",
                "buy_volume": 40,
                "sell_volume": 20,
                "unknown_volume": 2,
                "buy_notional": 12000,
                "sell_notional": 6400,
                "net_notional": 5600,
                "trade_count": 14,
                "unknown_trade_share": 0.20,
                "quote_coverage_pct": 0.80,
                "warnings": ["updated"],
            },
            {
                "symbol": "AAPL",
                "market_date": "2026-02-05",
                "source": "alpaca_stream",
                "contract_symbol": "AAPL260212P00185000",
                "expiry": "2026-02-12",
                "option_type": "put",
                "strike": 185,
                "delta_bucket": "d60_80",
                "buy_volume": 15,
                "sell_volume": 45,
                "unknown_volume": 0,
                "buy_notional": 3500,
                "sell_notional": 9900,
                "net_notional": -6400,
                "trade_count": 10,
                "unknown_trade_share": 0.0,
                "quote_coverage_pct": 1.0,
                "warnings": ["crossed_quotes"],
            },
        ]
    )
    assert store.upsert_intraday_option_flow(intraday_df, provider="alpaca") == 2

    loaded_intraday = store.load_intraday_option_flow(
        symbol="AAPL",
        market_date="2026-02-05",
        provider="alpaca",
    )
    assert list(loaded_intraday.columns) == list(INTRADAY_FLOW_CONTRACT_FIELDS) + [
        "provider",
        "updated_at",
    ]
    assert len(loaded_intraday) == 2
    assert list(loaded_intraday["contract_symbol"]) == [
        "AAPL260212P00185000",
        "AAPL260212C00190000",
    ]
    updated_intraday = loaded_intraday[loaded_intraday["contract_symbol"] == "AAPL260212C00190000"].iloc[0]
    assert float(updated_intraday["net_notional"]) == 5600.0
    assert updated_intraday["warnings"] == ["updated"]

    filtered_intraday = store.load_intraday_option_flow(
        symbol="AAPL",
        market_date="2026-02-05",
        provider="alpaca",
        contract_symbol="AAPL260212C00190000",
    )
    assert len(filtered_intraday) == 1

    loaded_intraday_latest = store.load_intraday_option_flow(symbol="AAPL", provider="alpaca")
    assert len(loaded_intraday_latest) == 2


def test_store_factory_builds_research_metrics_store_for_duckdb(tmp_path) -> None:
    storage_token = set_default_storage_backend("duckdb")
    path_token = set_default_duckdb_path(tmp_path / "factory.duckdb")
    try:
        close_warehouses()
        store = get_research_metrics_store(tmp_path / "research")
        assert isinstance(store, DuckDBResearchMetricsStore)
        info = ensure_schema(store.warehouse)
        assert info.schema_version == 5
    finally:
        close_warehouses()
        reset_default_duckdb_path(path_token)
        reset_default_storage_backend(storage_token)
