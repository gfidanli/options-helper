from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from apps.streamlit.components.research_metrics_page import (
    build_exposure_top_strikes,
    build_iv_delta_bucket_sparkline,
    build_iv_tenor_sparkline,
    build_latest_iv_delta_table,
    build_latest_iv_tenor_table,
    compute_exposure_flip_strike,
    load_exposure_by_strike_latest,
    load_intraday_flow_summary,
    load_intraday_flow_top_contracts,
    load_intraday_flow_top_strikes,
    load_iv_surface_delta_history,
    load_iv_surface_tenor_history,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_research_metrics(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO iv_surface_tenor(
              symbol, as_of, tenor_target_dte, expiry, dte, tenor_gap_dte, atm_strike, atm_iv,
              atm_mark, straddle_mark, expected_move_pct, skew_25d_pp, skew_10d_pp,
              contracts_used, warnings_json, provider
            )
            VALUES
              ('AAPL', '2026-02-04', 7, '2026-02-11', 7, 0, 100.0, 0.20, 2.1, 4.2, 0.04, -1.0, -1.5, 30, '[]', 'alpaca'),
              ('AAPL', '2026-02-05', 7, '2026-02-12', 7, 0, 101.0, 0.23, 2.4, 4.8, 0.045, -0.8, -1.2, 35, '[]', 'alpaca'),
              ('AAPL', '2026-02-05', 30, '2026-03-06', 29, -1, 101.0, 0.21, 5.6, 11.2, 0.08, -0.4, -0.9, 40, '[]', 'alpaca')
            """
        )
        tx.execute(
            """
            INSERT INTO iv_surface_delta_buckets(
              symbol, as_of, tenor_target_dte, expiry, option_type, delta_bucket,
              avg_iv, median_iv, n_contracts, warnings_json, provider
            )
            VALUES
              ('AAPL', '2026-02-04', 7, '2026-02-11', 'call', 'd20_40', 0.21, 0.20, 5, '[]', 'alpaca'),
              ('AAPL', '2026-02-05', 7, '2026-02-12', 'call', 'd20_40', 0.24, 0.23, 10, '[]', 'alpaca'),
              ('AAPL', '2026-02-05', 7, '2026-02-12', 'put', 'd60_80', 0.29, 0.28, 8, '[]', 'alpaca'),
              ('AAPL', '2026-02-05', 30, '2026-03-06', 'call', 'd40_60', 0.25, 0.24, 6, '[]', 'alpaca')
            """
        )
        tx.execute(
            """
            INSERT INTO dealer_exposure_strikes(
              symbol, as_of, expiry, strike, call_oi, put_oi, call_gex, put_gex, net_gex, provider
            )
            VALUES
              ('AAPL', '2026-02-05', '2026-02-12', 95.0, 1200, 1600, 500.0, 700.0, -200.0, 'alpaca'),
              ('AAPL', '2026-02-05', '2026-02-12', 100.0, 1400, 1300, 850.0, 1000.0, -150.0, 'alpaca'),
              ('AAPL', '2026-02-05', '2026-03-06', 100.0, 300, 100, 300.0, 250.0, 50.0, 'alpaca'),
              ('AAPL', '2026-02-05', '2026-02-12', 105.0, 1900, 700, 2000.0, 1400.0, 600.0, 'alpaca')
            """
        )
        tx.execute(
            """
            INSERT INTO intraday_option_flow(
              symbol, market_date, source, contract_symbol, expiry, option_type, strike, delta_bucket,
              buy_volume, sell_volume, unknown_volume, buy_notional, sell_notional, net_notional,
              trade_count, unknown_trade_share, quote_coverage_pct, warnings_json, provider
            )
            VALUES
              (
                'AAPL', '2026-02-05', 'alpaca_stream', 'AAPL260320C00100000', '2026-03-20',
                'call', 100.0, 'd40_60', 40, 20, 2, 12000, 6000, 6000, 10, 0.10, 0.90, '[]', 'alpaca'
              ),
              (
                'AAPL', '2026-02-05', 'alpaca_stream', 'AAPL260320P00095000', '2026-03-20',
                'put', 95.0, 'd60_80', 12, 28, 1, 3000, 9000, -6000, 9, 0.12, 0.88, '[]', 'alpaca'
              ),
              (
                'AAPL', '2026-02-05', 'tape', 'AAPL260320C00100000', '2026-03-20',
                'call', 100.0, 'd40_60', 15, 4, 0, 4000, 1000, 3000, 4, 0.00, 1.00, '[]', 'alpaca'
              )
            """
        )


def test_research_metrics_query_helpers(tmp_path: Path) -> None:
    db_path = tmp_path / "research.duckdb"
    _seed_research_metrics(db_path)

    tenor_df, tenor_note = load_iv_surface_tenor_history("AAPL", database_path=db_path)
    assert tenor_note is None
    assert len(tenor_df) == 3
    assert tenor_df["as_of"].is_monotonic_increasing

    tenor_spark = build_iv_tenor_sparkline(tenor_df)
    assert list(tenor_spark.columns) == ["7d", "30d"]

    latest_tenor = build_latest_iv_tenor_table(tenor_df)
    assert len(latest_tenor) == 2
    assert latest_tenor["tenor_target_dte"].tolist() == [7, 30]

    delta_df, delta_note = load_iv_surface_delta_history("AAPL", database_path=db_path)
    assert delta_note is None
    assert len(delta_df) == 4
    assert delta_df["as_of"].is_monotonic_increasing

    delta_spark = build_iv_delta_bucket_sparkline(delta_df, max_series=2)
    assert len(delta_spark.columns) == 2

    latest_delta = build_latest_iv_delta_table(delta_df, top_n=20)
    assert len(latest_delta) == 3
    assert set(latest_delta["delta_bucket"].tolist()) == {"d20_40", "d60_80", "d40_60"}

    exposure_df, exposure_note = load_exposure_by_strike_latest("AAPL", database_path=db_path)
    assert exposure_note is None
    assert len(exposure_df) == 3
    assert float(exposure_df[exposure_df["strike"] == 100.0].iloc[0]["net_gex"]) == pytest.approx(-100.0)

    flip_strike = compute_exposure_flip_strike(exposure_df)
    assert flip_strike == pytest.approx(102.5)

    top_exposure = build_exposure_top_strikes(exposure_df, top_n=2)
    assert len(top_exposure) == 2
    assert float(top_exposure.iloc[0]["abs_net_gex"]) >= float(top_exposure.iloc[1]["abs_net_gex"])

    summary, summary_note = load_intraday_flow_summary("AAPL", database_path=db_path)
    assert summary_note is None
    assert summary is not None
    assert summary["market_date"] == "2026-02-05"
    assert summary["contracts"] == 3
    assert summary["trade_count"] == 23
    assert summary["net_notional"] == pytest.approx(3000.0)

    top_strikes_df, top_strikes_note = load_intraday_flow_top_strikes("AAPL", database_path=db_path, top_n=10)
    assert top_strikes_note is None
    assert len(top_strikes_df) == 2
    assert float(top_strikes_df.iloc[0]["net_notional"]) == pytest.approx(9000.0)

    top_contracts_df, top_contracts_note = load_intraday_flow_top_contracts("AAPL", database_path=db_path, top_n=10)
    assert top_contracts_note is None
    assert len(top_contracts_df) == 2
    assert str(top_contracts_df.iloc[0]["contract_symbol"]) == "AAPL260320C00100000"
    assert int(top_contracts_df.iloc[0]["source_rows"]) == 2
    assert float(top_contracts_df.iloc[0]["net_notional"]) == pytest.approx(9000.0)


def test_research_metrics_query_helpers_missing_db_or_table(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"
    tenor_df, tenor_note = load_iv_surface_tenor_history("AAPL", database_path=missing_db)
    assert tenor_df.empty
    assert tenor_note is not None
    assert "not found" in tenor_note.lower()

    assert build_iv_tenor_sparkline(pd.DataFrame()).empty
    assert build_iv_delta_bucket_sparkline(pd.DataFrame()).empty
    assert compute_exposure_flip_strike(pd.DataFrame()) is None

    raw_db = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(raw_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
        conn.execute("INSERT INTO sample VALUES (1)")
    finally:
        conn.close()

    delta_df, delta_note = load_iv_surface_delta_history("AAPL", database_path=raw_db)
    assert delta_df.empty
    assert delta_note is not None
    assert "iv_surface_delta_buckets table not found" in delta_note

    exposure_df, exposure_note = load_exposure_by_strike_latest("AAPL", database_path=raw_db)
    assert exposure_df.empty
    assert exposure_note is not None
    assert "dealer_exposure_strikes table not found" in exposure_note

    summary, summary_note = load_intraday_flow_summary("AAPL", database_path=raw_db)
    assert summary is None
    assert summary_note is not None
    assert "intraday_option_flow table not found" in summary_note
