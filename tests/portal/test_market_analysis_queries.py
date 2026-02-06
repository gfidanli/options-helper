from __future__ import annotations

from pathlib import Path

import pytest

from apps.streamlit.components.market_analysis_page import (
    _compute_tail_risk_cached,
    compute_tail_risk_for_symbol,
    list_candle_symbols,
    load_candles_close,
    load_latest_derived_row,
)
from options_helper.analysis.tail_risk import TailRiskConfig
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_market_tables(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO candles_daily(
              symbol, interval, auto_adjust, back_adjust, ts, open, high, low, close, volume
            )
            SELECT
              'SPY', '1d', TRUE, FALSE,
              TIMESTAMP '2025-01-01' + i * INTERVAL 1 DAY,
              500 + i * 0.2,
              501 + i * 0.2,
              499 + i * 0.2,
              500 + i * 0.2,
              1000000 + i
            FROM range(0, 420) t(i)
            """
        )
        tx.execute(
            """
            INSERT INTO derived_daily(
              symbol, date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike, atm_iv_near,
              em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d, atm_iv_near_percentile, iv_term_slope
            )
            VALUES
              ('SPY', '2026-02-01', 610.0, 0.90, 0.95, 620, 600, 610, 0.21, 0.02, -0.01, 0.17, 0.19, 1.24, 72.0, 0.012)
            """
        )


def test_market_analysis_query_helpers_read_only(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    db_path = tmp_path / "market.duckdb"
    _seed_market_tables(db_path)
    _compute_tail_risk_cached.clear()

    symbols, symbols_note = list_candle_symbols(database_path=db_path)
    assert symbols_note is None
    assert symbols == ["SPY"]

    candles_df, candles_note = load_candles_close("SPY", database_path=db_path, limit=365)
    assert candles_note is None
    assert len(candles_df) == 365
    assert candles_df["ts"].is_monotonic_increasing

    derived_row, derived_note = load_latest_derived_row("SPY", database_path=db_path)
    assert derived_note is None
    assert derived_row is not None
    assert derived_row["iv_rv_20d"] == pytest.approx(1.24)

    warehouse = DuckDBWarehouse(db_path)
    before_rows = int(warehouse.fetch_df("SELECT COUNT(*) AS n FROM candles_daily").iloc[0]["n"])
    cfg = TailRiskConfig(lookback_days=252, horizon_days=20, num_simulations=2000, seed=42)
    result, result_note = compute_tail_risk_for_symbol("SPY", database_path=db_path, config=cfg)
    after_rows = int(warehouse.fetch_df("SELECT COUNT(*) AS n FROM candles_daily").iloc[0]["n"])

    assert result_note is None
    assert result is not None
    assert before_rows == after_rows


def test_market_analysis_query_helpers_missing_inputs(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    missing = tmp_path / "missing.duckdb"
    _compute_tail_risk_cached.clear()

    symbols, symbols_note = list_candle_symbols(database_path=missing)
    assert symbols == []
    assert symbols_note is not None
    assert "not found" in symbols_note.lower()

    cfg = TailRiskConfig(lookback_days=252, horizon_days=20, num_simulations=2000, seed=1)
    result, note = compute_tail_risk_for_symbol("SPY", database_path=missing, config=cfg)
    assert result is None
    assert note is not None

