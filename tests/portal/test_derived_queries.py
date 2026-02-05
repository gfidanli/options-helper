from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from apps.streamlit.components.derived_history_page import (
    build_derived_latest_summary,
    list_derived_symbols,
    load_derived_history,
    slice_derived_history_window,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_derived(db_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)
    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO derived_daily(
              symbol, date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike, atm_iv_near,
              em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d, atm_iv_near_percentile, iv_term_slope
            )
            VALUES
              ('AAPL', '2026-02-01', 100.0, 0.95, 1.01, 110, 90, 100, 0.20, 0.03, -0.02, 0.18, 0.20, 1.11, 0.40, 0.010),
              ('AAPL', '2026-02-02', 101.0, 0.99, 1.00, 111, 91, 100, 0.21, 0.03, -0.02, 0.18, 0.20, 1.16, 0.43, 0.011),
              ('AAPL', '2026-02-03', 103.0, 1.02, 0.99, 112, 92, 100, 0.23, 0.03, -0.01, 0.18, 0.20, 1.28, 0.49, 0.013),
              ('MSFT', '2026-02-03', 400.0, 0.88, 0.97, 420, 380, 400, 0.19, 0.02, -0.01, 0.16, 0.18, 1.18, 0.35, 0.008)
            """
        )


def test_derived_query_helpers_window_and_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "derived.duckdb"
    _seed_derived(db_path)

    symbols, symbols_note = list_derived_symbols(database_path=db_path)
    assert symbols_note is None
    assert symbols == ["AAPL", "MSFT"]

    history_df, history_note = load_derived_history("AAPL", database_path=db_path, limit=20)
    assert history_note is None
    assert len(history_df) == 3
    assert history_df["date"].is_monotonic_increasing

    window_df = slice_derived_history_window(history_df, window_days=2)
    assert len(window_df) == 2
    assert list(window_df["date"].dt.date.astype(str)) == ["2026-02-02", "2026-02-03"]

    summary = build_derived_latest_summary(window_df)
    assert summary is not None
    assert summary["as_of"] == "2026-02-03"
    assert summary["spot"] == 103.0
    assert summary["pc_oi"] == 1.02
    assert summary["atm_iv_near"] == 0.23
    assert summary["spot_change_1d"] == pytest.approx((103.0 - 101.0) / 101.0)
    assert summary["atm_iv_change_1d"] == pytest.approx(0.02)
    assert summary["sample_rows"] == 2


def test_derived_query_helpers_handle_missing_db_or_table(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing.duckdb"
    symbols, symbols_note = list_derived_symbols(database_path=missing_db)
    assert symbols == []
    assert symbols_note is not None
    assert "not found" in symbols_note.lower()

    raw_db = tmp_path / "raw.duckdb"
    conn = duckdb.connect(str(raw_db))
    try:
        conn.execute("CREATE TABLE sample(id INTEGER)")
    finally:
        conn.close()

    loaded_df, loaded_note = load_derived_history("AAPL", database_path=raw_db)
    assert loaded_df.empty
    assert loaded_note is not None
    assert "derived_daily table not found" in loaded_note
