from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import apps.streamlit.components.symbol_explorer_page as symbol_explorer_page
from apps.streamlit.components.symbol_explorer_page import (
    build_derived_snippet,
    build_snapshot_strike_table,
    list_available_symbols,
    load_candles_history,
    load_derived_history,
    load_latest_snapshot_header,
    load_snapshot_chain,
    normalize_symbol,
    resolve_symbol_selection,
    summarize_snapshot,
    sync_symbol_query_param,
)
from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_chain_parquet(path: Path) -> None:
    conn = duckdb.connect()
    try:
        escaped = str(path).replace("'", "''")
        conn.execute(
            f"""
            COPY (
              SELECT * FROM (
                VALUES
                  (100.0, 'call', 1000, 120, 0.20),
                  (100.0, 'put', 900, 110, 0.22),
                  (105.0, 'call', 800, 90, 0.24),
                  (95.0, 'put', 700, 80, 0.25)
              ) AS v(strike, optionType, openInterest, volume, impliedVolatility)
            ) TO '{escaped}' (FORMAT PARQUET)
            """
        )
    finally:
        conn.close()


def _seed_database(db_path: Path, chain_path: Path) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)

    with warehouse.transaction() as tx:
        tx.execute(
            """
            INSERT INTO candles_daily(
              symbol, interval, auto_adjust, back_adjust, ts, open, high, low, close, volume
            )
            VALUES
              ('AAPL', '1d', true, false, '2026-02-03T00:00:00', 100, 103, 99, 102, 1000),
              ('AAPL', '1d', false, false, '2026-02-03T00:00:00', 99, 101, 98, 100, 900),
              ('AAPL', '1d', true, false, '2026-02-04T00:00:00', 102, 106, 101, 105, 1200)
            """
        )
        tx.execute(
            """
            INSERT INTO options_snapshot_headers(
              symbol, snapshot_date, provider, chain_path, meta_path, raw_path, spot, risk_free_rate, contracts
            )
            VALUES (?, '2026-02-04', 'alpaca', ?, NULL, NULL, 101.0, 0.04, 4)
            """,
            ["AAPL", str(chain_path)],
        )
        tx.execute(
            """
            INSERT INTO derived_daily(
              symbol, date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike, atm_iv_near,
              em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d, atm_iv_near_percentile, iv_term_slope
            )
            VALUES
              ('AAPL', '2026-02-03', 100.0, 0.95, 1.02, 110, 90, 100, 0.21, 0.03, -0.02, 0.18, 0.20, 1.16, 0.40, 0.01),
              ('AAPL', '2026-02-04', 101.0, 1.05, 1.01, 112, 92, 100, 0.22, 0.031, -0.01, 0.18, 0.20, 1.22, 0.48, 0.015)
            """
        )


def test_symbol_helpers_and_query_param_sync() -> None:
    assert normalize_symbol(" aapl ") == "AAPL"
    assert normalize_symbol("spy$%") == "SPY"
    assert resolve_symbol_selection(["MSFT", "AAPL"], query_symbol="AAPL") == "AAPL"
    assert resolve_symbol_selection(["MSFT", "AAPL"], query_symbol="QQQ") == "MSFT"

    params: dict[str, str] = {}
    assert sync_symbol_query_param(symbol="SPY", query_params=params) == "SPY"
    assert "symbol" not in params

    assert sync_symbol_query_param(symbol="AAPL", query_params=params) == "AAPL"
    assert params["symbol"] == "AAPL"


def test_symbol_explorer_queries_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "options.duckdb"
    chain_path = tmp_path / "chain.parquet"
    _seed_chain_parquet(chain_path)
    _seed_database(db_path, chain_path)

    symbols, notes = list_available_symbols(database_path=db_path)
    assert "AAPL" in symbols
    assert notes == []

    candles_df, candles_note = load_candles_history("AAPL", database_path=db_path, limit=10)
    assert candles_note is None
    assert list(candles_df["close"]) == [102.0, 105.0]
    assert candles_df["ts"].is_monotonic_increasing

    header, header_note = load_latest_snapshot_header("AAPL", database_path=db_path)
    assert header_note is None
    assert header is not None
    assert str(header["snapshot_date"]).startswith("2026-02-04")

    chain_df, chain_note = load_snapshot_chain(header["chain_path"], database_path=db_path)
    assert chain_note is None
    assert len(chain_df) == 4

    summary = summarize_snapshot(snapshot_header=header, chain_df=chain_df)
    assert summary["contracts"] == 4
    assert summary["total_open_interest"] == 3400.0
    assert summary["put_call_oi_ratio"] == 1600.0 / 1800.0
    assert summary["atm_implied_volatility"] is not None

    strike_df = build_snapshot_strike_table(chain_df, top_n=2)
    assert isinstance(strike_df, pd.DataFrame)
    assert len(strike_df) == 2
    assert set(["strike", "call_oi", "put_oi", "total_oi"]).issubset(strike_df.columns)

    derived_df, derived_note = load_derived_history("AAPL", database_path=db_path, limit=10)
    assert derived_note is None
    assert len(derived_df) == 2
    assert derived_df["date"].is_monotonic_increasing

    snippet = build_derived_snippet(derived_df)
    assert snippet is not None
    assert snippet["as_of"] == "2026-02-04"
    assert snippet["spot"] == 101.0
    assert snippet["spot_change_1d"] == 0.01
    assert snippet["atm_iv_change_1d"] == pytest.approx(0.01)


def test_snapshot_header_triggers_alpaca_backfill_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "options.duckdb"
    chain_path = tmp_path / "chain.parquet"
    _seed_chain_parquet(chain_path)
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)

    calls: list[tuple[str, bool]] = []

    def _fake_backfill(symbol: str, *, database_path: Path, include_derived: bool) -> str | None:
        calls.append((symbol, include_derived))
        with warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO options_snapshot_headers(
                  symbol, snapshot_date, provider, chain_path, meta_path, raw_path, spot, risk_free_rate, contracts
                )
                VALUES (?, '2026-02-05', 'alpaca', ?, NULL, NULL, 180.0, 0.0, 4)
                """,
                [symbol, str(chain_path)],
            )
        return None

    monkeypatch.setattr(symbol_explorer_page, "_maybe_backfill_symbol_from_alpaca", _fake_backfill)

    header, note = load_latest_snapshot_header("CVX", database_path=db_path)
    assert note is None
    assert header is not None
    assert str(header["snapshot_date"]).startswith("2026-02-05")
    assert header["provider"] == "alpaca"
    assert calls == [("CVX", False)]


def test_derived_history_triggers_alpaca_backfill_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "options.duckdb"
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)

    calls: list[tuple[str, bool]] = []

    def _fake_backfill(symbol: str, *, database_path: Path, include_derived: bool) -> str | None:
        calls.append((symbol, include_derived))
        with warehouse.transaction() as tx:
            tx.execute(
                """
                INSERT INTO derived_daily(
                  symbol, date, spot, pc_oi, pc_vol, call_wall, put_wall, gamma_peak_strike, atm_iv_near,
                  em_near_pct, skew_near_pp, rv_20d, rv_60d, iv_rv_20d, atm_iv_near_percentile, iv_term_slope
                )
                VALUES
                  (?, '2026-02-04', 180.0, 0.9, 1.1, 185, 170, 180, 0.30, 0.02, -0.01, 0.20, 0.24, 1.5, 0.7, -0.02),
                  (?, '2026-02-05', 182.0, 1.0, 1.0, 190, 172, 180, 0.32, 0.03, -0.01, 0.21, 0.25, 1.52, 0.8, -0.01)
                """,
                [symbol, symbol],
            )
        return None

    monkeypatch.setattr(symbol_explorer_page, "_maybe_backfill_symbol_from_alpaca", _fake_backfill)

    derived_df, note = load_derived_history("CVX", database_path=db_path, limit=10)
    assert note is None
    assert len(derived_df) == 2
    assert list(derived_df["date"].dt.date.astype(str)) == ["2026-02-04", "2026-02-05"]
    assert calls == [("CVX", True)]
