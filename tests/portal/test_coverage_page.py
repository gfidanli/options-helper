from __future__ import annotations

from pathlib import Path

import pytest

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_db(path: Path) -> None:
    wh = DuckDBWarehouse(path)
    ensure_schema(wh)
    wh.execute(
        """
        INSERT INTO candles_daily(symbol, interval, auto_adjust, back_adjust, ts, open, high, low, close, volume)
        VALUES
          ('SPY', '1d', TRUE, FALSE, '2026-01-07', 600, 601, 599, 600, 1000),
          ('AAPL', '1d', TRUE, FALSE, '2026-01-07', 200, 201, 199, 200, 2000)
        """
    )
    wh.execute(
        """
        INSERT INTO option_contracts(contract_symbol, underlying, expiry, option_type, strike, multiplier, provider)
        VALUES ('SPY260220C00600000', 'SPY', '2026-02-20', 'call', 600, 100, 'alpaca')
        """
    )


def test_coverage_page_helpers_with_seeded_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import coverage_page

    db_path = tmp_path / "portal.duckdb"
    _seed_db(db_path)

    symbols, note = coverage_page.list_coverage_symbols(database_path=str(db_path))
    assert note is None
    assert "SPY" in symbols

    payload, payload_note = coverage_page.load_coverage_payload(
        symbol="spy",
        lookback_days=30,
        database_path=str(db_path),
    )
    assert payload_note is None
    assert payload is not None
    assert payload["symbol"] == "SPY"
    assert "candles" in payload


def test_coverage_page_helpers_handle_missing_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import coverage_page

    missing_path = tmp_path / "missing.duckdb"
    symbols, note = coverage_page.list_coverage_symbols(database_path=str(missing_path))
    assert symbols == []
    assert note is not None
