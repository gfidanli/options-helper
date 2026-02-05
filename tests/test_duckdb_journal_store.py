from __future__ import annotations

from datetime import date

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.data.journal import SignalContext, SignalEvent
from options_helper.data.stores_duckdb import DuckDBJournalStore


def test_duckdb_journal_store_roundtrip(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBJournalStore(root_dir=tmp_path / "journal", warehouse=wh)

    e1 = SignalEvent(date=date(2026, 2, 1), symbol="AAPL", context=SignalContext.RESEARCH, payload={"x": 1})
    e2 = SignalEvent(date=date(2026, 2, 2), symbol="MSFT", context=SignalContext.POSITION, payload={"y": "ok"})
    store.append_events([e1, e2])

    result = store.read_events()
    assert not result.errors
    assert len(result.events) == 2
    assert result.events[0].symbol == "AAPL"
    assert result.events[1].symbol == "MSFT"

    # Query filter.
    q = store.query(symbols=["AAPL"])
    assert len(q.events) == 1
    assert q.events[0].symbol == "AAPL"
