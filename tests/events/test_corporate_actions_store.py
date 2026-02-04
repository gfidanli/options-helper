from __future__ import annotations

from datetime import date

from options_helper.data.corporate_actions_store import CorporateActionsStore


def test_corporate_actions_store_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = CorporateActionsStore(tmp_path)
    actions = [
        {
            "type": "split",
            "symbol": "AAPL",
            "ex_date": "2026-02-03",
            "ratio": 2.0,
        },
        {
            "type": "split",
            "symbol": "AAPL",
            "ex_date": "2026-02-03",
            "ratio": 2.0,
        },
        {
            "type": "dividend",
            "symbol": "AAPL",
            "ex_date": "2026-02-10",
            "cash_amount": 0.25,
        },
    ]

    path = store.save("AAPL", actions, meta={"source": "alpaca"})
    assert path.exists()

    loaded = store.load("AAPL")
    assert loaded["symbol"] == "AAPL"
    assert loaded["rows"] == 2
    assert loaded["coverage_start"] == "2026-02-03"
    assert loaded["coverage_end"] == "2026-02-10"

    results = store.query("AAPL", start=date(2026, 2, 3), end=date(2026, 2, 3))
    assert len(results) == 1
    assert results[0]["type"] == "split"
