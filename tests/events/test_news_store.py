from __future__ import annotations

from datetime import date

from options_helper.data.news_store import NewsStore


def test_news_store_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = NewsStore(tmp_path)
    day = date(2026, 2, 3)
    items = [
        {
            "id": "n1",
            "created_at": "2026-02-03T14:30:00Z",
            "headline": "Alpha news",
            "summary": "Summary",
            "source": "Example",
            "symbols": ["AAPL"],
        },
        {
            "id": "n2",
            "created_at": "2026-02-03T15:30:00Z",
            "headline": "Beta news",
            "summary": "Summary",
            "source": "Example",
            "symbols": ["AAPL"],
        },
    ]

    path = store.save_partition("AAPL", day, items, meta={"source": "alpaca"})
    assert path.exists()

    loaded = store.load_partition("AAPL", day)
    assert len(loaded) == 2


def test_news_store_upsert(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = NewsStore(tmp_path)
    items = [
        {
            "id": "n1",
            "created_at": "2026-02-03T14:30:00Z",
            "headline": "Alpha",
            "summary": "Summary",
            "source": "Example",
            "symbols": ["AAPL"],
        }
    ]
    store.upsert_items("AAPL", items)

    updated = [
        {
            "id": "n1",
            "created_at": "2026-02-03T14:30:00Z",
            "headline": "Alpha updated",
            "summary": "Summary",
            "source": "Example",
            "symbols": ["AAPL"],
        },
        {
            "id": "n2",
            "created_at": "2026-02-03T15:30:00Z",
            "headline": "Beta",
            "summary": "Summary",
            "source": "Example",
            "symbols": ["AAPL"],
        },
    ]
    store.upsert_items("AAPL", updated)

    loaded = store.load_partition("AAPL", date(2026, 2, 3))
    assert len(loaded) == 2
