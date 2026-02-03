from __future__ import annotations

from datetime import date

from options_helper.data.journal import (
    JournalStore,
    SignalContext,
    SignalEvent,
    filter_events,
    index_by_context,
    index_by_date,
    index_by_symbol,
)


def test_journal_store_append_and_read(tmp_path) -> None:
    store = JournalStore(tmp_path)
    event = SignalEvent(
        date=date(2026, 2, 1),
        symbol="aapl",
        context=SignalContext.RESEARCH,
        payload={"action": "watch"},
        snapshot_date=date(2026, 2, 1),
        contract_symbol="AAPL260417C00180000",
    )
    store.append_event(event)

    result = store.read_events()
    assert result.errors == []
    assert len(result.events) == 1

    loaded = result.events[0]
    assert loaded.symbol == "AAPL"
    assert loaded.context == SignalContext.RESEARCH
    assert loaded.payload["action"] == "watch"


def test_journal_store_filters_and_indexes() -> None:
    events = [
        SignalEvent(date=date(2026, 1, 2), symbol="AAPL", context=SignalContext.POSITION, payload={}),
        SignalEvent(date=date(2026, 1, 5), symbol="MSFT", context=SignalContext.RESEARCH, payload={}),
        SignalEvent(date=date(2026, 1, 6), symbol="AAPL", context=SignalContext.SCANNER, payload={}),
    ]

    filtered = filter_events(
        events,
        symbols=["aapl"],
        contexts=["position"],
        start=date(2026, 1, 1),
        end=date(2026, 1, 4),
    )
    assert len(filtered) == 1
    assert filtered[0].context == SignalContext.POSITION

    by_symbol = index_by_symbol(events)
    assert set(by_symbol.keys()) == {"AAPL", "MSFT"}
    assert len(by_symbol["AAPL"]) == 2

    by_date = index_by_date(events)
    assert date(2026, 1, 6) in by_date

    by_context = index_by_context(events)
    assert set(by_context.keys()) == {"position", "research", "scanner"}


def test_journal_store_skips_bad_lines(tmp_path) -> None:
    path = tmp_path / "signal_events.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"date":"2026-02-01","symbol":"AAPL","context":"research"}',
                "{bad",
                "",
            ]
        ),
        encoding="utf-8",
    )
    store = JournalStore(tmp_path)
    result = store.read_events()
    assert len(result.events) == 1
    assert len(result.errors) == 1
