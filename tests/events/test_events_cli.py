from __future__ import annotations

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.commands import events


class _StubProvider:
    name = "alpaca"


class _StubClient:
    provider_version = "test"

    def __init__(self) -> None:
        self.actions_calls: list[dict[str, object]] = []
        self.news_calls: list[dict[str, object]] = []

    def get_corporate_actions(self, symbols, *, start, end, types, limit, page_limit):  # type: ignore[no-untyped-def]
        self.actions_calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "types": types,
                "limit": limit,
                "page_limit": page_limit,
            }
        )
        return [
            {"symbol": "AAPL", "type": "split", "ex_date": "2026-02-03", "ratio": 2.0},
            {"symbol": "MSFT", "type": "dividend", "ex_date": "2026-02-10", "cash_amount": 0.25},
        ]

    def get_news(  # type: ignore[no-untyped-def]
        self,
        symbols,
        *,
        start,
        end,
        include_content,
        limit,
        page_limit,
    ):
        self.news_calls.append(
            {
                "symbols": symbols,
                "start": start,
                "end": end,
                "include_content": include_content,
                "limit": limit,
                "page_limit": page_limit,
            }
        )
        return [
            {
                "id": "n1",
                "created_at": "2026-02-03T14:30:00Z",
                "headline": "Alpha",
                "summary": "Summary",
                "source": "Example",
                "symbols": ["AAPL"],
            }
        ]


def test_events_refresh_corporate_actions(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    runner = CliRunner()
    stub = _StubClient()

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    monkeypatch.setattr(events, "AlpacaClient", lambda: stub)

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "events",
            "refresh-corporate-actions",
            "--symbols",
            "AAPL,MSFT",
            "--start",
            "2026-02-01",
            "--end",
            "2026-02-28",
            "--actions-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "AAPL.json").exists()
    assert (tmp_path / "MSFT.json").exists()


def test_events_refresh_news(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    runner = CliRunner()
    stub = _StubClient()

    monkeypatch.setattr("options_helper.cli_deps.build_provider", lambda: _StubProvider())
    monkeypatch.setattr(events, "AlpacaClient", lambda: stub)

    result = runner.invoke(
        app,
        [
            "--provider",
            "alpaca",
            "events",
            "refresh-news",
            "--symbols",
            "AAPL",
            "--start",
            "2026-02-01",
            "--end",
            "2026-02-28",
            "--news-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "AAPL" / "2026-02-03.jsonl.gz").exists()
