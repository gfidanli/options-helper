from __future__ import annotations

from datetime import date

import pytest

from options_helper.data import alpaca_client
from options_helper.data.alpaca_client import AlpacaClient


class _StubCorporateActionsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_corporate_actions(self, **kwargs):
        self.calls.append(kwargs)
        token = kwargs.get("page_token")
        if token is None:
            return {
                "corporate_actions": [
                    {
                        "symbol": "AAPL",
                        "type": "split",
                        "ex_date": "2026-02-03",
                        "ratio": 2.0,
                    }
                ],
                "next_page_token": "page2",
            }
        return {
            "corporate_actions": [
                {
                    "symbol": "AAPL",
                    "type": "dividend",
                    "ex_date": "2026-02-10",
                    "cash_amount": 0.25,
                }
            ],
            "next_page_token": None,
        }


class _StubNewsClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_news(self, **kwargs):
        self.calls.append(kwargs)
        token = kwargs.get("page_token")
        if token is None:
            return {
                "news": [
                    {
                        "id": "n1",
                        "created_at": "2026-02-03T14:30:00Z",
                        "headline": "Alpha",
                        "summary": "Summary",
                        "source": "Example",
                        "symbols": ["AAPL"],
                    }
                ],
                "next_page_token": "page2",
            }
        return {
            "news": [
                {
                    "id": "n2",
                    "created_at": "2026-02-03T15:30:00Z",
                    "headline": "Beta",
                    "summary": "Summary",
                    "source": "Example",
                    "symbols": ["AAPL"],
                }
            ],
            "next_page_token": None,
        }


def test_get_corporate_actions_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_client, "_load_corporate_actions_request", lambda: None)
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    stub = _StubCorporateActionsClient()
    client._corporate_actions_client = stub

    actions = client.get_corporate_actions(
        ["AAPL"],
        start=date(2026, 2, 1),
        end=date(2026, 2, 28),
    )

    assert len(actions) == 2
    assert stub.calls
    assert actions[0]["symbol"] == "AAPL"
    assert actions[0]["type"] == "split"


def test_get_news_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_client, "_load_news_request", lambda: None)
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    stub = _StubNewsClient()
    client._news_client = stub

    items = client.get_news(
        ["AAPL"],
        start=date(2026, 2, 1),
        end=date(2026, 2, 28),
        include_content=False,
    )

    assert len(items) == 2
    assert stub.calls
    assert items[0]["symbols"] == ["AAPL"]
