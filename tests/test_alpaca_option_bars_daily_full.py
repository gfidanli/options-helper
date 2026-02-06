from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.data import alpaca_client
from options_helper.data.alpaca_client import AlpacaClient
from options_helper.data.market_types import DataFetchError


class _StubTimeFrame:
    Day = "1Day"


class _StubTimeFrameUnit:
    Day = "Day"


class _StubOptionBarsResponse:
    def __init__(self, df: pd.DataFrame, *, next_page_token: str | None = None) -> None:
        self.df = df
        self.next_page_token = next_page_token


class _StubOptionClient:
    def __init__(self, responses: dict[str | None, _StubOptionBarsResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get_option_bars(self, **kwargs):
        self.calls.append(kwargs)
        token = kwargs.get("page_token")
        return self.responses.get(token)


def _setup_timeframe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "TimeFrameUnit", _StubTimeFrameUnit)


def test_get_option_bars_daily_full_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_timeframe(monkeypatch)
    monkeypatch.setattr(alpaca_client, "_load_option_bars_request", lambda: None)

    idx_page1 = pd.MultiIndex.from_product(
        [["SPY260621C00100000"], pd.to_datetime(["2026-02-01T00:00:00Z"], utc=True)],
        names=["symbol", "timestamp"],
    )
    df_page1 = pd.DataFrame(
        {"o": [1.0], "h": [1.2], "l": [0.9], "c": [1.1], "v": [10], "vw": [1.05], "n": [2]},
        index=idx_page1,
    )

    idx_page2 = pd.MultiIndex.from_product(
        [["SPY260621P00100000"], pd.to_datetime(["2026-02-02T00:00:00Z"], utc=True)],
        names=["symbol", "timestamp"],
    )
    df_page2 = pd.DataFrame(
        {"o": [2.0], "h": [2.2], "l": [1.9], "c": [2.1], "v": [20], "vw": [2.05], "n": [4]},
        index=idx_page2,
    )

    responses = {
        None: _StubOptionBarsResponse(df_page1, next_page_token="next"),
        "next": _StubOptionBarsResponse(df_page2, next_page_token=None),
    }
    stub = _StubOptionClient(responses)
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._option_client = stub

    out = client.get_option_bars_daily_full(
        ["SPY260621C00100000", "SPY260621P00100000"],
        start=date(2026, 2, 1),
        end=date(2026, 2, 2),
        interval="1d",
        chunk_size=2,
        page_limit=5,
    )

    assert len(stub.calls) == 2
    assert stub.calls[0].get("limit") is None
    assert stub.calls[1].get("page_token") == "next"
    assert list(out.columns) == [
        "contractSymbol",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "trade_count",
    ]
    assert set(out["contractSymbol"]) == {"SPY260621C00100000", "SPY260621P00100000"}
    assert out["volume"].sum() == 30
    assert out["trade_count"].max() == 4


def test_get_option_bars_daily_full_page_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_timeframe(monkeypatch)
    monkeypatch.setattr(alpaca_client, "_load_option_bars_request", lambda: None)

    idx = pd.MultiIndex.from_product(
        [["SPY260621C00100000"], pd.to_datetime(["2026-02-01T00:00:00Z"], utc=True)],
        names=["symbol", "timestamp"],
    )
    df = pd.DataFrame(
        {"o": [1.0], "h": [1.2], "l": [0.9], "c": [1.1], "v": [10]},
        index=idx,
    )

    responses = {
        None: _StubOptionBarsResponse(df, next_page_token="next"),
        "next": _StubOptionBarsResponse(df, next_page_token=None),
    }
    stub = _StubOptionClient(responses)
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._option_client = stub

    with pytest.raises(DataFetchError) as exc:
        client.get_option_bars_daily_full(
            ["SPY260621C00100000"],
            start=date(2026, 2, 1),
            end=date(2026, 2, 2),
            interval="1d",
            chunk_size=1,
            page_limit=1,
        )

    assert "page limit" in str(exc.value).lower()
