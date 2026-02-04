from __future__ import annotations

import pytest

from options_helper.data import alpaca_client
from options_helper.data.market_types import DataFetchError


class _StubClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_alpaca_client_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "OptionHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)

    client = alpaca_client.AlpacaClient()
    with pytest.raises(DataFetchError) as exc:
        _ = client.trading_client

    assert "APCA_API_KEY_ID" in str(exc.value)
