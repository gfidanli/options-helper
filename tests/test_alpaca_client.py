from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from options_helper.data import alpaca_client
from options_helper.data.market_types import DataFetchError


class _StubClient:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _StubTimeFrame:
    Day = "1Day"


class _StubStockHistoricalDataClient:
    def __init__(
        self,
        *,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        feed: str | None = None,
        data_feed: str | None = None,
    ) -> None:
        self.api_key_id = api_key_id
        self.api_secret_key = api_secret_key
        self.feed = feed
        self.data_feed = data_feed


class _StubSession:
    def __init__(self) -> None:
        self.mount_calls: list[tuple[str, object]] = []
        self.hooks: dict[str, object] = {}

    def mount(self, prefix: str, adapter: object) -> None:
        self.mount_calls.append((prefix, adapter))


class _StubOptionHistoricalDataClientWithSession:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        self._session = _StubSession()


class _StubHTTPAdapter:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.kwargs = kwargs


def test_alpaca_client_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "OptionHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(alpaca_client, "_maybe_load_alpaca_env", lambda: None)

    client = alpaca_client.AlpacaClient()
    with pytest.raises(DataFetchError) as exc:
        _ = client.trading_client

    assert "APCA_API_KEY_ID" in str(exc.value)


def test_alpaca_option_chain_request_passes_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubOptionClient:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get_option_chain(self, **kwargs):
            self.calls.append(kwargs)
            return {"SPY260320C00450000": {"latest_quote": {"bid_price": 1.0, "ask_price": 1.2}}}

    stub = _StubOptionClient()
    client = alpaca_client.AlpacaClient(api_key_id="key", api_secret_key="secret")
    client._option_client = stub
    monkeypatch.setattr(alpaca_client, "_load_option_chain_request", lambda: None)

    payload = client.get_option_chain_snapshots("BRK-B", expiry=date(2026, 2, 21), feed="opra")

    assert payload
    assert len(stub.calls) == 1
    call = stub.calls[0]
    assert any(
        val == "BRK.B" or (isinstance(val, list) and "BRK.B" in val) for val in call.values()
    )
    assert call.get("expiration_date") == date(2026, 2, 21) or call.get("expiry") == date(2026, 2, 21)
    assert call.get("feed") == "opra"


def test_option_chain_to_rows_flattens_payload() -> None:
    payload = {
        "data": {
            "SPY260320C00450000": {
                "latest_quote": {"bid_price": 1.0, "ask_price": 1.2},
                "latest_trade": {
                    "price": 1.1,
                    "timestamp": datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc),
                },
                "implied_volatility": 0.2,
                "greeks": {"delta": 0.5, "gamma": 0.02, "theta": -0.01, "vega": 0.1, "rho": 0.01},
            }
        }
    }

    rows = alpaca_client.option_chain_to_rows(payload)

    assert len(rows) == 1
    row = rows[0]
    assert row["contractSymbol"] == "SPY260320C00450000"
    assert row["bid"] == 1.0
    assert row["ask"] == 1.2
    assert row["lastPrice"] == 1.1
    assert isinstance(row["lastTradeDate"], str)
    assert row["lastTradeDate"].startswith("2026-02-03")
    assert row["impliedVolatility"] == 0.2
    assert row["delta"] == 0.5


def test_get_stock_bars_preserves_datafetcherror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "OptionHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(alpaca_client, "TimeFrame", _StubTimeFrame)
    monkeypatch.setattr(alpaca_client, "_load_stock_bars_request", lambda: None)
    monkeypatch.setattr(alpaca_client, "_maybe_load_alpaca_env", lambda: None)

    client = alpaca_client.AlpacaClient()
    with pytest.raises(DataFetchError) as exc:
        client.get_stock_bars(
            "AAPL",
            start=None,
            end=None,
            interval="1d",
            adjustment="raw",
        )

    assert "Missing Alpaca credentials" in str(exc.value)


def test_alpaca_client_loads_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    env_path = tmp_path / "alpaca.env"
    env_path.write_text(
        "\n".join(
            [
                "APCA_API_KEY_ID=test_key",
                "APCA_API_SECRET_KEY=test_secret",
                "OH_ALPACA_STOCK_FEED=sip",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("OH_ALPACA_STOCK_FEED", raising=False)
    monkeypatch.setenv("OH_ALPACA_ENV_FILE", str(env_path))

    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubStockHistoricalDataClient)
    monkeypatch.setattr(alpaca_client, "OptionHistoricalDataClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)

    client = alpaca_client.AlpacaClient()
    assert client.stock_feed == "sip"

    stock_client = client.stock_client
    assert isinstance(stock_client, _StubStockHistoricalDataClient)
    assert stock_client.api_key_id == "test_key"
    assert stock_client.api_secret_key == "test_secret"
    assert stock_client.feed == "sip"


def test_alpaca_client_applies_http_pool_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubClient)
    monkeypatch.setattr(
        alpaca_client, "OptionHistoricalDataClient", _StubOptionHistoricalDataClientWithSession
    )
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(alpaca_client, "HTTPAdapter", _StubHTTPAdapter)

    client = alpaca_client.AlpacaClient(
        api_key_id="key",
        api_secret_key="secret",
        http_pool_connections=32,
        http_pool_maxsize=128,
    )
    option_client = client.option_client
    session = option_client._session

    assert len(session.mount_calls) == 2
    prefixes = [prefix for prefix, _ in session.mount_calls]
    assert prefixes == ["https://", "http://"]
    for _, adapter in session.mount_calls:
        assert isinstance(adapter, _StubHTTPAdapter)
        assert adapter.kwargs["pool_connections"] == 32
        assert adapter.kwargs["pool_maxsize"] == 128
        assert adapter.kwargs["pool_block"] is False


def test_alpaca_client_applies_http_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OH_ALPACA_HTTP_POOL_CONNECTIONS", "48")
    monkeypatch.setenv("OH_ALPACA_HTTP_POOL_MAXSIZE", "96")
    monkeypatch.setattr(alpaca_client, "StockHistoricalDataClient", _StubClient)
    monkeypatch.setattr(
        alpaca_client, "OptionHistoricalDataClient", _StubOptionHistoricalDataClientWithSession
    )
    monkeypatch.setattr(alpaca_client, "TradingClient", _StubClient)
    monkeypatch.setattr(alpaca_client, "_ALPACA_IMPORT_ERROR", None)
    monkeypatch.setattr(alpaca_client, "HTTPAdapter", _StubHTTPAdapter)

    client = alpaca_client.AlpacaClient(api_key_id="key", api_secret_key="secret")
    option_client = client.option_client
    session = option_client._session

    assert len(session.mount_calls) == 2
    for _, adapter in session.mount_calls:
        assert isinstance(adapter, _StubHTTPAdapter)
        assert adapter.kwargs["pool_connections"] == 48
        assert adapter.kwargs["pool_maxsize"] == 96
