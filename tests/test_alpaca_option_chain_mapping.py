from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.providers.base import OPTION_CHAIN_REQUIRED_COLUMNS
from options_helper.data.providers.alpaca import AlpacaProvider


class _StubClient:
    def __init__(self, payload: dict, contracts: list[dict]) -> None:
        self.payload = payload
        self.contracts = contracts
        self.calls: list[dict] = []
        self.options_feed = "opra"
        self.provider_version = "test"

    def get_option_chain_snapshots(self, underlying: str, *, expiry: date, feed: str | None = None):
        self.calls.append({"underlying": underlying, "expiry": expiry, "feed": feed})
        return self.payload

    def list_option_contracts(
        self,
        underlying: str,
        *,
        exp_gte: date | None = None,
        exp_lte: date | None = None,
        limit: int | None = None,
        page_limit: int | None = None,
    ) -> list[dict]:
        self.calls.append(
            {
                "underlying": underlying,
                "exp_gte": exp_gte,
                "exp_lte": exp_lte,
                "limit": limit,
                "page_limit": page_limit,
            }
        )
        return list(self.contracts)


def test_alpaca_option_chain_raw_splits_and_enriches(tmp_path: Path) -> None:
    expiry = date(2026, 2, 21)
    payload = {
        "data": {
            "SPY260221C00450000": {
                "latest_quote": {"bid_price": 1.0, "ask_price": 1.2},
                "latest_trade": {"price": 1.1, "timestamp": datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc)},
                "implied_volatility": 0.2,
            },
            "SPY260221P00450000": {
                "latest_quote": {"bid_price": 0.9, "ask_price": 1.1},
                "latest_trade": {"price": 1.0, "timestamp": datetime(2026, 2, 3, 15, 31, tzinfo=timezone.utc)},
            },
        }
    }
    contracts = [
        {
            "symbol": "SPY260221C00450000",
            "underlying_symbol": "SPY",
            "expiration_date": expiry,
            "option_type": "call",
            "strike_price": 450,
            "open_interest": 123,
            "open_interest_date": "2026-02-02",
        }
    ]

    stub = _StubClient(payload, contracts)
    store = OptionContractsStore(tmp_path / "contracts")
    provider = AlpacaProvider(client=stub, contracts_store=store)

    raw = provider.get_options_chain_raw("SPY", expiry)

    assert raw["underlying"]["symbol"] == "SPY"
    assert len(raw["calls"]) == 1
    assert len(raw["puts"]) == 1

    call = raw["calls"][0]
    put = raw["puts"][0]

    assert call["contractSymbol"] == "SPY260221C00450000"
    assert call["optionType"] == "call"
    assert call["strike"] == 450.0
    assert call["openInterest"] == 123
    assert call["impliedVolatility"] == 0.2

    assert put["contractSymbol"] == "SPY260221P00450000"
    assert put["optionType"] == "put"
    assert put["strike"] == 450.0


def test_alpaca_get_options_chain_normalizes(tmp_path: Path) -> None:
    expiry = date(2026, 2, 21)
    payload = {
        "data": {
            "SPY260221C00450000": {
                "latest_quote": {"bid_price": 1.0, "ask_price": 1.2},
                "latest_trade": {"price": 1.1, "timestamp": datetime(2026, 2, 3, 15, 30, tzinfo=timezone.utc)},
            },
            "SPY260221P00450000": {
                "latest_quote": {"bid_price": 0.9, "ask_price": 1.1},
                "latest_trade": {"price": 1.0, "timestamp": datetime(2026, 2, 3, 15, 31, tzinfo=timezone.utc)},
            },
        }
    }
    contracts = [
        {
            "symbol": "SPY260221C00450000",
            "underlying_symbol": "SPY",
            "expiration_date": expiry,
            "option_type": "call",
            "strike_price": 450,
        },
        {
            "symbol": "SPY260221P00450000",
            "underlying_symbol": "SPY",
            "expiration_date": expiry,
            "option_type": "put",
            "strike_price": 450,
        },
    ]

    stub = _StubClient(payload, contracts)
    store = OptionContractsStore(tmp_path / "contracts")
    provider = AlpacaProvider(client=stub, contracts_store=store)

    chain = provider.get_options_chain("SPY", expiry)

    for col in OPTION_CHAIN_REQUIRED_COLUMNS:
        assert col in chain.calls.columns
        assert col in chain.puts.columns
    assert chain.calls.loc[0, "optionType"] == "call"
    assert chain.puts.loc[0, "optionType"] == "put"
    assert chain.calls.loc[0, "expiry"] == "2026-02-21"
