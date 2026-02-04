from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from options_helper.data.market_types import DataFetchError
from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.providers.alpaca import AlpacaProvider


class _StubClient:
    def __init__(self, payload: dict, contracts: list[dict], bars_df: pd.DataFrame | None = None) -> None:
        self.payload = payload
        self.contracts = contracts
        self.bars_df = bars_df
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

    def get_option_bars(
        self,
        symbols: list[str],
        *,
        start,
        end,
        interval: str = "1d",
        feed: str | None = None,  # noqa: ARG002
        max_chunk_size: int = 200,  # noqa: ARG002
        max_retries: int = 3,  # noqa: ARG002
    ) -> pd.DataFrame:
        self.calls.append(
            {
                "symbols": list(symbols),
                "start": start,
                "end": end,
                "interval": interval,
            }
        )
        return self.bars_df.copy() if self.bars_df is not None else pd.DataFrame()


class _StubClientFailBars(_StubClient):
    def get_option_bars(self, *args, **kwargs):  # noqa: ARG002
        raise DataFetchError("rate limit")


def _make_payload(expiry: date) -> dict:
    return {
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


def _make_contracts(expiry: date) -> list[dict]:
    return [
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


def test_alpaca_option_volume_join(tmp_path: Path) -> None:
    expiry = date(2026, 2, 21)
    payload = _make_payload(expiry)
    contracts = _make_contracts(expiry)
    bars_df = pd.DataFrame(
        {
            "contractSymbol": ["SPY260221C00450000"],
            "volume": [123],
            "vwap": [1.05],
            "trade_count": [9],
        }
    )

    stub = _StubClient(payload, contracts, bars_df=bars_df)
    store = OptionContractsStore(tmp_path / "contracts")
    provider = AlpacaProvider(client=stub, contracts_store=store)

    raw = provider.get_options_chain_raw("SPY", expiry, snapshot_date=date(2026, 2, 3))

    call = raw["calls"][0]
    put = raw["puts"][0]
    assert call["volume"] == 123
    assert call["vwap"] == 1.05
    assert call["trade_count"] == 9
    assert pd.isna(put.get("volume"))


def test_alpaca_option_volume_join_soft_fail(tmp_path: Path) -> None:
    expiry = date(2026, 2, 21)
    payload = _make_payload(expiry)
    contracts = _make_contracts(expiry)

    stub = _StubClientFailBars(payload, contracts)
    store = OptionContractsStore(tmp_path / "contracts")
    provider = AlpacaProvider(client=stub, contracts_store=store)

    raw = provider.get_options_chain_raw("SPY", expiry, snapshot_date=date(2026, 2, 3))
    call = raw["calls"][0]

    assert "volume" in call
