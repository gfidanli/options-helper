from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.alpaca_client import AlpacaClient, contracts_to_df


class _StubPage:
    def __init__(self, contracts, next_page_token=None) -> None:
        self.option_contracts = contracts
        self.next_page_token = next_page_token


class _StubTradingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def get_option_contracts(self, **kwargs):
        self.calls.append(kwargs)
        status = kwargs.get("status")
        if status == "inactive":
            return _StubPage([{"symbol": "BRK240216P00350000"}], None)
        token = kwargs.get("page_token")
        if token == "page2":
            return _StubPage([{"symbol": "BRK240119P00350000"}], None)
        return _StubPage([{"symbol": "BRK240119C00350000"}], "page2")


def test_list_option_contracts_paginates() -> None:
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    stub = _StubTradingClient()
    client._trading_client = stub

    contracts = client.list_option_contracts(
        "BRK-B",
        exp_gte=date(2024, 1, 1),
        exp_lte=date(2024, 2, 1),
        limit=2,
        page_limit=5,
    )

    assert len(contracts) == 2
    assert len(stub.calls) == 2
    assert stub.calls[0].get("page_token") is None
    assert stub.calls[1].get("page_token") == "page2"
    assert any(
        val == "BRK.B" or (isinstance(val, list) and "BRK.B" in val) for val in stub.calls[0].values()
    )


def test_list_option_contracts_supports_all_status() -> None:
    client = AlpacaClient(api_key_id="key", api_secret_key="secret")
    stub = _StubTradingClient()
    client._trading_client = stub

    contracts = client.list_option_contracts(
        "BRK-B",
        exp_gte=date(2024, 1, 1),
        exp_lte=date(2024, 2, 1),
        contract_status="all",
        limit=2,
        page_limit=5,
    )

    statuses = [str(call.get("status")) for call in stub.calls]
    assert "active" in statuses
    assert "inactive" in statuses
    assert any(raw.get("symbol") == "BRK240216P00350000" for raw in contracts)


def test_contracts_to_df_normalizes() -> None:
    raw = [
        {
            "symbol": "SPY260320C00450000",
            "underlying_symbol": "SPY",
            "expiration_date": "2026-03-20",
            "option_type": "call",
            "strike_price": "450",
            "multiplier": 100,
            "open_interest": 1200,
            "open_interest_date": "2026-02-02",
            "close_price": 1.25,
            "close_price_date": date(2026, 2, 2),
        },
        {
            "symbol": "SPY260320P00425000",
            "underlying_symbol": "SPY",
            "expiration_date": date(2026, 3, 20),
            "option_type": "P",
            "strike_price": 425,
            "multiplier": "100",
        },
        {
            "symbol": "BRK240119C00350000",
        },
    ]

    df = contracts_to_df(raw)

    required = [
        "contractSymbol",
        "underlying",
        "expiry",
        "optionType",
        "strike",
        "multiplier",
        "openInterest",
        "openInterestDate",
        "closePrice",
        "closePriceDate",
    ]
    assert list(df.columns) == required
    assert len(df) == 3
    row = df[df["contractSymbol"] == "SPY260320C00450000"].iloc[0]
    assert row["underlying"] == "SPY"
    assert row["expiry"] == "2026-03-20"
    assert row["optionType"] == "call"
    assert row["strike"] == 450.0
    assert row["multiplier"] == 100

    fallback = df[df["contractSymbol"] == "BRK240119C00350000"].iloc[0]
    assert fallback["optionType"] == "call"
    assert fallback["expiry"] == "2024-01-19"
    assert pd.notna(fallback["strike"])
