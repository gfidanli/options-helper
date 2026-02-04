from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from options_helper.data.option_contracts import OptionContractsStore
from options_helper.data.options_snapshotter import snapshot_full_chain_for_symbols
from options_helper.data.providers.alpaca import AlpacaProvider


class _StubClient:
    def __init__(self, payload: dict, contracts: list[dict]) -> None:
        self.payload = payload
        self.contracts = contracts
        self.options_feed = "opra"
        self.stock_feed = "sip"
        self.recent_bars_buffer_minutes = 16
        self.provider_version = "test"

    def get_option_chain_snapshots(self, underlying: str, *, expiry: date, feed: str | None = None):
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
        return list(self.contracts)


def test_snapshot_full_chain_with_alpaca_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch  # type: ignore[name-defined]
) -> None:
    candle_day = date(2026, 2, 3)
    expiry = date(2026, 2, 21)

    def _stub_history(self, symbol: str, *, period: str = "10d"):  # noqa: ARG001
        idx = pd.to_datetime([candle_day.replace(day=candle_day.day - 1), candle_day])
        return pd.DataFrame({"Close": [100.0, 101.0]}, index=idx)

    monkeypatch.setattr(
        "options_helper.data.options_snapshotter.CandleStore.get_daily_history", _stub_history
    )

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

    snapshot_dir = tmp_path / "snapshots"
    candle_dir = tmp_path / "candles"
    results = snapshot_full_chain_for_symbols(
        ["SPY"],
        cache_dir=snapshot_dir,
        candle_cache_dir=candle_dir,
        provider=provider,
    )

    assert results and results[0].status == "ok"
    day_dir = snapshot_dir / "SPY" / candle_day.isoformat()
    assert (day_dir / f"{expiry.isoformat()}.csv").exists()
    assert (day_dir / f"{expiry.isoformat()}.raw.json").exists()
    meta = json.loads((day_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta.get("provider") == "alpaca"
    assert meta.get("provider_params") == {
        "options_feed": "opra",
        "stock_feed": "sip",
        "recent_bars_buffer_minutes": 16,
    }
