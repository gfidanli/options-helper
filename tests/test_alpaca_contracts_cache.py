from __future__ import annotations

import json
from datetime import date

import pandas as pd

from options_helper.data.option_contracts import OptionContractsStore


def test_option_contracts_store_round_trip(tmp_path) -> None:
    store = OptionContractsStore(tmp_path)
    as_of = date(2026, 2, 3)
    df = pd.DataFrame(
        {
            "contractSymbol": ["SPY260320C00450000", "SPY260320P00425000"],
            "underlying": ["SPY", "SPY"],
            "expiry": ["2026-03-20", "2026-03-20"],
            "optionType": ["call", "put"],
            "strike": [450.0, 425.0],
            "multiplier": [100, 100],
            "openInterest": [1234, 567],
            "openInterestDate": ["2026-02-02", "2026-02-02"],
        }
    )
    raw = {"contracts": [{"symbol": "SPY260320C00450000"}]}
    meta = {"provider": "alpaca", "provider_version": "0.0.0-test"}

    path = store.save("SPY", as_of, df, raw=raw, meta=meta)

    assert path.exists()
    loaded = store.load("SPY", as_of)
    assert loaded is not None
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)

    meta_out = store.load_meta("SPY", as_of)
    assert meta_out["provider"] == "alpaca"
    assert meta_out["provider_version"] == "0.0.0-test"
    assert meta_out["rows"] == 2
    assert meta_out["symbol"] == "SPY"
    assert meta_out["as_of"] == "2026-02-03"

    raw_path = tmp_path / "SPY" / "2026-02-03" / "contracts.raw.json"
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    assert payload["contracts"][0]["symbol"] == "SPY260320C00450000"

