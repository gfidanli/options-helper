from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse
from options_helper.data.stores_duckdb import DuckDBOptionContractsStore


def test_duckdb_option_contracts_store_upsert_list_and_snapshots(tmp_path):
    wh = DuckDBWarehouse(tmp_path / "options.duckdb")
    ensure_schema(wh)

    store = DuckDBOptionContractsStore(root_dir=tmp_path / "contracts", warehouse=wh)

    as_of = date(2026, 2, 1)
    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAPL260315C00100000",
                "underlying": "AAPL",
                "expiry": "2026-03-15",
                "optionType": "call",
                "strike": 100.0,
                "multiplier": 100,
                "openInterest": 120,
                "openInterestDate": "2026-02-01",
                "closePrice": 1.23,
                "closePriceDate": "2026-02-01",
            },
            {
                "contractSymbol": "AAPL260221P00095000",
                "underlying": "AAPL",
                "expiry": "2026-02-21",
                "optionType": "put",
                "strike": 95.0,
                "multiplier": 100,
                "openInterest": 80,
                "openInterestDate": "2026-02-01",
                "closePrice": 2.34,
                "closePriceDate": "2026-02-01",
            },
        ]
    )

    store.upsert_contracts(df, provider="alpaca", as_of_date=as_of)

    all_contracts = store.list_contracts("AAPL")
    assert len(all_contracts) == 2
    assert set(all_contracts["contractSymbol"].astype(str)) == {
        "AAPL260315C00100000",
        "AAPL260221P00095000",
    }

    filtered = store.list_contracts("AAPL", exp_gte=date(2026, 3, 1))
    assert len(filtered) == 1
    assert filtered.iloc[0]["contractSymbol"] == "AAPL260315C00100000"

    snapshots = wh.fetch_df(
        """
        SELECT contract_symbol, open_interest, close_price
        FROM option_contract_snapshots
        WHERE as_of_date = ? AND provider = ?
        ORDER BY contract_symbol ASC
        """,
        [as_of, "alpaca"],
    )
    assert len(snapshots) == 2

    df2 = df.copy()
    df2.loc[df2["contractSymbol"] == "AAPL260315C00100000", "openInterest"] = 200
    store.upsert_contracts(df2, provider="alpaca", as_of_date=as_of)

    snapshots2 = wh.fetch_df(
        """
        SELECT contract_symbol, open_interest
        FROM option_contract_snapshots
        WHERE as_of_date = ? AND provider = ?
        ORDER BY contract_symbol ASC
        """,
        [as_of, "alpaca"],
    )
    assert len(snapshots2) == 2
    updated = snapshots2[snapshots2["contract_symbol"] == "AAPL260315C00100000"].iloc[0]
    assert int(updated["open_interest"]) == 200
