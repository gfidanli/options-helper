from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.data.candles import close_asof
from options_helper.data.options_snapshots import find_snapshot_row


def test_close_asof_picks_last_close_at_or_before_date() -> None:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
    history = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=idx)

    assert close_asof(history, date(2026, 1, 5)) == 3.0
    assert close_asof(history, date(2026, 1, 4)) == 2.0
    assert close_asof(history, date(2025, 12, 31)) is None


def test_find_snapshot_row_contract_symbol_takes_precedence() -> None:
    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "expiry": "2026-04-17",
                "optionType": "call",
                "strike": 5.0,
            },
            {
                "contractSymbol": "AAA260417P00005000",
                "expiry": "2026-04-17",
                "optionType": "put",
                "strike": 5.0,
            },
        ]
    )
    row = find_snapshot_row(
        df,
        expiry=date(2026, 4, 17),
        strike=5.0,
        option_type="call",
        contract_symbol="AAA260417P00005000",
    )
    assert row is not None
    assert str(row["contractSymbol"]) == "AAA260417P00005000"


def test_find_snapshot_row_fallbacks_to_expiry_strike_type() -> None:
    df = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "expiry": "2026-04-17",
                "optionType": "call",
                "strike": 5.0,
            },
            {
                "contractSymbol": "AAA260417P00005000",
                "expiry": "2026-04-17",
                "optionType": "put",
                "strike": 5.0,
            },
        ]
    )
    row = find_snapshot_row(df, expiry=date(2026, 4, 17), strike=5.0, option_type="call")
    assert row is not None
    assert str(row["contractSymbol"]) == "AAA260417C00005000"


def test_find_snapshot_row_infers_type_when_optiontype_missing() -> None:
    df = pd.DataFrame(
        [
            {"contractSymbol": "AAA260417C00005000", "expiry": "2026-04-17", "strike": 5.0},
            {"contractSymbol": "AAA260417P00005000", "expiry": "2026-04-17", "strike": 5.0},
        ]
    )
    row = find_snapshot_row(df, expiry=date(2026, 4, 17), strike=5.0, option_type="put")
    assert row is not None
    assert str(row["contractSymbol"]) == "AAA260417P00005000"
