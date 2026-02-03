from datetime import date
from pathlib import Path

import pandas as pd

from options_helper.analysis.osi import ParsedContract, format_osi
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.models import OptionType


def _osi(underlying: str, expiry: date, option_type: OptionType, strike: float) -> str:
    parsed = ParsedContract(
        underlying=underlying,
        underlying_norm=underlying,
        expiry=expiry,
        option_type=option_type,
        strike=strike,
    )
    return format_osi(parsed)


def test_load_day_dedupes_by_osi_when_available(tmp_path: Path) -> None:
    store = OptionsSnapshotStore(tmp_path)
    snapshot_date = date(2026, 1, 2)
    expiry_1 = date(2026, 1, 16)
    expiry_2 = date(2026, 2, 20)
    sym = "AAA"

    osi = _osi("AAA", expiry_1, "call", 100.0)

    df1 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_FIRST",
                "osi": osi,
                "optionType": "call",
                "expiry": expiry_1.isoformat(),
                "strike": 100.0,
                "openInterest": 100,
            }
        ]
    )
    df2 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_SECOND",
                "osi": osi,
                "optionType": "call",
                "expiry": expiry_2.isoformat(),
                "strike": 100.0,
                "openInterest": 200,
            }
        ]
    )

    store.save_expiry_snapshot(sym, snapshot_date, expiry=expiry_1, snapshot=df1)
    store.save_expiry_snapshot(sym, snapshot_date, expiry=expiry_2, snapshot=df2)

    loaded = store.load_day(sym, snapshot_date)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["openInterest"] == 200


def test_load_day_dedupes_by_contract_symbol_when_osi_missing(tmp_path: Path) -> None:
    store = OptionsSnapshotStore(tmp_path)
    snapshot_date = date(2026, 1, 2)
    expiry_1 = date(2026, 1, 16)
    expiry_2 = date(2026, 2, 20)
    sym = "AAA"

    df1 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_DUP",
                "optionType": "call",
                "expiry": expiry_1.isoformat(),
                "strike": 100.0,
                "openInterest": 100,
            }
        ]
    )
    df2 = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_DUP",
                "optionType": "call",
                "expiry": expiry_2.isoformat(),
                "strike": 100.0,
                "openInterest": 250,
            }
        ]
    )

    store.save_expiry_snapshot(sym, snapshot_date, expiry=expiry_1, snapshot=df1)
    store.save_expiry_snapshot(sym, snapshot_date, expiry=expiry_2, snapshot=df2)

    loaded = store.load_day(sym, snapshot_date)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["openInterest"] == 250
