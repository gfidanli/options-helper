from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from options_helper.backtesting.roll import RollPolicy, select_roll_candidate, should_roll
from options_helper.data.options_snapshots import OptionsSnapshotStore
from options_helper.models import Position


def test_should_roll_threshold() -> None:
    as_of = date(2026, 1, 10)
    expiry = date(2026, 1, 15)
    assert should_roll(as_of=as_of, expiry=expiry, dte_threshold=7, thesis_ok=True) is True
    assert should_roll(as_of=as_of, expiry=expiry, dte_threshold=3, thesis_ok=True) is False
    assert should_roll(as_of=as_of, expiry=expiry, dte_threshold=7, thesis_ok=False) is False


def test_select_roll_candidate_from_snapshot(tmp_path: Path) -> None:
    store = OptionsSnapshotStore(tmp_path)
    snapshot_date = date(2026, 1, 10)
    expiry_current = date(2026, 2, 20)
    expiry_roll = date(2026, 3, 20)

    current = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260220C00100000",
                "optionType": "call",
                "expiry": expiry_current.isoformat(),
                "strike": 100.0,
                "bid": 1.0,
                "ask": 1.2,
                "lastPrice": 1.1,
                "impliedVolatility": 0.25,
                "openInterest": 100,
                "volume": 20,
            }
        ]
    )
    roll = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260320C00100000",
                "optionType": "call",
                "expiry": expiry_roll.isoformat(),
                "strike": 100.0,
                "bid": 1.5,
                "ask": 1.7,
                "lastPrice": 1.6,
                "impliedVolatility": 0.25,
                "openInterest": 100,
                "volume": 20,
            }
        ]
    )

    store.save_expiry_snapshot("AAA", snapshot_date, expiry=expiry_current, snapshot=current)
    store.save_expiry_snapshot("AAA", snapshot_date, expiry=expiry_roll, snapshot=roll)

    position = Position(
        id="pos-1",
        symbol="AAA",
        option_type="call",
        expiry=expiry_current,
        strike=100.0,
        contracts=1,
        cost_basis=1.0,
    )

    policy = RollPolicy(
        dte_threshold=21,
        horizon_months=3,
        shape="out-same-strike",
        min_open_interest=0,
        min_volume=0,
        max_spread_pct=1.0,
        include_bad_quotes=True,
    )

    decision = select_roll_candidate(
        store,
        symbol="AAA",
        as_of=snapshot_date,
        spot=100.0,
        position=position,
        policy=policy,
    )

    assert decision.candidate is not None
    assert decision.candidate.contract.expiry == expiry_roll.isoformat()
