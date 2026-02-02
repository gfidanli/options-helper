from __future__ import annotations

from datetime import date

import pandas as pd

from options_helper.analysis.options_liquidity import evaluate_liquidity


def test_liquidity_requires_dte_volume_and_oi() -> None:
    snapshot_date = date(2026, 1, 1)
    df = pd.DataFrame(
        {
            "expiry": ["2026-03-20", "2026-04-17", "2026-02-15"],
            "volume": [5, 12, 20],
            "openInterest": [600, 400, 800],
        }
    )

    res = evaluate_liquidity(
        df,
        snapshot_date=snapshot_date,
        min_dte=60,
        min_volume=10,
        min_open_interest=500,
    )
    assert res.is_liquid is False
    assert res.eligible_contracts == 0

    df.loc[0, "volume"] = 10
    df.loc[0, "openInterest"] = 500
    res2 = evaluate_liquidity(
        df,
        snapshot_date=snapshot_date,
        min_dte=60,
        min_volume=10,
        min_open_interest=500,
    )
    assert res2.is_liquid is True
    assert res2.eligible_contracts >= 1
    assert "2026-03-20" in res2.eligible_expiries
