from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from options_helper.analysis.research import choose_expiry, select_option_candidate


def test_choose_expiry_picks_closest_to_target() -> None:
    today = date(2026, 1, 1)
    expiries = ["2026-02-15", "2026-03-15", "2026-04-15"]
    # DTE: 45, 73, 104
    exp = choose_expiry(expiries, min_dte=30, max_dte=90, target_dte=60, today=today)
    assert exp == date(2026, 3, 15)


def test_select_option_candidate_prefers_target_delta() -> None:
    expiry = date.today() + timedelta(days=60)
    spot = 100.0

    # Constant IV, monotonic delta by strike.
    df = pd.DataFrame(
        {
            "strike": [90.0, 100.0, 110.0],
            "bid": [12.0, 6.0, 2.5],
            "ask": [13.0, 7.0, 3.0],
            "lastPrice": [12.5, 6.5, 2.75],
            "impliedVolatility": [0.25, 0.25, 0.25],
            "openInterest": [500, 500, 500],
            "volume": [50, 50, 50],
        }
    )

    # LEAPS-style (higher delta) should pick ITM 90 strike.
    itm = select_option_candidate(
        df,
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=0.70,
        window_pct=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert itm is not None
    assert itm.strike == 90.0

    # Short-dated momentum (lower delta) should lean toward higher strike.
    otm = select_option_candidate(
        df,
        symbol="TEST",
        option_type="call",
        expiry=expiry,
        spot=spot,
        target_delta=0.35,
        window_pct=0.50,
        min_open_interest=0,
        min_volume=0,
    )
    assert otm is not None
    assert otm.strike in {100.0, 110.0}

