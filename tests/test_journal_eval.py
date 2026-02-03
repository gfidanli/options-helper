from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from options_helper.analysis.journal_eval import build_journal_report
from options_helper.data.journal import SignalContext, SignalEvent


def test_build_journal_report_computes_returns() -> None:
    history = pd.DataFrame(
        {"Close": [100.0, 102.0, 101.0, 104.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]),
    )

    event = SignalEvent(
        date=date(2026, 1, 2),
        symbol="AAA",
        context=SignalContext.POSITION,
        payload={"advice": {"action": "HOLD"}},
        snapshot_date=date(2026, 1, 2),
        contract_symbol="AAA260417C00005000",
    )

    snap_start = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "bid": 1.0,
                "ask": 1.0,
                "lastPrice": 1.0,
            }
        ]
    )
    snap_end = pd.DataFrame(
        [
            {
                "contractSymbol": "AAA260417C00005000",
                "bid": 1.5,
                "ask": 1.5,
                "lastPrice": 1.5,
            }
        ]
    )

    snapshots = {
        ("AAA", date(2026, 1, 2)): snap_start,
        ("AAA", date(2026, 1, 5)): snap_end,
    }

    def _snapshot_loader(symbol: str, snapshot_date: date) -> pd.DataFrame | None:
        return snapshots.get((symbol.upper(), snapshot_date))

    report = build_journal_report(
        [event],
        history_by_symbol={"AAA": history},
        horizons=[1],
        snapshot_loader=_snapshot_loader,
        top_n=1,
    )

    assert report["summary"]["position"][1]["underlying"]["count"] == 1
    outcome = report["events"][0]["outcomes"]["1"]
    assert outcome["underlying_return"] == pytest.approx((101.0 / 102.0) - 1.0)
    assert outcome["option_return"] == pytest.approx((1.5 / 1.0) - 1.0)
