from __future__ import annotations

from pathlib import Path

import pandas as pd

from options_helper.data.derived import DerivedStore
from options_helper.data.scanner import (
    ScannerLiquidityRow,
    ScannerScanRow,
    rank_shortlist_candidates,
)


class _StubCandleStore:
    def __init__(self, history: pd.DataFrame) -> None:
        self._history = history

    def get_daily_history(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        return self._history


def test_rank_shortlist_candidates_extension_score(tmp_path: Path) -> None:
    idx = pd.date_range("2025-01-01", periods=120, freq="B")
    close = pd.Series(range(120), index=idx, dtype="float64")
    history = pd.DataFrame({"Close": close})
    candle_store = _StubCandleStore(history)

    derived_store = DerivedStore(tmp_path)
    (tmp_path / "AAA.csv").write_text("date,iv_rv_20d\n2026-01-31,1.25\n", encoding="utf-8")

    scan_rows = [
        ScannerScanRow(
            symbol="AAA",
            asof="2026-01-31",
            extension_atr=1.5,
            percentile=98.0,
            window_years=3,
            window_bars=756,
            tail=True,
            status="ok",
            error=None,
        )
    ]
    liquidity_rows = [
        ScannerLiquidityRow(
            symbol="AAA",
            snapshot_date="2026-01-31",
            eligible_contracts=10,
            eligible_expiries="2026-04-17,2026-05-15",
            is_liquid=True,
            status="ok",
            error=None,
        )
    ]
    rank_cfg = {
        "weights": {
            "extension": 10.0,
            "weekly_trend": 0.0,
            "rsi_divergence": 0.0,
            "liquidity": 0.0,
            "iv_regime": 0.0,
            "flow": 0.0,
        },
        "extension": {"tail_low": 5.0, "tail_high": 95.0},
    }

    results = rank_shortlist_candidates(
        ["AAA"],
        candle_store=candle_store,
        rank_cfg=rank_cfg,
        scan_rows=scan_rows,
        liquidity_rows=liquidity_rows,
        derived_store=derived_store,
        period="6mo",
    )

    assert "AAA" in results
    assert results["AAA"].score > 50.0
