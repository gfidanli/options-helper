from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from options_helper.db.migrations import ensure_schema
from options_helper.db.warehouse import DuckDBWarehouse


def _seed_candles(db_path: Path, *, symbol: str = "SPY", periods: int = 180) -> None:
    warehouse = DuckDBWarehouse(db_path)
    ensure_schema(warehouse)

    idx = pd.bdate_range(start="2025-01-02", periods=periods)
    base = 100.0 + np.linspace(0.0, 8.0, periods)
    wave = np.sin(np.arange(periods) / 3.0) * 2.0
    close = base + wave
    open_ = close + np.sin(np.arange(periods) / 5.0) * 0.4
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = np.full(periods, 1_000_000.0)

    # Inject periodic bullish/bearish close-through structure breaks.
    for i in range(16, periods, 30):
        high[i] = max(high[i], high[i - 3] + 1.2)
        close[i] = high[i - 3] + 0.4
        open_[i] = close[i] - 0.3
        low[i] = min(low[i], close[i] - 1.0)
    for i in range(28, periods, 30):
        low[i] = min(low[i], low[i - 3] - 1.2)
        close[i] = low[i - 3] - 0.4
        open_[i] = close[i] + 0.3
        high[i] = max(high[i], close[i] + 1.0)

    records = [
        (
            symbol,
            "1d",
            True,
            False,
            ts.to_pydatetime(),
            float(open_[i]),
            float(high[i]),
            float(low[i]),
            float(close[i]),
            float(volume[i]),
        )
        for i, ts in enumerate(idx)
    ]

    with warehouse.transaction() as tx:
        tx.executemany(
            """
            INSERT INTO candles_daily(
              symbol, interval, auto_adjust, back_adjust, ts, open, high, low, close, volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )


def test_msb_page_helpers_with_seeded_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import msb_page

    db_path = tmp_path / "portal.duckdb"
    _seed_candles(db_path, symbol="SPY", periods=220)

    symbols, symbols_note = msb_page.list_msb_symbols(database_path=str(db_path))
    assert symbols_note is None
    assert "SPY" in symbols

    candles, candles_note = msb_page.load_candles_history("SPY", database_path=str(db_path), limit=120)
    assert candles_note is None
    assert not candles.empty
    assert candles["ts"].is_monotonic_increasing

    payload, payload_note = msb_page.load_msb_payload(
        symbol="SPY",
        lookback_days=700,
        tail_low_pct=5.0,
        tail_high_pct=95.0,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        database_path=str(db_path),
    )
    assert payload_note is None
    assert payload is not None
    assert payload["symbol"] == "SPY"
    assert payload["asof"] is not None

    daily_events = payload["daily_events"]
    weekly_events = payload["weekly_events"]
    summary = payload["summary_rows"]
    assert isinstance(daily_events, list)
    assert isinstance(weekly_events, list)
    assert isinstance(summary, list)
    groups = {str(row.get("group")) for row in summary}
    assert "Daily MSB at RSI Extremes (Bullish)" in groups
    assert "Daily MSB at RSI Extremes (Bearish)" in groups
    assert "Weekly MSB + Daily Extension Extreme in Week (Bullish)" in groups
    assert "Weekly MSB + Daily Extension Extreme in Week (Bearish)" in groups
    for row in summary:
        assert "median_1d_pct" in row
        assert "median_5d_pct" in row
        assert "median_10d_pct" in row

    if daily_events:
        ev = daily_events[0]
        assert "T" not in str(ev["event_ts"])
        assert "extension_percentile" in ev
        assert {"forward_1d_pct", "forward_5d_pct", "forward_10d_pct"}.issubset(ev.keys())

    if weekly_events:
        evw = weekly_events[0]
        assert "T" not in str(evw["event_ts"])
        assert "week_has_daily_extension_extreme" in evw


def test_msb_page_helpers_handle_missing_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import msb_page

    missing_path = tmp_path / "missing.duckdb"
    symbols, note = msb_page.list_msb_symbols(database_path=str(missing_path))
    assert symbols == []
    assert note is not None

    payload, payload_note = msb_page.load_msb_payload(
        symbol="SPY",
        lookback_days=365,
        tail_low_pct=5.0,
        tail_high_pct=95.0,
        database_path=str(missing_path),
    )
    assert payload_note is None
    assert payload is not None
    assert payload["daily_events"] == []
    assert payload["weekly_events"] == []
    assert payload["notes"]
