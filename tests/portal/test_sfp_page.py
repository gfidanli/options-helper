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

    # Inject periodic bearish/bullish sweep-and-reclaim patterns.
    for i in range(16, periods, 30):
        high[i] = max(high[i], high[i - 3] + 1.8)
        close[i] = high[i - 3] - 0.3
        open_[i] = close[i] + 0.6
        low[i] = min(low[i], close[i] - 1.0)
    for i in range(28, periods, 30):
        low[i] = min(low[i], low[i - 3] - 1.8)
        close[i] = low[i - 3] + 0.3
        open_[i] = close[i] - 0.6
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


def test_sfp_page_helpers_with_seeded_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import sfp_page

    db_path = tmp_path / "portal.duckdb"
    _seed_candles(db_path, symbol="SPY", periods=220)

    symbols, symbols_note = sfp_page.list_sfp_symbols(database_path=str(db_path))
    assert symbols_note is None
    assert "SPY" in symbols

    candles, candles_note = sfp_page.load_candles_history("SPY", database_path=str(db_path), limit=120)
    assert candles_note is None
    assert not candles.empty
    assert candles["ts"].is_monotonic_increasing

    payload, payload_note = sfp_page.load_sfp_payload(
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
    assert "Daily SFP at RSI Extremes (Bullish)" in groups
    assert "Daily SFP at RSI Extremes (Bearish)" in groups
    assert "Weekly SFP + Daily Extension Extreme in Week (Bullish)" in groups
    assert "Weekly SFP + Daily Extension Extreme in Week (Bearish)" in groups
    for row in summary:
        assert "median_1d_pct" in row
        assert "median_5d_pct" in row
        assert "median_10d_pct" in row

    if daily_events:
        ev = daily_events[0]
        assert "T" not in str(ev["event_ts"])
        assert "extension_percentile" in ev
        assert {"forward_1d_pct", "forward_5d_pct", "forward_10d_pct"}.issubset(ev.keys())
        for row in daily_events:
            direction = str(row.get("direction") or "").lower()
            for key in ("forward_1d_pct", "forward_5d_pct", "forward_10d_pct"):
                value = row.get(key)
                if value is None:
                    continue
                if direction == "bullish":
                    assert float(value) >= 0.0
                if direction == "bearish":
                    assert float(value) <= 0.0

    if weekly_events:
        evw = weekly_events[0]
        assert "T" not in str(evw["event_ts"])
        assert "week_has_daily_extension_extreme" in evw
        for row in weekly_events:
            direction = str(row.get("direction") or "").lower()
            for key in ("forward_1d_pct", "forward_5d_pct", "forward_10d_pct"):
                value = row.get(key)
                if value is None:
                    continue
                if direction == "bullish":
                    assert float(value) >= 0.0
                if direction == "bearish":
                    assert float(value) <= 0.0

    payload_ignore, payload_ignore_note = sfp_page.load_sfp_payload(
        symbol="SPY",
        lookback_days=700,
        tail_low_pct=5.0,
        tail_high_pct=95.0,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        ignore_swept_swings=True,
        database_path=str(db_path),
    )
    assert payload_ignore_note is None
    assert payload_ignore is not None
    assert len(payload_ignore.get("daily_events") or []) <= len(daily_events)


def test_sfp_page_helpers_handle_missing_db(tmp_path: Path) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import sfp_page

    missing_path = tmp_path / "missing.duckdb"
    symbols, note = sfp_page.list_sfp_symbols(database_path=str(missing_path))
    assert symbols == []
    assert note is not None

    payload, payload_note = sfp_page.load_sfp_payload(
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


def test_forward_max_move_uses_next_open_entry_and_directional_sign() -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components.sfp_page import _forward_returns_from_anchor

    daily_open = pd.Series([100.0, 90.0, 95.0, 96.0], dtype="float64")
    daily_high = pd.Series([101.0, 92.0, 105.0, 97.0], dtype="float64")
    daily_low = pd.Series([99.0, 88.0, 94.0, 80.0], dtype="float64")

    # Anchor at index 0 => entry is next open (index 1 = 90.0).
    bullish = _forward_returns_from_anchor(
        daily_open=daily_open,
        daily_high=daily_high,
        daily_low=daily_low,
        direction="bullish",
        anchor_pos=0,
    )
    bearish = _forward_returns_from_anchor(
        daily_open=daily_open,
        daily_high=daily_high,
        daily_low=daily_low,
        direction="bearish",
        anchor_pos=0,
    )

    # 1d bullish uses day-1 high 92 from entry 90 => +2.22%
    assert bullish["forward_1d_pct"] == pytest.approx(2.22, abs=1e-2)
    # 1d bearish uses day-1 low 88 from entry 90 => -2.22%
    assert bearish["forward_1d_pct"] == pytest.approx(-2.22, abs=1e-2)

    # 5d window here truncates to available future bars; bullish max high=105, bearish min low=80.
    assert bullish["forward_5d_pct"] == pytest.approx(16.67, abs=1e-2)
    assert bearish["forward_5d_pct"] == pytest.approx(-11.11, abs=1e-2)


def test_sfp_page_compat_with_legacy_compute_signature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("streamlit")
    from apps.streamlit.components import sfp_page
    from options_helper.analysis.sfp import compute_sfp_signals as current_compute

    db_path = tmp_path / "portal_legacy.duckdb"
    _seed_candles(db_path, symbol="SPY", periods=220)

    def legacy_compute_sfp_signals(
        ohlc: pd.DataFrame,
        *,
        swing_left_bars: int = 1,
        swing_right_bars: int = 1,
        min_swing_distance_bars: int = 1,
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        return current_compute(
            ohlc,
            swing_left_bars=swing_left_bars,
            swing_right_bars=swing_right_bars,
            min_swing_distance_bars=min_swing_distance_bars,
            timeframe=timeframe,
        )

    monkeypatch.setattr(sfp_page, "compute_sfp_signals", legacy_compute_sfp_signals)

    payload_off, note_off = sfp_page.load_sfp_payload(
        symbol="SPY",
        lookback_days=365,
        tail_low_pct=5.0,
        tail_high_pct=95.0,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        ignore_swept_swings=False,
        database_path=str(db_path),
    )
    payload_on, note_on = sfp_page.load_sfp_payload(
        symbol="SPY",
        lookback_days=365,
        tail_low_pct=5.0,
        tail_high_pct=95.0,
        swing_left_bars=1,
        swing_right_bars=1,
        min_swing_distance_bars=1,
        ignore_swept_swings=True,
        database_path=str(db_path),
    )
    assert note_off is None
    assert note_on is None
    assert payload_off is not None
    assert payload_on is not None
    assert len(payload_on.get("daily_events") or []) < len(payload_off.get("daily_events") or [])
