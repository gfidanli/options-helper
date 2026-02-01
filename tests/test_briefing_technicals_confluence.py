from __future__ import annotations

import json

from options_helper.data.technical_backtesting_config import load_technical_backtesting_config
from options_helper.reporting_briefing import (
    BriefingSymbolSection,
    build_briefing_payload,
    render_briefing_markdown,
)
from options_helper.technicals_backtesting.snapshot import compute_technical_snapshot
from tests.technical_backtesting_helpers import make_synthetic_ohlc


def test_compute_technical_snapshot_smoke() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=320, seed=123)

    snap = compute_technical_snapshot(df, cfg)
    assert snap is not None
    assert snap.atr_window == int(cfg["indicators"]["atr"]["window_default"])
    assert snap.z_window == int(cfg["indicators"]["zscore"]["window_default"])
    assert snap.bb_window == int(cfg["indicators"]["bollinger"]["window_default"])
    assert snap.asof == df.index.max().date().isoformat()
    assert snap.atr is not None


def test_briefing_renders_technicals_without_chain() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=320, seed=7)
    snap = compute_technical_snapshot(df, cfg)
    assert snap is not None

    sec = BriefingSymbolSection(
        symbol="TEST",
        as_of="2024-01-01",
        chain=None,
        compare=None,
        flow_net=None,
        technicals=snap,
        errors=[],
        warnings=[],
        derived_updated=False,
    )
    md = render_briefing_markdown(
        report_date="2024-01-01",
        portfolio_path="portfolio.json",
        symbol_sections=[sec],
        top=3,
    )
    assert "### Technicals (canonical: technicals_backtesting)" in md


def test_briefing_payload_is_json_serializable() -> None:
    cfg = load_technical_backtesting_config()
    df = make_synthetic_ohlc(rows=320, seed=8)
    snap = compute_technical_snapshot(df, cfg)
    assert snap is not None

    sec = BriefingSymbolSection(
        symbol="TEST",
        as_of="2024-01-01",
        chain=None,
        compare=None,
        flow_net=None,
        technicals=snap,
        errors=[],
        warnings=[],
        derived_updated=False,
    )
    payload = build_briefing_payload(
        report_date="2024-01-01",
        portfolio_path="portfolio.json",
        symbol_sections=[sec],
        top=3,
        technicals_config="config/technical_backtesting.yaml",
    )
    s = json.dumps(payload, allow_nan=False)
    assert "technicals_backtesting" in s
