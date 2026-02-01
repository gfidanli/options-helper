from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def test_extension_stats_cli_writes_schema_v5_and_max_move_section(tmp_path: Path) -> None:
    # Synthetic OHLC: gentle uptrend with tight ranges to create extended readings.
    idx = pd.date_range("2024-01-01", periods=160, freq="B")
    close = pd.Series([100.0 + i * 0.15 for i in range(len(idx))], index=idx, dtype="float64")
    ohlc = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        },
        index=idx,
    )

    ohlc_path = tmp_path / "ohlc.csv"
    ohlc.to_csv(ohlc_path)

    # Lower warmup + tail thresholds to ensure we get tail events deterministically.
    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = (
        cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
        .replace("tail_high_pct: 97.5", "tail_high_pct: 80.0")
        .replace("tail_low_pct: 2.5", "tail_low_pct: 20.0")
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_mod, encoding="utf-8")

    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "technicals",
            "extension-stats",
            "--ohlc-path",
            str(ohlc_path),
            "--config",
            str(cfg_path),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    json_paths = list((out_dir / "UNKNOWN").glob("*.json"))
    md_paths = list((out_dir / "UNKNOWN").glob("*.md"))
    assert json_paths, "expected extension-stats JSON artifact"
    assert md_paths, "expected extension-stats Markdown artifact"

    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 5
    assert 15 in payload["config"]["extension_percentiles"]["forward_days_daily"]
    assert "max_upside_summary_daily" in payload
    assert "max_move_summary_daily" in payload
    assert payload["max_upside_summary_daily"]["buckets"], "expected non-empty max-upside buckets"
    assert "9m" in payload["config"]["max_forward_returns"]["horizons_days"]

    # Ensure the summary carries both favorable and drawdown stats per horizon.
    b0 = payload["max_move_summary_daily"]["buckets"][0]
    s1w = b0["stats"]["1w"]
    assert "fav_median" in s1w
    assert "fav_p75" in s1w
    assert "dd_median" in s1w
    assert "dd_p75" in s1w

    md = md_paths[0].read_text(encoding="utf-8")
    assert "## Max Favorable Move (Daily)" in md
    assert "Cells: fav (med/p75); dd (med/p75)." in md
