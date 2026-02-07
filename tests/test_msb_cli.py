from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from options_helper.cli import app


def _sample_ohlc() -> pd.DataFrame:
    idx = pd.date_range("2026-01-05", periods=15, freq="B")
    high = [10.0, 11.0, 12.0, 11.0, 10.0, 13.0, 12.0, 11.0, 10.0, 9.0, 10.0, 11.0, 10.0, 9.0, 8.0]
    low = [9.0, 10.0, 11.0, 10.0, 9.0, 11.5, 11.0, 10.0, 9.0, 8.0, 9.0, 10.0, 6.5, 6.0, 5.0]
    close = [9.5, 10.5, 11.5, 10.5, 9.5, 12.3, 11.5, 10.5, 9.5, 8.5, 9.5, 10.5, 6.8, 6.2, 5.5]
    open_ = [9.2, 10.2, 11.2, 10.2, 9.2, 11.8, 11.7, 10.7, 9.7, 8.7, 9.3, 10.2, 7.2, 6.4, 5.8]
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def test_msb_scan_cli_writes_json_and_markdown_artifacts(tmp_path: Path) -> None:
    df = _sample_ohlc()
    ohlc_path = tmp_path / "ohlc.csv"
    df.to_csv(ohlc_path)
    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "msb-scan",
            "--ohlc-path",
            str(ohlc_path),
            "--out",
            str(out_dir),
            "--swing-left-bars",
            "1",
            "--swing-right-bars",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output

    asof = df.index.max().date().isoformat()
    json_path = out_dir / "UNKNOWN" / f"{asof}.json"
    md_path = out_dir / "UNKNOWN" / f"{asof}.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["symbol"] == "UNKNOWN"
    assert payload["counts"]["bullish_msb"] >= 1
    assert payload["counts"]["bearish_msb"] >= 1
    assert payload["events"]
    assert any(ev["direction"] == "bullish" for ev in payload["events"])
    assert any(ev["direction"] == "bearish" for ev in payload["events"])

    ev0 = payload["events"][0]
    assert "T" not in ev0["timestamp"]
    assert "T" not in ev0["broken_swing_timestamp"]
    for key in ("candle_open", "candle_high", "candle_low", "candle_close", "break_level"):
        assert ev0[key] == round(ev0[key], 2)
    assert ev0["forward_returns_pct"].keys() == {"1d", "5d", "10d"}
    assert "extension_percentile" in ev0

    md = md_path.read_text(encoding="utf-8")
    assert "fwd(1/5/10d)=" in md
    assert "ext_pct=" in md


def test_msb_scan_cli_symbol_path_backfills_cache_when_missing(tmp_path: Path, monkeypatch) -> None:
    df = _sample_ohlc()
    out_dir = tmp_path / "reports"
    cache_dir = tmp_path / "candles"
    calls: dict[str, str] = {}

    def fake_get_daily_history(self, symbol: str, period: str = "2y", today=None):  # noqa: ANN001,ARG001
        calls["symbol"] = symbol
        calls["period"] = period
        return df

    monkeypatch.setattr(
        "options_helper.data.candles.CandleStore.get_daily_history",
        fake_get_daily_history,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "msb-scan",
            "--symbol",
            "ABC",
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(out_dir),
            "--swing-left-bars",
            "1",
            "--swing-right-bars",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls.get("symbol") == "ABC"
    assert calls.get("period") == "max"

    asof = df.index.max().date().isoformat()
    json_path = out_dir / "ABC" / f"{asof}.json"
    assert json_path.exists()


def test_msb_scan_cli_weekly_timeframe_labels_asof_to_monday(tmp_path: Path) -> None:
    df = _sample_ohlc()
    ohlc_path = tmp_path / "ohlc.csv"
    df.to_csv(ohlc_path)
    out_dir = tmp_path / "reports"

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "--storage",
            "filesystem",
            "technicals",
            "msb-scan",
            "--ohlc-path",
            str(ohlc_path),
            "--out",
            str(out_dir),
            "--timeframe",
            "W-FRI",
        ],
    )
    assert res.exit_code == 0, res.output

    json_path = out_dir / "UNKNOWN" / "2026-01-19.json"
    assert json_path.exists()
