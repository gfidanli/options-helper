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


def test_extension_stats_cli_symbol_backfills_missing_cache(tmp_path: Path, monkeypatch) -> None:
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

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg_mod, encoding="utf-8")

    out_dir = tmp_path / "reports"
    cache_dir = tmp_path / "candles"

    calls: dict[str, str] = {}

    def fake_get_daily_history(self, symbol: str, period: str = "2y", today=None):  # noqa: ANN001
        calls["symbol"] = symbol
        calls["period"] = period
        return ohlc

    monkeypatch.setattr(
        "options_helper.data.candles.CandleStore.get_daily_history",
        fake_get_daily_history,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "technicals",
            "extension-stats",
            "--symbol",
            "URG",
            "--cache-dir",
            str(cache_dir),
            "--config",
            str(cfg_path),
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output
    assert calls.get("symbol") == "URG"
    assert calls.get("period") == "max"

    json_paths = list((out_dir / "URG").glob("*.json"))
    assert json_paths, "expected extension-stats JSON artifact"


def test_extension_stats_cli_tail_pct_overrides_tail_thresholds(tmp_path: Path) -> None:
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

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
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
            "--tail-pct",
            "5",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    json_paths = list((out_dir / "UNKNOWN").glob("*.json"))
    assert json_paths, "expected extension-stats JSON artifact"
    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))

    ext_cfg = payload["config"]["extension_percentiles"]
    assert ext_cfg["tail_low_pct"] == 5.0
    assert ext_cfg["tail_high_pct"] == 95.0

    rsi_cfg = payload["config"]["rsi_divergence"]
    assert rsi_cfg["min_extension_percentile"] == 95.0
    assert rsi_cfg["max_extension_percentile"] == 5.0


def test_extension_stats_cli_auto_window_years_short_history_uses_1y(tmp_path: Path) -> None:
    idx = pd.date_range("2022-01-03", periods=900, freq="B")
    close = pd.Series([50.0 + i * 0.05 for i in range(len(idx))], index=idx, dtype="float64")
    ohlc = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        },
        index=idx,
    )

    ohlc_path = tmp_path / "ohlc_short.csv"
    ohlc.to_csv(ohlc_path)

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
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
    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["config"]["extension_percentiles"]["windows_years"] == [1]


def test_extension_stats_cli_short_history_uses_last_date_filename(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-02", periods=7, freq="B")
    close = pd.Series([50.0 + i * 0.1 for i in range(len(idx))], index=idx, dtype="float64")
    ohlc = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        },
        index=idx,
    )

    ohlc_path = tmp_path / "ohlc_short.csv"
    ohlc.to_csv(ohlc_path)

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 0")
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

    last_date = idx.max().date().isoformat()
    json_path = out_dir / "UNKNOWN" / f"{last_date}.json"
    md_path = out_dir / "UNKNOWN" / f"{last_date}.md"
    assert json_path.exists()
    assert md_path.exists()
    assert not (out_dir / "UNKNOWN" / "-.json").exists()


def test_extension_stats_cli_auto_window_years_long_history_uses_3y(tmp_path: Path) -> None:
    idx = pd.date_range("2018-01-02", periods=1600, freq="B")
    close = pd.Series([100.0 + i * 0.10 for i in range(len(idx))], index=idx, dtype="float64")
    ohlc = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        },
        index=idx,
    )

    ohlc_path = tmp_path / "ohlc_long.csv"
    ohlc.to_csv(ohlc_path)

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
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
    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["config"]["extension_percentiles"]["windows_years"] == [3]


def test_extension_stats_cli_percentile_window_years_override(tmp_path: Path) -> None:
    idx = pd.date_range("2020-01-02", periods=900, freq="B")
    close = pd.Series([100.0 + i * 0.05 for i in range(len(idx))], index=idx, dtype="float64")
    ohlc = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
        },
        index=idx,
    )

    ohlc_path = tmp_path / "ohlc_override.csv"
    ohlc.to_csv(ohlc_path)

    cfg_src = Path("config/technical_backtesting.yaml").read_text(encoding="utf-8")
    cfg_mod = cfg_src.replace("warmup_bars: 200", "warmup_bars: 20")
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
            "--percentile-window-years",
            "2",
            "--out",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0, res.output

    json_paths = list((out_dir / "UNKNOWN").glob("*.json"))
    payload = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert payload["config"]["extension_percentiles"]["windows_years"] == [2]
