from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from options_helper.data.technical_backtesting_artifacts import (
    build_artifact_paths,
    write_artifacts,
)


def _base_cfg(tmp_path: Path) -> dict:
    return {
        "data": {"candles": {"price_adjustment": "auto", "interval": "1d"}},
        "artifacts": {
            "base_dir": str(tmp_path / "artifacts"),
            "params_path_template": "{ticker}/{strategy}/params.json",
            "report_path_template": "{ticker}/{strategy}/summary.md",
            "write_heatmap": False,
            "overwrite": True,
        },
    }


def test_build_artifact_paths_supports_optional_interval_placeholder(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["artifacts"]["params_path_template"] = "{ticker}/{strategy}/{interval}/params.json"
    cfg["artifacts"]["report_path_template"] = "{ticker}/{strategy}/{interval}/summary.md"

    paths = build_artifact_paths(cfg, ticker="spy", strategy="cvd", interval="15m")

    assert paths.params_path == tmp_path / "artifacts" / "SPY" / "cvd" / "15m" / "params.json"
    assert paths.report_path == tmp_path / "artifacts" / "SPY" / "cvd" / "15m" / "summary.md"
    assert paths.heatmap_path is None


def test_build_artifact_paths_remains_backward_compatible_without_interval_token(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)

    paths = build_artifact_paths(cfg, ticker="spy", strategy="cvd")

    assert paths.params_path == tmp_path / "artifacts" / "SPY" / "cvd" / "params.json"
    assert paths.report_path == tmp_path / "artifacts" / "SPY" / "cvd" / "summary.md"
    assert paths.heatmap_path is None


def test_write_artifacts_includes_interval_in_data_payload(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    data_meta = {
        "start": pd.Timestamp("2026-01-01T00:00:00Z"),
        "end": pd.Timestamp("2026-01-31T00:00:00Z"),
        "bars": 42,
        "warmup_bars": 5,
    }

    paths = write_artifacts(
        cfg,
        ticker="spy",
        strategy="cvd",
        params={"lookback": 20},
        train_stats=None,
        walk_forward_result=None,
        optimize_meta={"method": "grid", "maximize": "sharpe", "constraints": []},
        data_meta=data_meta,
        heatmap=None,
    )

    payload = json.loads(paths.params_path.read_text(encoding="utf-8"))
    assert payload["data"]["interval"] == "1d"
    assert payload["data"]["bars"] == 42
    assert payload["data"]["warmup_bars"] == 5


def test_write_artifacts_interval_argument_overrides_data_meta_for_paths_and_payload(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    cfg["artifacts"]["params_path_template"] = "{ticker}/{strategy}/{interval}/params.json"
    cfg["artifacts"]["report_path_template"] = "{ticker}/{strategy}/{interval}/summary.md"

    paths = write_artifacts(
        cfg,
        ticker="spy",
        strategy="cvd",
        interval="30m",
        params={"lookback": 20},
        train_stats=None,
        walk_forward_result=None,
        optimize_meta={"method": "grid", "maximize": "sharpe", "constraints": []},
        data_meta={"interval": "1h", "bars": 10},
        heatmap=None,
    )

    payload = json.loads(paths.params_path.read_text(encoding="utf-8"))
    assert paths.params_path == tmp_path / "artifacts" / "SPY" / "cvd" / "30m" / "params.json"
    assert paths.report_path == tmp_path / "artifacts" / "SPY" / "cvd" / "30m" / "summary.md"
    assert payload["data"]["interval"] == "30m"
