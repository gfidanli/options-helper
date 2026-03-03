from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from options_helper.data.technical_backtesting_config import ConfigError, load_technical_backtesting_config


_SCHEMA_PATH = Path("config/technical_backtesting.schema.json")


def _load_from_tmp_config(tmp_path: Path, cfg: dict) -> dict:
    config_path = tmp_path / "technical_backtesting.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return load_technical_backtesting_config(config_path=config_path, schema_path=_SCHEMA_PATH)


def test_load_defaults_include_cvd_strategy_interval_templates_and_bar_fields() -> None:
    cfg = load_technical_backtesting_config()

    cvd_cfg = cfg["strategies"]["CvdDivergenceMSB"]
    assert cvd_cfg["enabled"] is False
    ibs_cfg = cfg["strategies"]["MeanReversionIBS"]
    assert ibs_cfg["enabled"] is True
    assert "cost_overrides" not in ibs_cfg

    walk_cfg = cfg["walk_forward"]
    assert walk_cfg["min_history_bars"] == 0
    assert walk_cfg["train_bars"] == 0
    assert walk_cfg["validate_bars"] == 0
    assert walk_cfg["step_bars"] == 0

    artifacts_cfg = cfg["artifacts"]
    assert "{interval}" in artifacts_cfg["params_path_template"]
    assert "{interval}" in artifacts_cfg["report_path_template"]
    assert "{interval}" in artifacts_cfg["heatmap_path_template"]


def test_load_accepts_legacy_config_without_new_fields(tmp_path: Path) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["strategies"].pop("CvdDivergenceMSB", None)
    cfg["strategies"].pop("MeanReversionIBS", None)
    for strategy_cfg in cfg["strategies"].values():
        if isinstance(strategy_cfg, dict):
            strategy_cfg.pop("cost_overrides", None)
    for key in ("min_history_bars", "train_bars", "validate_bars", "step_bars"):
        cfg["walk_forward"].pop(key, None)

    loaded = _load_from_tmp_config(tmp_path, cfg)
    assert "CvdDivergenceMSB" not in loaded["strategies"]
    assert "MeanReversionIBS" not in loaded["strategies"]
    assert "train_years" in loaded["walk_forward"]


def test_load_accepts_mean_reversion_ibs_with_optional_cost_overrides(tmp_path: Path) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["strategies"]["MeanReversionIBS"]["cost_overrides"] = {"commission": 0.001}

    loaded = _load_from_tmp_config(tmp_path, cfg)
    assert loaded["strategies"]["MeanReversionIBS"]["cost_overrides"] == {"commission": 0.001}


@pytest.mark.parametrize(
    ("cost_overrides", "match"),
    [
        ({"commission": -0.001}, r"cost_overrides\.commission"),
        ({"fee_bps": 2.0}, r"cost_overrides"),
        ("invalid", r"cost_overrides"),
    ],
)
def test_load_rejects_invalid_strategy_cost_overrides(
    tmp_path: Path,
    cost_overrides: dict[str, float] | str,
    match: str,
) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["strategies"]["MeanReversionIBS"]["cost_overrides"] = cost_overrides

    with pytest.raises(ConfigError, match=match):
        _load_from_tmp_config(tmp_path, cfg)


def test_load_rejects_bar_mode_without_validate_bars(tmp_path: Path) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["walk_forward"]["train_bars"] = 250
    cfg["walk_forward"]["validate_bars"] = 0

    with pytest.raises(ConfigError, match="walk_forward\\.validate_bars must be > 0 when train_bars > 0"):
        _load_from_tmp_config(tmp_path, cfg)


def test_load_rejects_bar_mode_without_positive_step_bars(tmp_path: Path) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["walk_forward"]["train_bars"] = 250
    cfg["walk_forward"]["validate_bars"] = 60
    cfg["walk_forward"]["step_bars"] = 0

    with pytest.raises(ConfigError, match="walk_forward\\.step_bars must be > 0 when train_bars > 0"):
        _load_from_tmp_config(tmp_path, cfg)


def test_load_rejects_min_history_bars_below_train_plus_validate(tmp_path: Path) -> None:
    cfg = deepcopy(load_technical_backtesting_config())
    cfg["walk_forward"]["train_bars"] = 250
    cfg["walk_forward"]["validate_bars"] = 60
    cfg["walk_forward"]["step_bars"] = 30
    cfg["walk_forward"]["min_history_bars"] = 200

    with pytest.raises(
        ConfigError,
        match="walk_forward\\.min_history_bars must be >= train_bars \\+ validate_bars when train_bars > 0",
    ):
        _load_from_tmp_config(tmp_path, cfg)
