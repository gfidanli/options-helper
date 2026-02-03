from __future__ import annotations

from options_helper.data.confluence_config import load_confluence_config


def test_load_confluence_config_defaults() -> None:
    cfg = load_confluence_config()
    assert cfg["schema_version"] == 1
    assert cfg["weights"]["weekly_trend"] > 0
    assert cfg["extension"]["tail_low"] < cfg["extension"]["tail_high"]
