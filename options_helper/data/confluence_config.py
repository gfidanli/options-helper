from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


class ConfigError(ValueError):
    pass


def load_confluence_config(
    config_path: Path | str = Path("config/confluence.yaml"),
    schema_path: Path | str = Path("config/confluence.schema.json"),
) -> dict:
    config_path = Path(config_path)
    schema_path = Path(schema_path)

    if not config_path.exists():
        raise ConfigError(f"Missing config file: {config_path}")
    if not schema_path.exists():
        raise ConfigError(f"Missing schema file: {schema_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ConfigError("Confluence config is empty or invalid.")

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        messages = []
        for err in errors[:10]:
            loc = ".".join(str(p) for p in err.path) or "<root>"
            messages.append(f"{loc}: {err.message}")
        raise ConfigError("Confluence config schema validation failed: " + "; ".join(messages))

    _light_validate(cfg)
    return cfg


def _light_validate(cfg: dict) -> None:
    required_sections = ["schema_version", "weights", "extension", "flow", "rsi_divergence", "iv_regime"]
    missing = [k for k in required_sections if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    weights = cfg.get("weights", {})
    if any(float(weights.get(k, 0.0)) < 0 for k in weights.keys()):
        raise ConfigError("weights must be >= 0")

    ext_cfg = cfg.get("extension", {}) or {}
    tail_low = float(ext_cfg.get("tail_low", 0.0))
    tail_high = float(ext_cfg.get("tail_high", 100.0))
    if tail_low >= tail_high:
        raise ConfigError("extension.tail_low must be < tail_high")

    flow_cfg = cfg.get("flow", {}) or {}
    min_abs = float(flow_cfg.get("min_abs_notional", 0.0))
    if min_abs < 0:
        raise ConfigError("flow.min_abs_notional must be >= 0")

    rsi_cfg = cfg.get("rsi_divergence", {}) or {}
    favor_factor = float(rsi_cfg.get("favor_factor", 0.5))
    if not (0.0 <= favor_factor <= 1.0):
        raise ConfigError("rsi_divergence.favor_factor must be between 0 and 1")

    iv_cfg = cfg.get("iv_regime", {}) or {}
    low = float(iv_cfg.get("low", 0.0))
    high = float(iv_cfg.get("high", 0.0))
    if low >= high:
        raise ConfigError("iv_regime.low must be < iv_regime.high")

    scanner_cfg = cfg.get("scanner_rank") or {}
    if scanner_cfg:
        rank_weights = scanner_cfg.get("weights", {})
        if any(float(rank_weights.get(k, 0.0)) < 0 for k in rank_weights.keys()):
            raise ConfigError("scanner_rank.weights must be >= 0")

        rank_ext = scanner_cfg.get("extension", {}) or {}
        rank_low = float(rank_ext.get("tail_low", 0.0))
        rank_high = float(rank_ext.get("tail_high", 100.0))
        if rank_low >= rank_high:
            raise ConfigError("scanner_rank.extension.tail_low must be < tail_high")

        rank_flow = scanner_cfg.get("flow", {}) or {}
        rank_min_abs = float(rank_flow.get("min_abs_notional", 0.0))
        if rank_min_abs < 0:
            raise ConfigError("scanner_rank.flow.min_abs_notional must be >= 0")

        rank_rsi = scanner_cfg.get("rsi_divergence", {}) or {}
        rank_favor = float(rank_rsi.get("favor_factor", 0.5))
        if not (0.0 <= rank_favor <= 1.0):
            raise ConfigError("scanner_rank.rsi_divergence.favor_factor must be between 0 and 1")

        rank_iv = scanner_cfg.get("iv_regime", {}) or {}
        rank_iv_low = float(rank_iv.get("low", 0.0))
        rank_iv_high = float(rank_iv.get("high", 0.0))
        if rank_iv_low >= rank_iv_high:
            raise ConfigError("scanner_rank.iv_regime.low must be < iv_regime.high")

        top_reasons = int(scanner_cfg.get("top_reasons", 0))
        if top_reasons < 0:
            raise ConfigError("scanner_rank.top_reasons must be >= 0")
