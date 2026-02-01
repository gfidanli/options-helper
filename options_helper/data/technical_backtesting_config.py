from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator


class ConfigError(ValueError):
    pass


def load_technical_backtesting_config(
    config_path: Path | str = Path("config/technical_backtesting.yaml"),
    schema_path: Path | str = Path("config/technical_backtesting.schema.json"),
) -> dict:
    config_path = Path(config_path)
    schema_path = Path(schema_path)

    if not config_path.exists():
        raise ConfigError(f"Missing config file: {config_path}")
    if not schema_path.exists():
        raise ConfigError(f"Missing schema file: {schema_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(cfg), key=lambda e: e.path)
    if errors:
        messages = []
        for err in errors[:10]:
            loc = ".".join(str(p) for p in err.path) or "<root>"
            messages.append(f"{loc}: {err.message}")
        raise ConfigError("Config schema validation failed: " + "; ".join(messages))

    _light_validate(cfg)
    return cfg


def _light_validate(cfg: dict) -> None:
    required_sections = [
        "schema_version",
        "timezone",
        "data",
        "indicators",
        "weekly_regime",
        "backtest",
        "optimization",
        "walk_forward",
        "calibration",
        "strategies",
        "artifacts",
        "logging",
        "extension_percentiles",
    ]
    missing = [k for k in required_sections if k not in cfg]
    if missing:
        raise ConfigError(f"Missing required config sections: {missing}")

    for strat_name in ("TrendPullbackATR", "MeanReversionBollinger"):
        strat = cfg["strategies"].get(strat_name)
        if not strat:
            raise ConfigError(f"Missing strategy config: {strat_name}")
        for key in ("enabled", "defaults", "search_space", "constraints"):
            if key not in strat:
                raise ConfigError(f"Strategy {strat_name} missing '{key}'")

    provider = cfg["indicators"].get("provider")
    if provider not in {"ta", "talib"}:
        raise ConfigError("indicators.provider must be 'ta' or 'talib'")

    price_adj = (cfg.get("data", {}).get("candles", {}) or {}).get("price_adjustment", {}) or {}
    if bool(price_adj.get("auto_adjust")) and bool(price_adj.get("back_adjust")):
        raise ConfigError("data.candles.price_adjustment: auto_adjust and back_adjust cannot both be true")

    ext_cfg = cfg.get("extension_percentiles", {})
    if ext_cfg:
        high = float(ext_cfg.get("tail_high_pct", 95))
        low = float(ext_cfg.get("tail_low_pct", 5))
        if low >= high:
            raise ConfigError("extension_percentiles.tail_low_pct must be < tail_high_pct")
        days_per_year = int(ext_cfg.get("days_per_year", 252))
        if days_per_year <= 0:
            raise ConfigError("extension_percentiles.days_per_year must be > 0")
        windows_years = ext_cfg.get("windows_years", [])
        if not windows_years:
            raise ConfigError("extension_percentiles.windows_years must be non-empty")
        forward_days = ext_cfg.get("forward_days", [])
        if not forward_days:
            raise ConfigError("extension_percentiles.forward_days must be non-empty")
