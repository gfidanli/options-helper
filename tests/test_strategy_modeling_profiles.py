from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pytest

from options_helper.data.strategy_modeling_profiles import (
    StrategyModelingProfileStoreError,
    list_strategy_modeling_profiles,
    load_strategy_modeling_profile,
    save_strategy_modeling_profile,
)
from options_helper.schemas.strategy_modeling_profile import StrategyModelingProfile


def _sample_profile() -> StrategyModelingProfile:
    return StrategyModelingProfile.model_validate(
        {
            "strategy": "orb",
            "symbols": ["SPY", "QQQ"],
            "start_date": date(2026, 1, 1),
            "end_date": date(2026, 1, 31),
            "intraday_timeframe": "5Min",
            "intraday_source": "stocks_bars_local",
            "starting_capital": 25_000.0,
            "risk_per_trade_pct": 2.5,
            "gap_fill_policy": "fill_at_open",
            "max_hold_bars": 20,
            "one_open_per_symbol": True,
            "r_ladder_min_tenths": 12,
            "r_ladder_max_tenths": 20,
            "r_ladder_step_tenths": 2,
            "allow_shorts": False,
            "enable_orb_confirmation": True,
            "orb_range_minutes": 20,
            "orb_confirmation_cutoff_et": "10:15",
            "orb_stop_policy": "tighten",
            "enable_atr_stop_floor": True,
            "atr_stop_floor_multiple": 0.8,
            "enable_rsi_extremes": True,
            "enable_ema9_regime": True,
            "ema9_slope_lookback_bars": 5,
            "enable_volatility_regime": True,
            "allowed_volatility_regimes": ["low", "normal"],
            "ma_fast_window": 21,
            "ma_slow_window": 55,
            "ma_trend_window": 200,
            "ma_fast_type": "ema",
            "ma_slow_type": "sma",
            "ma_trend_type": "sma",
            "trend_slope_lookback_bars": 4,
            "atr_window": 10,
            "atr_stop_multiple": 1.7,
        }
    )


def test_strategy_modeling_profiles_missing_store_lists_empty(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    assert list_strategy_modeling_profiles(path) == []


def test_strategy_modeling_profiles_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    profile = _sample_profile()

    save_strategy_modeling_profile(path, "swing_orb", profile, overwrite=False)
    names = list_strategy_modeling_profiles(path)
    loaded = load_strategy_modeling_profile(path, "swing_orb")

    assert names == ["swing_orb"]
    assert loaded == profile
    assert loaded.max_hold_timeframe == "entry"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "updated_at" in payload
    assert "swing_orb" in payload["profiles"]


def test_strategy_modeling_profiles_overwrite_requires_explicit_flag(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    profile = _sample_profile()
    save_strategy_modeling_profile(path, "orb_core", profile, overwrite=False)

    with pytest.raises(StrategyModelingProfileStoreError, match="already exists"):
        save_strategy_modeling_profile(path, "orb_core", profile, overwrite=False)

    updated = profile.model_copy(update={"starting_capital": 30_000.0})
    save_strategy_modeling_profile(path, "orb_core", updated, overwrite=True)
    loaded = load_strategy_modeling_profile(path, "orb_core")
    assert loaded.starting_capital == 30_000.0


def test_strategy_modeling_profiles_malformed_json_raises_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(StrategyModelingProfileStoreError, match="not valid JSON"):
        list_strategy_modeling_profiles(path)


def test_strategy_modeling_profiles_reject_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 99,
                "updated_at": "2026-02-10T00:00:00+00:00",
                "profiles": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StrategyModelingProfileStoreError, match="Unsupported profile store schema_version"):
        list_strategy_modeling_profiles(path)
