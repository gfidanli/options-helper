from __future__ import annotations

import json
from pathlib import Path

from options_helper.data.ingestion.tuning import (
    build_endpoint_stats,
    load_tuning_profile,
    recommend_profile,
    save_tuning_profile,
)


def test_tuning_profile_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "ingest_tuning.json"

    profile = {
        "candles": {"max_rps": 10.0, "concurrency": 2},
        "contracts": {"max_rps": 3.0, "page_size": 5000},
        "bars": {"max_rps": 50.0, "concurrency": 16, "batch_mode": "adaptive", "batch_size": 12},
    }
    save_tuning_profile(path, provider="alpaca", profile=profile)

    loaded = load_tuning_profile(path, provider="alpaca")
    assert loaded["candles"]["max_rps"] == 10.0
    assert loaded["contracts"]["page_size"] == 5000
    assert loaded["bars"]["batch_size"] == 12


def test_tuning_profile_fallback_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "ingest_tuning.json"
    path.write_text("{not json", encoding="utf-8")

    loaded = load_tuning_profile(path, provider="alpaca")
    assert loaded["contracts"]["page_size"] == 10000
    assert loaded["bars"]["batch_mode"] == "adaptive"


def test_recommend_profile_adjusts_from_endpoint_stats() -> None:
    profile = {
        "candles": {"max_rps": 10.0, "concurrency": 2},
        "contracts": {"max_rps": 2.5, "page_size": 10000},
        "bars": {"max_rps": 30.0, "concurrency": 8, "batch_mode": "adaptive", "batch_size": 8},
    }

    contracts_stats = build_endpoint_stats(
        calls=10,
        rate_limit_429=1,
        error_count=2,
        latencies_ms=[100.0, 120.0],
    )
    bars_stats = build_endpoint_stats(
        calls=20,
        rate_limit_429=0,
        error_count=0,
        latencies_ms=[20.0, 30.0, 40.0],
    )

    updated = recommend_profile(profile, contracts_stats=contracts_stats, bars_stats=bars_stats)

    assert updated["contracts"]["max_rps"] < profile["contracts"]["max_rps"]
    assert updated["bars"]["max_rps"] > profile["bars"]["max_rps"]


def test_tuning_profile_file_shape(tmp_path: Path) -> None:
    path = tmp_path / "ingest_tuning.json"
    save_tuning_profile(
        path,
        provider="alpaca",
        profile={
            "candles": {"max_rps": 8.0, "concurrency": 1},
            "contracts": {"max_rps": 2.5, "page_size": 10000},
            "bars": {"max_rps": 30.0, "concurrency": 8, "batch_mode": "adaptive", "batch_size": 8},
        },
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "updated_at" in payload
    assert "profiles" in payload
    assert "alpaca" in payload["profiles"]
