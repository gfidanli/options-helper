from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from options_helper.data.options_snapshots import OptionsSnapshotStore


def _build_chain(expiry_1: date, expiry_2: date) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contractSymbol": "AAA_C_1",
                "optionType": "call",
                "expiry": expiry_1.isoformat(),
                "strike": 100.0,
                "bid": 1.0,
            },
            {
                "contractSymbol": "AAA_P_1",
                "optionType": "put",
                "expiry": expiry_1.isoformat(),
                "strike": 100.0,
                "bid": 1.2,
            },
            {
                "contractSymbol": "AAA_C_2",
                "optionType": "call",
                "expiry": expiry_2.isoformat(),
                "strike": 110.0,
                "bid": 0.8,
            },
            {
                "contractSymbol": "AAA_P_2",
                "optionType": "put",
                "expiry": expiry_2.isoformat(),
                "strike": 110.0,
                "bid": 0.9,
            },
        ]
    )


def test_save_day_snapshot_writes_expected_files(tmp_path: Path) -> None:
    store = OptionsSnapshotStore(tmp_path)
    snapshot_date = date(2026, 2, 1)
    exp_1 = date(2026, 3, 15)
    exp_2 = date(2026, 4, 19)

    chain = _build_chain(exp_1, exp_2)
    raw_by_expiry = {
        exp_1: {"underlying": {"symbol": "AAA"}},
        exp_2: {"underlying": {"symbol": "AAA"}},
    }
    meta = {"spot": 101.0, "full_chain": True}

    day_dir = store.save_day_snapshot(
        "AAA",
        snapshot_date,
        chain=chain,
        expiries=[exp_1, exp_2],
        raw_by_expiry=raw_by_expiry,
        meta=meta,
    )

    assert day_dir.exists()
    for exp in (exp_1, exp_2):
        csv_path = day_dir / f"{exp.isoformat()}.csv"
        raw_path = day_dir / f"{exp.isoformat()}.raw.json"
        assert csv_path.exists()
        assert raw_path.exists()

    meta_path = day_dir / "meta.json"
    assert meta_path.exists()
    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded_meta.get("spot") == 101.0
    assert loaded_meta.get("full_chain") is True


def test_save_day_snapshot_prunes_stale_expiries(tmp_path: Path) -> None:
    store = OptionsSnapshotStore(tmp_path)
    snapshot_date = date(2026, 2, 1)
    exp_1 = date(2026, 3, 15)
    exp_2 = date(2026, 4, 19)

    chain = _build_chain(exp_1, exp_2)
    raw_by_expiry = {
        exp_1: {"underlying": {"symbol": "AAA"}},
        exp_2: {"underlying": {"symbol": "AAA"}},
    }

    day_dir = store.save_day_snapshot(
        "AAA",
        snapshot_date,
        chain=chain,
        expiries=[exp_1, exp_2],
        raw_by_expiry=raw_by_expiry,
        meta={"spot": 101.0},
    )

    assert (day_dir / f"{exp_1.isoformat()}.csv").exists()
    assert (day_dir / f"{exp_2.isoformat()}.csv").exists()
    assert (day_dir / f"{exp_2.isoformat()}.raw.json").exists()

    # Re-save with only one expiry; stale artifacts should be removed.
    chain_one = chain[chain["expiry"] == exp_1.isoformat()].copy()
    store.save_day_snapshot(
        "AAA",
        snapshot_date,
        chain=chain_one,
        expiries=[exp_1],
        raw_by_expiry={exp_1: raw_by_expiry[exp_1]},
        meta={"spot": 102.0},
    )

    assert (day_dir / f"{exp_1.isoformat()}.csv").exists()
    assert not (day_dir / f"{exp_2.isoformat()}.csv").exists()
    assert not (day_dir / f"{exp_2.isoformat()}.raw.json").exists()
