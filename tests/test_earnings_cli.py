from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app


def test_earnings_cli_manual_set_writes_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "earnings"
    runner = CliRunner()

    res = runner.invoke(app, ["earnings", "AAA", "--set", "2026-02-06", "--cache-dir", str(cache_dir)])
    assert res.exit_code == 0, res.output
    assert "Saved:" in res.output

    cache_path = cache_dir / "AAA.json"
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "AAA"
    assert payload["next_earnings_date"] == "2026-02-06"
    assert payload["source"] == "manual"

    res2 = runner.invoke(app, ["earnings", "AAA", "--cache-dir", str(cache_dir)])
    assert res2.exit_code == 0, res2.output
    assert "2026-02-06" in res2.output


def test_earnings_cli_refresh_uses_client(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "earnings"

    class StubProvider:
        def get_next_earnings_event(self, symbol: str):
            class Ev:
                source = "stub"
                next_date = date(2026, 2, 6)
                window_start = None
                window_end = None
                raw = {"stub": True}

            return Ev()

    monkeypatch.setattr("options_helper.cli.get_provider", lambda *_args, **_kwargs: StubProvider())

    runner = CliRunner()
    res = runner.invoke(app, ["earnings", "AAA", "--refresh", "--cache-dir", str(cache_dir)])
    assert res.exit_code == 0, res.output
    assert "2026-02-06" in res.output

    payload = json.loads((cache_dir / "AAA.json").read_text(encoding="utf-8"))
    assert payload["source"] == "stub"


def test_refresh_earnings_uses_watchlists(tmp_path: Path, monkeypatch) -> None:
    watchlists_path = tmp_path / "watchlists.json"
    watchlists_path.write_text(
        json.dumps({"watchlists": {"wl1": ["AAA", "BBB"], "wl2": ["CCC"]}}, indent=2),
        encoding="utf-8",
    )

    cache_dir = tmp_path / "earnings"

    class StubProvider:
        def get_next_earnings_event(self, symbol: str):
            class Ev:
                source = "stub"
                next_date = date(2026, 2, 6)
                window_start = None
                window_end = None
                raw = None

            return Ev()

    monkeypatch.setattr("options_helper.cli.get_provider", lambda *_args, **_kwargs: StubProvider())

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["refresh-earnings", "--watchlists-path", str(watchlists_path), "--cache-dir", str(cache_dir)],
    )
    assert res.exit_code == 0, res.output
    assert (cache_dir / "AAA.json").exists()
    assert (cache_dir / "BBB.json").exists()
    assert (cache_dir / "CCC.json").exists()

    res2 = runner.invoke(
        app,
        [
            "refresh-earnings",
            "--watchlists-path",
            str(watchlists_path),
            "--watchlist",
            "wl2",
            "--cache-dir",
            str(cache_dir),
        ],
    )
    assert res2.exit_code == 0, res2.output
    assert "CCC" in res2.output
