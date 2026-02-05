from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from options_helper.cli import app
from options_helper.data.alpaca_rate_limits import parse_alpaca_rate_limit_headers


def test_parse_alpaca_rate_limit_headers_epoch_seconds() -> None:
    snap = parse_alpaca_rate_limit_headers(
        {
            "X-RateLimit-Limit": "200",
            "X-RateLimit-Remaining": "123",
            "X-RateLimit-Reset": "1700000000",
        }
    )
    assert snap is not None
    assert snap.limit == 200
    assert snap.remaining == 123
    assert snap.reset_epoch == 1700000000
    assert snap.reset_at is not None


def test_parse_alpaca_rate_limit_headers_rfc_date() -> None:
    snap = parse_alpaca_rate_limit_headers(
        {
            "x-ratelimit-reset": "Wed, 21 Oct 2015 07:28:00 GMT",
        }
    )
    assert snap is not None
    assert snap.reset_epoch == 1445412480
    assert snap.reset_at is not None
    assert snap.reset_at.isoformat().startswith("2015-10-21T07:28:00")


def test_debug_rate_limits_uses_latest_log(tmp_path: Path) -> None:
    runner = CliRunner()
    inspect_dir = tmp_path / "inspect_logs"
    run_dir = tmp_path / "run_logs"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    older = inspect_dir / "older.log"
    newer = inspect_dir / "newer.log"

    older.write_text(
        "2026-02-05 00:00:00,000 INFO options_helper.cli: ALPACA_RATELIMIT client=stock method=GET path=/v2/stocks/bars status=200 limit=200 remaining=199 reset_epoch=1700000000 reset_in_s=10.000\n",
        encoding="utf-8",
    )
    newer.write_text(
        "2026-02-05 00:00:01,000 INFO options_helper.cli: ALPACA_RATELIMIT client=option method=GET path=/v2/options/bars status=200 limit=200 remaining=2 reset_epoch=1700000000 reset_in_s=1.234\n",
        encoding="utf-8",
    )

    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    res = runner.invoke(
        app,
        [
            "--log-dir",
            str(run_dir),
            "debug",
            "rate-limits",
            "--log-dir",
            str(inspect_dir),
            "--tail",
            "1",
        ],
    )

    assert res.exit_code == 0, res.output
    assert "newer.log" in res.output
    assert "remaining=2" in res.output

