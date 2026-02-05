from __future__ import annotations

import re
from pathlib import Path


def _read_schedule_var(path: Path, var_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(
        rf'^\s*{re.escape(var_name)}\s*=\s*"([^"]+)"\s*$',
        text,
        flags=re.MULTILINE,
    )
    assert m, f"Missing {var_name} in {path}"
    return m.group(1)


def test_automation_md_schedule_table_matches_installers() -> None:
    repo_dir = Path(__file__).resolve().parents[1]
    docs_path = repo_dir / "docs" / "AUTOMATION.md"
    docs = docs_path.read_text(encoding="utf-8")

    checks: list[tuple[str, str]] = []

    checks.append(
        (
            "weekly_refresh_earnings",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_weekly_refresh_earnings.sh", "SCHEDULE"),
        )
    )
    checks.append(
        (
            "daily_options_snapshot",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_daily_options_snapshot.sh", "SCHEDULE"),
        )
    )
    checks.append(
        (
            "daily_monitor_options_snapshot",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_daily_monitor_options_snapshot.sh", "SCHEDULE"),
        )
    )
    checks.append(
        (
            "daily_briefing",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_daily_briefing.sh", "SCHEDULE"),
        )
    )
    checks.append(
        (
            "offline_report_pack",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_offline_report_pack.sh", "SCHEDULE"),
        )
    )
    checks.append(
        (
            "intraday_capture",
            _read_schedule_var(repo_dir / "scripts" / "install_cron_intraday_capture.sh", "SCHEDULE"),
        )
    )

    scanner_install = repo_dir / "scripts" / "install_cron_daily_scanner_full.sh"
    checks.append(("daily_scanner_full_run", _read_schedule_var(scanner_install, "SCHEDULE_RUN")))
    checks.append(("check_scanner_full", _read_schedule_var(scanner_install, "SCHEDULE_CHECK")))

    missing = []
    for name, cron in checks:
        if f"`{cron}`" not in docs:
            missing.append((name, cron))

    assert not missing, (
        "AUTOMATION.md is missing one or more installer cron schedules in the schedule table:\n"
        + "\n".join(f"- {name}: `{cron}`" for name, cron in missing)
    )
