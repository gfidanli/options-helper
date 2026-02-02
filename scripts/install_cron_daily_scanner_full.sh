#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_CMD_RUN="${REPO_DIR}/scripts/cron_daily_scanner_full.sh"
CRON_CMD_CHECK="${REPO_DIR}/scripts/cron_check_scanner_full.sh"

CRON_TZ="America/Chicago"
SCHEDULE_RUN="30 19 * * 1-5"
SCHEDULE_CHECK="0 21,22,23 * * 1-5"

BEGIN_MARK="# options-helper: daily full scanner"
END_MARK="# end options-helper: daily full scanner"

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
  INSTALL=1
fi

if [[ "${INSTALL}" -eq 0 ]]; then
  echo "Add the following block to your crontab (crontab -e), or re-run with --install:"
  echo
  echo "${BEGIN_MARK}"
  echo "CRON_TZ=${CRON_TZ}"
  echo "${SCHEDULE_RUN} ${CRON_CMD_RUN}"
  echo "${SCHEDULE_CHECK} ${CRON_CMD_CHECK}"
  echo "${END_MARK}"
  echo
  echo "Logs: ${REPO_DIR}/data/logs/scanner_full_YYYY-MM-DD.log"
  echo "Status: ${REPO_DIR}/data/logs/scanner_full_status.json"
  exit 0
fi

REPO_DIR="${REPO_DIR}" \
CRON_CMD_RUN="${CRON_CMD_RUN}" \
CRON_CMD_CHECK="${CRON_CMD_CHECK}" \
CRON_TZ="${CRON_TZ}" \
SCHEDULE_RUN="${SCHEDULE_RUN}" \
SCHEDULE_CHECK="${SCHEDULE_CHECK}" \
BEGIN_MARK="${BEGIN_MARK}" \
END_MARK="${END_MARK}" \
python3 - <<'PY'
from __future__ import annotations

import os
import subprocess
import tempfile

repo_dir = os.environ["REPO_DIR"]
cron_cmd_run = os.environ["CRON_CMD_RUN"]
cron_cmd_check = os.environ["CRON_CMD_CHECK"]
cron_tz = os.environ["CRON_TZ"]
schedule_run = os.environ["SCHEDULE_RUN"]
schedule_check = os.environ["SCHEDULE_CHECK"]
begin = os.environ["BEGIN_MARK"]
end = os.environ["END_MARK"]

try:
    existing = subprocess.run(["crontab", "-l"], check=False, capture_output=True, text=True).stdout
except Exception:
    existing = ""

lines = existing.splitlines()
filtered: list[str] = []
skip = False
for line in lines:
    if line.strip() == begin:
        skip = True
        continue
    if line.strip() == end:
        skip = False
        continue
    if not skip:
        filtered.append(line)

filtered += [
    "",
    begin,
    f"CRON_TZ={cron_tz}",
    f"{schedule_run} {cron_cmd_run}",
    f"{schedule_check} {cron_cmd_check}",
    end,
    "",
]
payload = "\n".join(filtered).lstrip("\n")

with tempfile.NamedTemporaryFile("w", delete=False) as f:
    f.write(payload)
    tmp_path = f.name

subprocess.run(["crontab", tmp_path], check=True, timeout=10)
print("Installed cron job:")
print(f"  {schedule_run} {cron_cmd_run}")
print(f"  {schedule_check} {cron_cmd_check}")
print(f"Logs: {repo_dir}/data/logs/scanner_full_YYYY-MM-DD.log")
print(f"Status: {repo_dir}/data/logs/scanner_full_status.json")
PY
