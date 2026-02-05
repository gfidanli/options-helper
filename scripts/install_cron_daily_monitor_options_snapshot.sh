#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_CMD="${REPO_DIR}/scripts/cron_daily_monitor_options_snapshot.sh"

# Default: weekdays at 16:05 America/Chicago time (after the main snapshot job).
CRON_TZ="America/Chicago"
SCHEDULE="5 16 * * 1-5"

BEGIN_MARK="# options-helper: daily monitor options snapshot"
END_MARK="# end options-helper: daily monitor options snapshot"

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
  INSTALL=1
fi

if [[ "${INSTALL}" -eq 0 ]]; then
  echo "Add the following block to your crontab (crontab -e), or re-run with --install:"
  echo
  echo "${BEGIN_MARK}"
  echo "CRON_TZ=${CRON_TZ}"
  echo "${SCHEDULE} ${CRON_CMD}"
  echo "${END_MARK}"
  echo
  echo "Logs: ${REPO_DIR}/data/logs/monitor_snapshot.log"
  exit 0
fi

REPO_DIR="${REPO_DIR}" \
CRON_CMD="${CRON_CMD}" \
CRON_TZ="${CRON_TZ}" \
SCHEDULE="${SCHEDULE}" \
BEGIN_MARK="${BEGIN_MARK}" \
END_MARK="${END_MARK}" \
python3 - <<'PY'
from __future__ import annotations

import os
import subprocess
import tempfile

repo_dir = os.environ["REPO_DIR"]
cron_cmd = os.environ["CRON_CMD"]
cron_tz = os.environ["CRON_TZ"]
schedule = os.environ["SCHEDULE"]
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

filtered += ["", begin, f"CRON_TZ={cron_tz}", f"{schedule} {cron_cmd}", end, ""]
payload = "\n".join(filtered).lstrip("\n")

with tempfile.NamedTemporaryFile("w", delete=False) as f:
    f.write(payload)
    tmp_path = f.name

subprocess.run(["crontab", tmp_path], check=True, timeout=10)
print("Installed cron job:")
print(f"  CRON_TZ={cron_tz}")
print(f"  {schedule} {cron_cmd}")
print(f"Logs: {repo_dir}/data/logs/monitor_snapshot.log")
PY
