#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_CMD="${REPO_DIR}/scripts/cron_weekly_refresh_earnings.sh"

# Default: Mondays at 17:50 local time.
SCHEDULE="50 17 * * 1"

BEGIN_MARK="# options-helper: weekly earnings refresh"
END_MARK="# end options-helper: weekly earnings refresh"

INSTALL=0
if [[ "${1:-}" == "--install" ]]; then
  INSTALL=1
fi

if [[ "${INSTALL}" -eq 0 ]]; then
  echo "Add the following block to your crontab (crontab -e), or re-run with --install:"
  echo
  echo "${BEGIN_MARK}"
  echo "${SCHEDULE} ${CRON_CMD}"
  echo "${END_MARK}"
  echo
  echo "Logs: ${REPO_DIR}/data/logs/earnings_refresh.log"
  exit 0
fi

REPO_DIR="${REPO_DIR}" \
CRON_CMD="${CRON_CMD}" \
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

filtered += ["", begin, f"{schedule} {cron_cmd}", end, ""]
payload = "\n".join(filtered).lstrip("\n")

with tempfile.NamedTemporaryFile("w", delete=False) as f:
    f.write(payload)
    tmp_path = f.name

subprocess.run(["crontab", tmp_path], check=True, timeout=10)
print("Installed cron job:")
print(f"  {schedule} {cron_cmd}")
print(f"Logs: {repo_dir}/data/logs/earnings_refresh.log")
PY

