#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"
WATCHLISTS="${REPO_DIR}/data/watchlists.json"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"

# Wait for any network-heavy jobs to finish (avoid concurrent writes / stale watchlists).
WAIT_SECONDS="${WAIT_SECONDS:-14400}"
start_ts="$(date +%s)"
while ! mkdir "${LOCK_PATH}" 2>/dev/null; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= WAIT_SECONDS )); then
    echo "[$(date)] Timed out waiting for lock (${LOCK_PATH}); skipping report pack." >> "${LOG_DIR}/report_pack.log"
    exit 0
  fi
  sleep 30
done
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ."
  exit 1
fi

cd "${REPO_DIR}"

DATA_TZ="${DATA_TZ:-America/Chicago}"
RUN_DATE="$(TZ="${DATA_TZ}" date +%F)"

SNAPSHOT_ROOT="${REPO_DIR}/data/options_snapshots"
if [[ ! -d "${SNAPSHOT_ROOT}" ]]; then
  echo "[$(date)] No snapshot directory at ${SNAPSHOT_ROOT}; skipping report pack." >> "${LOG_DIR}/report_pack.log"
  exit 0
fi

if ! find "${SNAPSHOT_ROOT}" -mindepth 2 -maxdepth 2 -type d -name "${RUN_DATE}" -print -quit | grep -q .; then
  echo "[$(date)] No snapshot folders found for ${RUN_DATE}; skipping report pack." >> "${LOG_DIR}/report_pack.log"
  exit 0
fi

# Only include "Scanner - Shortlist" if today's scanner run succeeded; otherwise it's often stale.
INCLUDE_SCANNER=0
STATUS_PATH="${REPO_DIR}/data/logs/scanner_full_status.json"
if [[ -f "${STATUS_PATH}" ]]; then
  if python3 - <<'PY' "${STATUS_PATH}" "${RUN_DATE}"
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
run_date = sys.argv[2]
try:
    data = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)

date = str(data.get("date") or "")
status = str(data.get("status") or "")
raise SystemExit(0 if (date == run_date and status == "success") else 1)
PY
  then
    INCLUDE_SCANNER=1
  fi
fi

echo "[$(date)] Running offline report pack for ${RUN_DATE} (include_scanner=${INCLUDE_SCANNER})..." \
  >> "${LOG_DIR}/report_pack.log"

args=(
  report-pack
  "${PORTFOLIO}"
  --watchlists-path "${WATCHLISTS}"
  --watchlist positions
  --watchlist monitor
  --cache-dir "${REPO_DIR}/data/options_snapshots"
  --candle-cache-dir "${REPO_DIR}/data/candles"
  --derived-dir "${REPO_DIR}/data/derived"
  --out "${REPO_DIR}/data/reports"
  --as-of latest
  --compare-from -1
  --require-snapshot-date today
  --require-snapshot-tz "${DATA_TZ}"
)

if [[ "${INCLUDE_SCANNER}" -eq 1 ]]; then
  args+=(--watchlist "Scanner - Shortlist")
fi

"${VENV_BIN}/options-helper" "${args[@]}" >> "${LOG_DIR}/report_pack.log" 2>&1

