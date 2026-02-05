#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"
WATCHLISTS="${REPO_DIR}/data/watchlists.json"
DATA_TZ="${DATA_TZ:-America/Chicago}"
PROVIDER="${PROVIDER:-alpaca}"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"
SCRIPT_START_TS="$(date +%s)"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"

# Wait for snapshot jobs to finish (avoid concurrent cache writes).
WAIT_SECONDS="${WAIT_SECONDS:-14400}"
start_ts="$(date +%s)"
while ! mkdir "${LOCK_PATH}" 2>/dev/null; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= WAIT_SECONDS )); then
    echo "[$(date)] Timed out waiting for lock (${LOCK_PATH}); skipping briefing." >> "${LOG_DIR}/briefing.log"
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

echo "[$(date)] Running daily briefing..." >> "${LOG_DIR}/briefing.log"

EXPECTED_DATE="$(TZ="${DATA_TZ}" date +%F)"
SNAPSHOT_ROOT="${REPO_DIR}/data/options_snapshots"
if [[ ! -d "${SNAPSHOT_ROOT}" ]]; then
  echo "[$(date)] No snapshot directory at ${SNAPSHOT_ROOT}; skipping briefing." >> "${LOG_DIR}/briefing.log"
  exit 0
fi

if ! find "${SNAPSHOT_ROOT}" -mindepth 2 -maxdepth 2 -type d -name "${EXPECTED_DATE}" -print -quit \
  | grep -q .
then
  echo "[$(date)] No snapshot folders found for ${EXPECTED_DATE}; skipping briefing to avoid overwriting an older report." \
    >> "${LOG_DIR}/briefing.log"
  exit 0
fi

if [[ -f "${WATCHLISTS}" ]]; then
  "${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" briefing "${PORTFOLIO}" \
    --watchlists-path "${WATCHLISTS}" \
    --watchlist positions \
    --watchlist monitor \
    --as-of latest \
    --compare -1 \
    --out "${REPO_DIR}/data/reports/daily" \
    >> "${LOG_DIR}/briefing.log" 2>&1
else
  "${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" briefing "${PORTFOLIO}" \
    --as-of latest \
    --compare -1 \
    --out "${REPO_DIR}/data/reports/daily" \
    >> "${LOG_DIR}/briefing.log" 2>&1
fi

SCRIPT_FINISH_TS="$(date +%s)"
SCRIPT_ELAPSED="$((SCRIPT_FINISH_TS - SCRIPT_START_TS))"
echo "[$(date)] Briefing complete in ${SCRIPT_ELAPSED}s (provider=${PROVIDER})." >> "${LOG_DIR}/briefing.log"
