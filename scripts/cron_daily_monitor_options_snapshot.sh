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

# Wait briefly for the daily snapshot job to finish (avoid concurrent cache writes).
WAIT_SECONDS="${WAIT_SECONDS:-3600}"
start_ts="$(date +%s)"
while ! mkdir "${LOCK_PATH}" 2>/dev/null; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= WAIT_SECONDS )); then
    echo "[$(date)] Timed out waiting for lock (${LOCK_PATH}); skipping monitor snapshot." >> "${LOG_DIR}/monitor_snapshot.log"
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

if [[ ! -f "${WATCHLISTS}" ]]; then
  echo "[$(date)] No watchlists file at ${WATCHLISTS}; skipping monitor snapshot." >> "${LOG_DIR}/monitor_snapshot.log"
  exit 0
fi

echo "[$(date)] Running monitor watchlist options snapshot..." >> "${LOG_DIR}/monitor_snapshot.log"

"${VENV_BIN}/options-helper" snapshot-options "${PORTFOLIO}" \
  --cache-dir "${REPO_DIR}/data/options_snapshots" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --watchlists-path "${WATCHLISTS}" \
  --watchlist monitor \
  --max-expiries 2 \
  --window-pct 1.0 \
  >> "${LOG_DIR}/monitor_snapshot.log" 2>&1

