#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"
WATCHLISTS="${REPO_DIR}/data/watchlists.json"
PROVIDER="${PROVIDER:-alpaca}"
DATA_TZ="${DATA_TZ:-America/Chicago}"

RUN_DATE="$(TZ="${DATA_TZ}" date +%F)"
LOG_DIR="${REPO_DIR}/data/logs/${RUN_DATE}"
LOG_PATH="${LOG_DIR}/earnings_refresh.log"
mkdir -p "${LOG_DIR}"
SCRIPT_START_TS="$(date +%s)"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"

if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping earnings refresh." >> "${LOG_PATH}"
  exit 0
fi
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ."
  exit 1
fi

cd "${REPO_DIR}"

if [[ ! -f "${WATCHLISTS}" ]]; then
  echo "[$(date)] No watchlists file at ${WATCHLISTS}; skipping earnings refresh." >> "${LOG_PATH}"
  exit 0
fi

echo "[$(date)] Syncing positions watchlist + refreshing earnings..." >> "${LOG_PATH}"

"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" watchlists sync-positions "${PORTFOLIO}" \
  --path "${WATCHLISTS}" \
  --name positions \
  >> "${LOG_PATH}" 2>&1

"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" refresh-earnings \
  --watchlists-path "${WATCHLISTS}" \
  --cache-dir "${REPO_DIR}/data/earnings" \
  >> "${LOG_PATH}" 2>&1

SCRIPT_FINISH_TS="$(date +%s)"
SCRIPT_ELAPSED="$((SCRIPT_FINISH_TS - SCRIPT_START_TS))"
echo "[$(date)] Earnings refresh complete in ${SCRIPT_ELAPSED}s (provider=${PROVIDER})." >> "${LOG_PATH}"
