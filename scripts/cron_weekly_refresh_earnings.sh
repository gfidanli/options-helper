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

if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping earnings refresh." >> "${LOG_DIR}/earnings_refresh.log"
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
  echo "[$(date)] No watchlists file at ${WATCHLISTS}; skipping earnings refresh." >> "${LOG_DIR}/earnings_refresh.log"
  exit 0
fi

echo "[$(date)] Syncing positions watchlist + refreshing earnings..." >> "${LOG_DIR}/earnings_refresh.log"

"${VENV_BIN}/options-helper" watchlists sync-positions "${PORTFOLIO}" \
  --path "${WATCHLISTS}" \
  --name positions \
  >> "${LOG_DIR}/earnings_refresh.log" 2>&1

"${VENV_BIN}/options-helper" refresh-earnings \
  --watchlists-path "${WATCHLISTS}" \
  --cache-dir "${REPO_DIR}/data/earnings" \
  >> "${LOG_DIR}/earnings_refresh.log" 2>&1

