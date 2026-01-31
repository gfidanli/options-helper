#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping daily options snapshot." >> "${LOG_DIR}/options_snapshot.log"
  exit 0
fi
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ."
  exit 1
fi

cd "${REPO_DIR}"

echo "[$(date)] Running daily options snapshot..."
"${VENV_BIN}/options-helper" refresh-candles "${PORTFOLIO}" \
  --watchlists-path "${REPO_DIR}/data/watchlists.json" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --period 5y \
  >> "${LOG_DIR}/options_snapshot.log" 2>&1

"${VENV_BIN}/options-helper" snapshot-options "${PORTFOLIO}" \
  --cache-dir "${REPO_DIR}/data/options_snapshots" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --window-pct 1.0 \
  >> "${LOG_DIR}/options_snapshot.log" 2>&1
