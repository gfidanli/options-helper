#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"
WAIT_SCRIPT="${REPO_DIR}/scripts/wait_for_daily_candle_date.py"
DATA_TZ="${DATA_TZ:-America/Chicago}"
CANARY_SYMBOLS="${CANARY_SYMBOLS:-SPY,QQQ}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-10800}"   # 3h
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"           # 5m
PROVIDER="${PROVIDER:-alpaca}"

RUN_DATE="$(TZ="${DATA_TZ}" date +%F)"
LOG_DIR="${REPO_DIR}/data/logs/${RUN_DATE}"
LOG_PATH="${LOG_DIR}/options_snapshot.log"
mkdir -p "${LOG_DIR}"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping daily options snapshot." >> "${LOG_PATH}"
  exit 0
fi
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ."
  exit 1
fi

cd "${REPO_DIR}"

SCRIPT_START_TS="$(date +%s)"
echo "[$(date)] Running daily options snapshot (provider=${PROVIDER})..." >> "${LOG_PATH}"
"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" watchlists sync-positions "${PORTFOLIO}" \
  --path "${REPO_DIR}/data/watchlists.json" \
  --name positions \
  >> "${LOG_PATH}" 2>&1 || true

"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" refresh-candles "${PORTFOLIO}" \
  --watchlists-path "${REPO_DIR}/data/watchlists.json" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --period 5y \
  >> "${LOG_PATH}" 2>&1

if ! "${VENV_BIN}/python" "${WAIT_SCRIPT}" \
  --symbols "${CANARY_SYMBOLS}" \
  --provider "${PROVIDER}" \
  --tz "${DATA_TZ}" \
  --expected-date today \
  --timeout-seconds "${WAIT_TIMEOUT_SECONDS}" \
  --poll-seconds "${WAIT_POLL_SECONDS}" \
  >> "${LOG_PATH}" 2>&1
then
  echo "[$(date)] Timed out waiting for daily candle update; skipping options snapshot to avoid mis-dating." \
    >> "${LOG_PATH}"
  exit 0
fi

# Re-run the candle refresh once the daily candle is published.
"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" refresh-candles "${PORTFOLIO}" \
  --watchlists-path "${REPO_DIR}/data/watchlists.json" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --period 5y \
  >> "${LOG_PATH}" 2>&1

"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" --log-path "${LOG_PATH}" snapshot-options "${PORTFOLIO}" \
  --cache-dir "${REPO_DIR}/data/options_snapshots" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --all-expiries \
  --full-chain \
  --require-data-date today \
  --require-data-tz "${DATA_TZ}" \
  >> "${LOG_PATH}" 2>&1

SCRIPT_FINISH_TS="$(date +%s)"
SCRIPT_ELAPSED="$((SCRIPT_FINISH_TS - SCRIPT_START_TS))"
echo "[$(date)] Daily options snapshot complete in ${SCRIPT_ELAPSED}s (provider=${PROVIDER})." \
  >> "${LOG_PATH}"
