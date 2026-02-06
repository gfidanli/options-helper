#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
WAIT_SCRIPT="${REPO_DIR}/scripts/wait_for_daily_candle_date.py"
DATA_TZ="${DATA_TZ:-America/Chicago}"
CANARY_SYMBOLS="${CANARY_SYMBOLS:-SPY,QQQ}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-3600}"  # 1h
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"         # 5m
PROVIDER="${PROVIDER:-alpaca}"

RUN_DATE="$(TZ="${DATA_TZ}" date +%F)"
LOG_DIR="${REPO_DIR}/data/logs/${RUN_DATE}"
LOG_PATH="${LOG_DIR}/scanner_full.log"
STATUS_PATH="${LOG_DIR}/scanner_full_status.json"
mkdir -p "${LOG_DIR}"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ."
  exit 1
fi

RUN_ID="scanner-full-${RUN_DATE}"
START_TS="$(TZ="${DATA_TZ}" date -Iseconds)"
SCRIPT_START_TS="$(date +%s)"

write_status() {
  STATUS="${1}"
  FINISH_TS="${2}"
  EXIT_CODE="${3}"
  STATUS_PATH="${STATUS_PATH}" \
  RUN_DATE="${RUN_DATE}" \
  RUN_ID="${RUN_ID}" \
  STATUS="${STATUS}" \
  START_TS="${START_TS}" \
  FINISH_TS="${FINISH_TS}" \
  EXIT_CODE="${EXIT_CODE}" \
  LOG_PATH="${LOG_PATH}" \
  python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

payload = {
    "date": os.environ["RUN_DATE"],
    "run_id": os.environ["RUN_ID"],
    "status": os.environ["STATUS"],
    "started_at": os.environ["START_TS"],
    "finished_at": os.environ["FINISH_TS"] or None,
    "exit_code": int(os.environ["EXIT_CODE"]) if os.environ["EXIT_CODE"] else None,
    "log_path": os.environ["LOG_PATH"],
}

path = Path(os.environ["STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

write_status "waiting_candle" "" ""
if ! "${VENV_BIN}/python" "${WAIT_SCRIPT}" \
  --symbols "${CANARY_SYMBOLS}" \
  --provider "${PROVIDER}" \
  --tz "${DATA_TZ}" \
  --expected-date today \
  --timeout-seconds "${WAIT_TIMEOUT_SECONDS}" \
  --poll-seconds "${WAIT_POLL_SECONDS}" \
  >> "${LOG_PATH}" 2>&1
then
  echo "[$(date)] Daily candle not ready; skipping scanner run for ${RUN_ID}." >> "${LOG_PATH}"
  FINISH_TS="$(TZ="${DATA_TZ}" date -Iseconds)"
  write_status "skipped" "${FINISH_TS}" "0"
  exit 0
fi

write_status "queued" "" ""
WAIT_SECONDS="${WAIT_SECONDS:-14400}"
start_ts="$(date +%s)"
while ! mkdir "${LOCK_PATH}" 2>/dev/null; do
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= WAIT_SECONDS )); then
    echo "[$(date)] Timed out waiting for lock (${LOCK_PATH}); skipping full scanner run." >> "${LOG_PATH}"
    FINISH_TS="$(TZ="${DATA_TZ}" date -Iseconds)"
    write_status "lock_timeout" "${FINISH_TS}" "0"
    exit 0
  fi
  sleep 30
done
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

cd "${REPO_DIR}"

echo "[$(date)] Starting full scanner run (run_id=${RUN_ID})..." >> "${LOG_PATH}"
write_status "running" "" ""

set +e
"${VENV_BIN}/options-helper" --provider "${PROVIDER}" --log-dir "${LOG_DIR}" scanner run \
  --log-path "${LOG_PATH}" \
  --universe "file:${REPO_DIR}/data/universe/sec_company_tickers.json" \
  --no-skip-scanned \
  --no-write-scanned \
  --batch-size 25 \
  --run-id "${RUN_ID}" \
  --run-dir "${REPO_DIR}/data/scanner/runs" \
  --watchlists-path "${REPO_DIR}/data/watchlists.json" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --options-cache-dir "${REPO_DIR}/data/options_snapshots" \
  >> "${LOG_PATH}" 2>&1
EXIT_CODE=$?
set -e

FINISH_TS="$(TZ="${DATA_TZ}" date -Iseconds)"
if [[ "${EXIT_CODE}" -eq 0 ]]; then
  echo "[$(date)] Full scanner run completed successfully." >> "${LOG_PATH}"
  write_status "success" "${FINISH_TS}" "${EXIT_CODE}"
else
  echo "[$(date)] Full scanner run failed (exit ${EXIT_CODE})." >> "${LOG_PATH}"
  write_status "failed" "${FINISH_TS}" "${EXIT_CODE}"
fi

SCRIPT_FINISH_TS="$(date +%s)"
SCRIPT_ELAPSED="$((SCRIPT_FINISH_TS - SCRIPT_START_TS))"
echo "[$(date)] Full scanner runtime ${SCRIPT_ELAPSED}s (provider=${PROVIDER}, exit=${EXIT_CODE})." \
  >> "${LOG_PATH}"

exit "${EXIT_CODE}"
