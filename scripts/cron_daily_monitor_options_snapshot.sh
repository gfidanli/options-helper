#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"
PORTFOLIO="${REPO_DIR}/portfolio.json"
WATCHLISTS="${REPO_DIR}/data/watchlists.json"
WAIT_SCRIPT="${REPO_DIR}/scripts/wait_for_daily_candle_date.py"
DATA_TZ="${DATA_TZ:-America/Chicago}"
CANARY_SYMBOLS="${CANARY_SYMBOLS:-SPY,QQQ}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-7200}"   # 2h
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"          # 5m

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_cron.lock"
mkdir -p "${LOCKS_DIR}"

# Wait briefly for the daily snapshot job to finish (avoid concurrent cache writes).
WAIT_SECONDS="${WAIT_SECONDS:-14400}"
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

if ! python3 - <<'PY' "${WATCHLISTS}"
from __future__ import annotations

import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
except Exception:
    sys.exit(1)

watchlists = raw.get("watchlists") or {}
symbols = []
for name in ("monitor", "positions"):
    symbols.extend(watchlists.get(name) or [])
clean = {s.strip().upper() for s in symbols if isinstance(s, str) and s.strip()}
sys.exit(0 if clean else 1)
PY
then
  echo "[$(date)] Watchlists 'monitor'/'positions' missing/empty in ${WATCHLISTS}; skipping watchlist snapshot." >> "${LOG_DIR}/monitor_snapshot.log"
  exit 0
fi

echo "[$(date)] Running watchlist options snapshot (monitor + positions)..." >> "${LOG_DIR}/monitor_snapshot.log"

if ! "${VENV_BIN}/python" "${WAIT_SCRIPT}" \
  --symbols "${CANARY_SYMBOLS}" \
  --tz "${DATA_TZ}" \
  --expected-date today \
  --timeout-seconds "${WAIT_TIMEOUT_SECONDS}" \
  --poll-seconds "${WAIT_POLL_SECONDS}" \
  >> "${LOG_DIR}/monitor_snapshot.log" 2>&1
then
  echo "[$(date)] Timed out waiting for daily candle update; skipping watchlist snapshot to avoid mis-dating." \
    >> "${LOG_DIR}/monitor_snapshot.log"
  exit 0
fi

"${VENV_BIN}/options-helper" snapshot-options "${PORTFOLIO}" \
  --cache-dir "${REPO_DIR}/data/options_snapshots" \
  --candle-cache-dir "${REPO_DIR}/data/candles" \
  --watchlists-path "${WATCHLISTS}" \
  --watchlist monitor \
  --watchlist positions \
  --max-expiries 2 \
  --require-data-date today \
  --require-data-tz "${DATA_TZ}" \
  --windowed \
  --position-expiries \
  --window-pct 1.0 \
  >> "${LOG_DIR}/monitor_snapshot.log" 2>&1
