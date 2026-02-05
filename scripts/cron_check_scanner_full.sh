#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"

RUN_DATE="$(TZ=America/Chicago date +%F)"
LOG_PATH="${LOG_DIR}/scanner_full_${RUN_DATE}.log"
STATUS_PATH="${LOG_DIR}/scanner_full_status.json"

STATUS_LINE="$(STATUS_PATH="${STATUS_PATH}" RUN_DATE="${RUN_DATE}" python3 - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
run_date = os.environ["RUN_DATE"]

if not status_path.exists():
    print("missing")
    raise SystemExit(0)

try:
    data = json.loads(status_path.read_text(encoding="utf-8"))
except Exception:
    print("invalid")
    raise SystemExit(0)

date = str(data.get("date") or "")
status = str(data.get("status") or "")
print(f"{date}::{status}")
PY
)"

if [[ "${STATUS_LINE}" == "${RUN_DATE}::success" ]]; then
  echo "[$(date)] Full scanner already complete for ${RUN_DATE}; skipping." >> "${LOG_PATH}"
  exit 0
fi

if [[ "${STATUS_LINE}" == "${RUN_DATE}::running" || "${STATUS_LINE}" == "${RUN_DATE}::queued" || "${STATUS_LINE}" == "${RUN_DATE}::waiting_candle" ]]; then
  echo "[$(date)] Full scanner still in progress for ${RUN_DATE}; skipping check retry." >> "${LOG_PATH}"
  exit 0
fi

echo "[$(date)] Full scanner incomplete for ${RUN_DATE} (status=${STATUS_LINE}); retrying." >> "${LOG_PATH}"
"${REPO_DIR}/scripts/cron_daily_scanner_full.sh" >> "${LOG_PATH}" 2>&1
