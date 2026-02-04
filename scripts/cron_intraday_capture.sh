#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/intraday_capture.log"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_intraday_capture.lock"
mkdir -p "${LOCKS_DIR}"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping intraday capture." >> "${LOG_PATH}"
  exit 0
fi
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper" >> "${LOG_PATH}"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e ." \
    >> "${LOG_PATH}"
  exit 1
fi

DAY="${DAY:-$(date +%F)}"
TIMEFRAME="${TIMEFRAME:-1Min}"
STOCK_SYMBOLS="${STOCK_SYMBOLS:-}"
OPTION_UNDERLYINGS="${OPTION_UNDERLYINGS:-}"
OPTION_EXPIRIES="${OPTION_EXPIRIES:-}"
CONTRACTS_DIR="${CONTRACTS_DIR:-${REPO_DIR}/data/option_contracts}"
CONTRACTS_AS_OF="${CONTRACTS_AS_OF:-latest}"
INTRADAY_DIR="${INTRADAY_DIR:-${REPO_DIR}/data/intraday}"

if [[ -z "${STOCK_SYMBOLS}" && -z "${OPTION_UNDERLYINGS}" ]]; then
  echo "[$(date)] No symbols configured (set STOCK_SYMBOLS and/or OPTION_UNDERLYINGS)." >> "${LOG_PATH}"
  exit 0
fi

echo "[$(date)] Running intraday capture (day=${DAY}, timeframe=${TIMEFRAME})" >> "${LOG_PATH}"

if [[ -n "${STOCK_SYMBOLS}" ]]; then
  "${VENV_BIN}/options-helper" --log-dir "${LOG_DIR}" --provider alpaca intraday pull-stocks-bars \
    --symbols "${STOCK_SYMBOLS}" \
    --day "${DAY}" \
    --timeframe "${TIMEFRAME}" \
    --out-dir "${INTRADAY_DIR}" \
    >> "${LOG_PATH}" 2>&1 || true
fi

if [[ -n "${OPTION_UNDERLYINGS}" ]]; then
  extra_expiries=()
  if [[ -n "${OPTION_EXPIRIES}" ]]; then
    extra_expiries=(--expiries "${OPTION_EXPIRIES}")
  fi
  "${VENV_BIN}/options-helper" --log-dir "${LOG_DIR}" --provider alpaca intraday pull-options-bars \
    --underlyings "${OPTION_UNDERLYINGS}" \
    --contracts-dir "${CONTRACTS_DIR}" \
    --contracts-as-of "${CONTRACTS_AS_OF}" \
    --day "${DAY}" \
    --timeframe "${TIMEFRAME}" \
    --out-dir "${INTRADAY_DIR}" \
    "${extra_expiries[@]}" \
    >> "${LOG_PATH}" 2>&1 || true
fi
