#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_DIR}/.venv/bin"

LOG_DIR="${REPO_DIR}/data/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/stream_capture.log"

LOCKS_DIR="${REPO_DIR}/data/locks"
LOCK_PATH="${LOCKS_DIR}/options_helper_stream_capture.lock"
mkdir -p "${LOCKS_DIR}"
if ! mkdir "${LOCK_PATH}" 2>/dev/null; then
  echo "[$(date)] Lock already held (${LOCK_PATH}); skipping stream capture." >> "${LOG_PATH}"
  exit 0
fi
trap 'rmdir "${LOCK_PATH}" 2>/dev/null || true' EXIT

if [[ ! -x "${VENV_BIN}/options-helper" ]]; then
  echo "options-helper not found at ${VENV_BIN}/options-helper" >> "${LOG_PATH}"
  echo "Create a venv and install deps: python3 -m venv .venv && ./.venv/bin/pip install -e \".[alpaca]\"" \
    >> "${LOG_PATH}"
  exit 1
fi

MARKET_TZ="${OH_ALPACA_MARKET_TZ:-${MARKET_TZ:-America/New_York}}"
export OH_ALPACA_MARKET_TZ="${MARKET_TZ}"

STOCK_SYMBOLS="${STOCK_SYMBOLS:-}"
OPTION_CONTRACTS="${OPTION_CONTRACTS:-}"
OPTION_UNDERLYINGS="${OPTION_UNDERLYINGS:-}"
OPTION_EXPIRIES="${OPTION_EXPIRIES:-}"
CONTRACTS_DIR="${CONTRACTS_DIR:-${REPO_DIR}/data/option_contracts}"
CONTRACTS_AS_OF="${CONTRACTS_AS_OF:-latest}"
MAX_CONTRACTS="${MAX_CONTRACTS:-250}"
INTRADAY_DIR="${INTRADAY_DIR:-${REPO_DIR}/data/intraday}"

DURATION_SECONDS="${DURATION_SECONDS:-3600}"
FLUSH_SECONDS="${FLUSH_SECONDS:-10}"
FLUSH_EVERY="${FLUSH_EVERY:-250}"
MAX_RECONNECTS="${MAX_RECONNECTS:-5}"

CAPTURE_BARS="${CAPTURE_BARS:-1}"
CAPTURE_QUOTES="${CAPTURE_QUOTES:-0}"
CAPTURE_TRADES="${CAPTURE_TRADES:-0}"

STOCK_FEED="${STOCK_FEED:-}"
OPTIONS_FEED="${OPTIONS_FEED:-}"

if [[ -z "${STOCK_SYMBOLS}" && -z "${OPTION_CONTRACTS}" && -z "${OPTION_UNDERLYINGS}" ]]; then
  echo "[$(date)] No symbols configured (set STOCK_SYMBOLS and/or OPTION_CONTRACTS and/or OPTION_UNDERLYINGS)." \
    >> "${LOG_PATH}"
  exit 0
fi

echo "[$(date)] Running stream capture (duration=${DURATION_SECONDS}s, market_tz=${MARKET_TZ})" >> "${LOG_PATH}"

extra_expiries=()
if [[ -n "${OPTION_EXPIRIES}" ]]; then
  extra_expiries=(--expiries "${OPTION_EXPIRIES}")
fi

bars_flag=(--no-bars)
if [[ "${CAPTURE_BARS}" == "1" ]]; then
  bars_flag=(--bars)
fi

quotes_flag=(--no-quotes)
if [[ "${CAPTURE_QUOTES}" == "1" ]]; then
  quotes_flag=(--quotes)
fi

trades_flag=(--no-trades)
if [[ "${CAPTURE_TRADES}" == "1" ]]; then
  trades_flag=(--trades)
fi

extra_feeds=()
if [[ -n "${STOCK_FEED}" ]]; then
  extra_feeds+=(--stock-feed "${STOCK_FEED}")
fi
if [[ -n "${OPTIONS_FEED}" ]]; then
  extra_feeds+=(--options-feed "${OPTIONS_FEED}")
fi

extra_stocks=()
if [[ -n "${STOCK_SYMBOLS}" ]]; then
  extra_stocks=(--stocks "${STOCK_SYMBOLS}")
fi

extra_contracts=()
if [[ -n "${OPTION_CONTRACTS}" ]]; then
  extra_contracts=(--options-contracts "${OPTION_CONTRACTS}")
fi

extra_underlyings=()
if [[ -n "${OPTION_UNDERLYINGS}" ]]; then
  extra_underlyings=(--options-underlyings "${OPTION_UNDERLYINGS}")
fi

"${VENV_BIN}/options-helper" --log-dir "${LOG_DIR}" --provider alpaca stream capture \
  "${extra_stocks[@]}" \
  "${extra_contracts[@]}" \
  "${extra_underlyings[@]}" \
  --contracts-dir "${CONTRACTS_DIR}" \
  --contracts-as-of "${CONTRACTS_AS_OF}" \
  --max-contracts "${MAX_CONTRACTS}" \
  "${extra_expiries[@]}" \
  --duration "${DURATION_SECONDS}" \
  --flush-seconds "${FLUSH_SECONDS}" \
  --flush-every "${FLUSH_EVERY}" \
  --max-reconnects "${MAX_RECONNECTS}" \
  --out-dir "${INTRADAY_DIR}" \
  "${bars_flag[@]}" \
  "${quotes_flag[@]}" \
  "${trades_flag[@]}" \
  "${extra_feeds[@]}" \
  >> "${LOG_PATH}" 2>&1 || true
