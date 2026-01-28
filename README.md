# options-helper

CLI-first MVP to analyze a simple options portfolio stored in JSON.

## Install (local dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

```bash
options-helper init portfolio.json
options-helper add-position portfolio.json --symbol UROY --expiry 2026-04-17 --strike 5 --type call --contracts 1 --cost-basis 0.45
options-helper analyze portfolio.json
```

Without installing an entrypoint script:

```bash
python3 -m options_helper init portfolio.json
```

## Features

### Portfolio (JSON) + CRUD
- Positions and risk profile are stored locally in a JSON file.
- Commands: `init`, `add-position`, `remove-position`, `list`
- Spec: `PRD.md`

### Multi-timeframe analysis + advice (LEAPS-oriented)
- Uses daily candles resampled into 3-business-day and weekly bars.
- Computes RSI + EMA trend state across timeframes and provides rule-based suggestions.
- Command: `analyze`

### Candle cache (daily OHLCV)
`analyze` maintains a local daily OHLCV cache per symbol and resamples it for 3‑day and weekly indicators.

- Design/usage: `docs/CANDLE_CACHE.md`

### Daily performance (best-effort)
Computes day’s P&L from options-chain `change` * contracts * 100 (best-effort; can be stale on illiquid chains).

- Command: `daily`

### Options flow (OI/volume deltas)
Collects a once-daily snapshot of a window around spot and computes ΔOI/volume-based positioning proxies.

- Commands: `snapshot-options`, `flow`
- Design/usage + cron setup: `docs/OPTIONS_FLOW.md`

## Disclaimer
This tool is for informational/educational use only and is not financial advice.
