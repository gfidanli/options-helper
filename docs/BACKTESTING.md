# Backtesting (Options)

Offline, daily-resolution options backtesting using **stored snapshots** and **cached candles**.
This tool is for research and education only — **not financial advice**.

## Requirements
- Options snapshots already saved under `data/options_snapshots/` (from `snapshot-options`).
- Cached daily candles under `data/candles/` (used for trend + extension context).

## Run a backtest

Example (single contract):
```
./.venv/bin/options-helper backtest run \
  --symbol AAPL \
  --contract-symbol AAPL260320C00150000 \
  --fill-mode mark_slippage \
  --slippage-factor 0.5
```

Example (expiry/strike selector):
```
./.venv/bin/options-helper backtest run \
  --symbol AAPL \
  --expiry 2026-03-20 \
  --strike 150 \
  --option-type call
```

## Fill models
- `worst_case`: buy at ask, sell at bid (skips trades when bid/ask missing).
- `mark_slippage`: uses mark (or mid) +/- spread-based slippage.

## Rolling
Enable rolling with:
```
--roll-dte-threshold 14
```
Optional knobs:
- `--roll-horizon-months`
- `--roll-shape` (out-same-strike | out-up | out-down)
- `--roll-intent` (max-upside | reduce-theta | increase-delta | de-risk)

## Artifacts
Backtest runs write to:
```
data/reports/backtests/<run_id>/
  trades.csv
  trades.json
  summary.json
  report.md
```

View the latest report:
```
./.venv/bin/options-helper backtest report --latest
```

## Notes / Caveats
- Daily resolution only (no intraday fills).
- Uses snapshot dates as the timeline (gaps remain gaps).
- Quotes are best-effort; missing bid/ask can skip fills.
- Results are research-only; **not financial advice**.
