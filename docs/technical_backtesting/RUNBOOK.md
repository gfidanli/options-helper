# Runbook

This tooling is for research and decision-support only. It is not financial advice.

## 1) Prerequisites
- Installed repo environment (`./.venv`)
- Technical backtesting config at `config/technical_backtesting.yaml`
- For intraday mode: local intraday partitions under `data/intraday`

## 2) MeanReversionIBS Batch Backtest (`technicals backtest-batch`)
Single symbol:

```bash
./.venv/bin/options-helper technicals backtest-batch \
  --symbol SPY \
  --strategy MeanReversionIBS \
  --config-path config/technical_backtesting.yaml \
  --cache-dir data/cache/candles
```

Multiple symbols:

```bash
./.venv/bin/options-helper technicals backtest-batch \
  --symbol SPY \
  --tickers QQQ,IWM \
  --strategy MeanReversionIBS \
  --cache-dir data/cache/candles
```

Available run-time overrides:
- Costs: `--commission`, `--slippage-bps`
- Overlays: `--require-sma-trend/--no-require-sma-trend`, `--sma-trend-window`,
  `--require-weekly-trend/--no-require-weekly-trend`,
  `--require-ma-direction/--no-require-ma-direction`, `--ma-direction-window`,
  `--ma-direction-lookback`

Completion output:
- Always prints `Batch backtest complete: symbols=<N> success=<N> failed=<N>`.
- If any symbol fails, prints `Failed symbols: ...` and continues processing the rest.
- Prints disclaimer: `Informational output only; not financial advice.`

## 3) Backtest-Batch Artifacts and Parity Inputs
`technicals backtest-batch` currently emits console output and does not expose a `--out` artifact flag.

The batch artifact contract is still defined and used by writer/test seams:
- Root: `data/reports/technicals/backtest_batch/{RUN_ID}/`
- Files:
  - `summary.json`
  - `report.html`
  - `per_symbol_metrics.csv`
  - `equity_curve.csv`
  - `monthly_returns.csv`
  - `yearly_returns.csv`
  - `failed_symbols.csv`

Reference parity workflow:

```bash
./.venv/bin/python scripts/validate_mean_reversion_reference.py
```

Optional run-summary tolerance check (against locked SPY/QQQ targets):

```bash
./.venv/bin/python scripts/validate_mean_reversion_reference.py \
  --run-summary-path /path/to/summary.json
```

## 4) Data Source Selection (CLI)
For `technicals optimize`, `technicals walk-forward`, and `technicals run-all`, OHLC source precedence is:
1. `--ohlc-path`
2. `--intraday-dir` (requires intraday options below)
3. `--symbol` + `--cache-dir`

## 5) Intraday Options
Available on `optimize`, `walk-forward`, and `run-all`:
- `--interval`: target candle interval token (used for intraday resampling and artifact namespacing)
- `--intraday-dir`: intraday partition root (enables intraday mode)
- `--intraday-timeframe`: source partition timeframe (`1Min` or `5Min`)
- `--intraday-start`: start session date (`YYYY-MM-DD`)
- `--intraday-end`: end session date (`YYYY-MM-DD`)

Intraday mode requirements:
- `--symbol` is required.
- `--intraday-start` and `--intraday-end` are required.
- `--intraday-end` must be on/after `--intraday-start`.
- Target `--interval` must be >= base timeframe and an integer multiple.

If `--interval` is omitted:
- intraday mode defaults to `--intraday-timeframe`
- non-intraday mode defaults to `1d`

## 6) Daily Workflow Examples
Optimize one strategy from daily candle cache:
```bash
./.venv/bin/options-helper technicals optimize \
  --strategy TrendPullbackATR \
  --symbol AAPL \
  --cache-dir data/candles
```

Walk-forward one strategy:
```bash
./.venv/bin/options-helper technicals walk-forward \
  --strategy MeanReversionBollinger \
  --symbol SPY \
  --cache-dir data/candles
```

## 7) Intraday Workflow Examples
Walk-forward `CvdDivergenceMSB` on intraday input:
```bash
./.venv/bin/options-helper technicals walk-forward \
  --strategy CvdDivergenceMSB \
  --symbol SPY \
  --intraday-dir data/intraday \
  --intraday-timeframe 1Min \
  --intraday-start 2025-01-02 \
  --intraday-end 2025-01-31 \
  --interval 15m
```

Run all enabled strategies for multiple symbols in intraday mode:
```bash
./.venv/bin/options-helper technicals run-all \
  --tickers SPY,QQQ \
  --intraday-dir data/intraday \
  --intraday-timeframe 5Min \
  --intraday-start 2025-01-02 \
  --intraday-end 2025-01-31 \
  --interval 30m
```

## 8) No-Lookahead Semantics
- Weekly regime (`weekly_trend_up`, `weekly_sma_*`) is shifted by one completed weekly bar before forward-fill.
- `CvdDivergenceMSB` pivots are consumed only after pivot confirmation lag (`pivot_right`).
- Entry is close-confirmed (`Close[t] > break_level`), then filled next bar open when `backtest.trade_on_close=false`.
- For `MeanReversionIBS`, entry/exit signals are close-known, and with `trade_on_close=false` fills occur at the next bar open.

## 9) UTC / Resampling Semantics
- Canonical timestamps are UTC-naive.
- tz-aware timestamps are converted to UTC and then made tz-naive.
- tz-naive timestamps are treated as UTC.
- Intraday resampling uses UTC bucket boundaries (`label=left`, `closed=left`).

## 10) Artifacts
Default templates include `{interval}`:
- `params/{interval}/{ticker}/{strategy}.json`
- `reports/{interval}/{ticker}/{strategy}/summary.md`
- `reports/{interval}/{ticker}/{strategy}/heatmap.csv`

Intraday runs also write `data.intraday_coverage` into params artifacts (requested/loaded/missing/empty day counts and day lists).

## 11) Troubleshooting
- `No OHLC data found`: verify selected input source and required flags.
- Interval validation errors: use intervals like `1Min`, `5Min`, `15m`, `30m`, `1h` and keep them compatible with base timeframe.
- Sparse intraday history: loader warns and continues; check `intraday_coverage` in params artifact.
- Slow optimization: narrow `search_space` or use `optimization.method: sambo`.
