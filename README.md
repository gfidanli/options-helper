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

Offline/deterministic analysis (no live Yahoo calls; uses `data/options_snapshots/` + `data/candles/`):

```bash
options-helper analyze portfolio.json --offline --as-of latest
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
- Also prints portfolio-level Greeks + stress scenarios (spot/vol/time). Flags: `--stress-spot-pct`, `--stress-vol-pp`, `--stress-days`.

### Watchlists
Maintain multiple named symbol lists for research workflows.

- Commands: `watchlists init`, `watchlists list`, `watchlists show`, `watchlists add`, `watchlists remove`, `watchlists sync-positions`
- Design/usage: `docs/WATCHLISTS.md`

### Research (option ideas)
Recommends short-dated (30–90 DTE) and long-dated (LEAPS) contracts to consider, based on multi-timeframe technicals.

- Command: `research`
- Design/usage: `docs/RESEARCH.md`

### Candle cache (daily OHLCV)
`analyze` maintains a local daily OHLCV cache per symbol and resamples it for 3‑day and weekly indicators.

- Design/usage: `docs/CANDLE_CACHE.md`

### Daily performance (best-effort)
Computes day’s P&L from options-chain `change` * contracts * 100 (best-effort; can be stale on illiquid chains).

- Command: `daily`

### Options flow (OI/volume deltas)
Collects once-daily options-chain snapshots and computes ΔOI/volume-based positioning proxies.
Defaults to full-chain + all expiries; use `--windowed --position-expiries` for smaller flow-focused snapshots.

- Commands: `snapshot-options`, `flow`
- Design/usage + cron setup: `docs/OPTIONS_FLOW.md`

### Automation (cron)
Recommended recurring data pulls and reporting jobs, with order/dependencies.

- Doc: `docs/AUTOMATION.md`

### Offline chain intelligence (from snapshots)
Builds repeatable chain dashboards and diffs using the local snapshot files under `data/options_snapshots/`.

- Commands: `chain-report`, `compare`, `roll-plan`
- Design/usage: `docs/CHAIN_REPORT.md`, `docs/COMPARE.md`, `docs/ROLL_PLAN.md`

### Daily briefing + derived history (from snapshots)
Automates a daily Markdown artifact and maintains a compact per-symbol derived-metrics history.

- Commands: `briefing`, `derived`
- Design/usage: `docs/BRIEFING.md`, `docs/DERIVED.md`

## Roadmap
- Ranked improvements: `docs/REPO_IMPROVEMENTS.md`
- Feature PRDs and milestones: `docs/BACKLOG.md`
- Implementation plans (LLM-friendly): `docs/plans/`

## Disclaimer
This tool is for informational/educational use only and is not financial advice.
