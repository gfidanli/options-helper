# Backlog & Roadmap (Options Helper)

This document is the working backlog for next features, written as PRDs you can build against.
It is intentionally biased toward **offline, repeatable analysis** using the locally stored snapshot data under
`data/options_snapshots/`.

## Guiding principles
- **Offline-first:** reports should work from snapshot files (no live Yahoo calls required) so they can run on a schedule.
- **Change-aware:** default to “what changed since last snapshot?” not just “what is the chain today?”
- **Actionable outputs:** prioritize levels (strikes/expiries), ranges (expected move), and positioning deltas (ΔOI).
- **Deterministic & testable:** stable output schemas and golden tests using fixture snapshots; avoid network in tests.

---

## Ranked backlog (highest impact → lowest)

| Rank | ID    | Feature | Why it matters | Primary dependencies |
|------|-------|---------|----------------|----------------------|
| 1 | F-007 | Extension percentiles + regime drift | Adds per-ticker percentile context for extension and checks whether tails mean‑revert; supports adaptive thresholds. | technicals_backtesting + derived stats |

## Completed (implemented)
- F-001 `chain-report`
- F-002 `compare`
- F-003 `roll-plan`
- F-004 `flow` upgrade (aggregations + netting)
- F-005 `briefing` (daily report)
- F-006 `derived` (metrics store + stats)

---

## Milestones (delivery plan)

### Milestone M6 — Extension percentiles + regime drift
**Goal:** add per-ticker extension percentiles and tail‑event follow‑through analysis, with rolling-window drift checks.

Deliverables:
- F-007 MVP: compute extension percentiles from technicals_backtesting and add to briefing JSON + derived stats.
- F-007 v1: tail‑event forward‑window analysis (1/3/5/10 day percentiles) + rolling 1y/3y/5y distribution comparisons.

---

## Dependency map (summary)
- F-007 consumes technicals_backtesting features and extends derived stats with distribution/percentile analytics.

---

# PRDs

## F-007 — Extension percentiles + regime drift

### Summary
Add per-ticker percentile context for `extension_atr` and analyze tail‑event follow‑through (1/3/5/10D),
with rolling 1y/3y/5y distribution comparisons to detect regime drift.

### Goals
- Provide a **normalized, per‑ticker** extension percentile (current and rolling windows).
- Detect **tail events** (>= 95th / <= 5th) and summarize what typically happens next.
- Compare **rolling distributions** (1y/3y/5y) to identify drift and inform which window is most appropriate.

### Non‑goals
- Predictive modeling beyond descriptive statistics.
- Intraday signals (daily bars only).

### Outputs (v1)
- `technicals_backtesting` snapshot: `extension_atr` + percentile(s).
- `derived stats`: extension percentile(s) + drift summary.
- Tail event table: next‑day/3‑day/5‑day/10‑day percentile outcomes for each tail event day.

### Implementation notes
- Use `technicals_backtesting` as the canonical indicator source.
- Percentiles should be computed on a **per‑ticker** basis from cached candles.
- Rolling windows: 1y / 3y / 5y (if enough data).
- Drift: compare key quantiles (e.g., 5/50/95) and dispersion across windows.
- Drift **flagging** (programmatic thresholds) is a future enhancement; v1 outputs a table only.

### Testing
- Deterministic synthetic OHLC series for percentile and tail‑event logic.
- JSON payload tests for LLM‑friendly outputs.
  - top |ΔOI premium-notional|
  - top expiries by net ΔOI premium-notional

### Milestones
- **MVP:** spot + IV term changes + wall diffs + top contract ΔOI (where possible).
- **v1:** better wall change detection (new wall vs moved wall), multi-day comparison shortcuts.
- **v2:** “change narratives” (e.g., “front vol crushed”, “upside OI building shifted to X”).

### Dependencies
- F-001 metrics API (compute metrics for each day).
- Multi-day snapshot loader (list dates, load day).
- Flow computation (existing `compute_flow`) for contract-level deltas where coverage exists.

### Acceptance criteria
- For symbols with ≥2 snapshots, `compare` prints a concise diff and writes a deterministic JSON artifact.

### Testing
- Fixture snapshots with known diffs and golden expected outputs.

---

## F-003 — `roll-plan` (Roll / Position Planner)

### Summary
Add `options-helper roll-plan` to propose and rank roll candidates for an existing portfolio position based on intent
and horizon (e.g., “maximize upside over 12 months”).

### Problem
Turning a winning option into a longer-duration bet is a common need, but users lack a consistent, explainable way to:
- align DTE to thesis horizon,
- compare delta/theta tradeoffs,
- avoid illiquid traps,
- understand roll cost.

### Goals
- Provide a ranked list of roll candidates with transparent scoring and rationale.
- Default to using local snapshots when available; optionally allow live chain fetch explicitly.
- Support intent-driven planning (`max-upside`, `reduce-theta`, `increase-delta`, etc.).

### Non-goals
- Autotrading / broker integration.
- Full multi-leg strategy system (keep to simple roll shapes at first).

### CLI / UX
```bash
options-helper roll-plan portfolio.json --id cvx-2026-06-18-190c --intent max-upside --horizon-months 12
options-helper roll-plan portfolio.json --id cvx-2026-06-18-190c --intent max-upside --horizon-months 12 --max-debit 600
```

Options (v1):
- `--as-of YYYY-MM-DD|latest` (prefer snapshot date)
- `--intent max-upside|reduce-theta|increase-delta|de-risk`
- `--horizon-months INT`
- `--shape out-same-strike|out-up|out-down` (default depends on intent)
- `--max-debit FLOAT` / `--min-credit FLOAT`
- `--min-open-interest INT` / `--min-volume INT` (defaults from `risk_profile`)
- `--use-live-chain` (opt-in; otherwise offline snapshot-only)

### Candidate generation
- Select expiries nearest to a target DTE implied by `--horizon-months` (e.g., ±90 days).
- Generate strikes via:
  - same strike (anchor thesis),
  - target strike (e.g., 200),
  - target delta bands (e.g., 0.30/0.40/0.50) using BS deltas in snapshots (best-effort).

### Scoring (transparent)
- Liquidity gate: OI/volume, spread sanity (bid/ask).
- Horizon fit: DTE closeness to target.
- Intent fit:
  - `max-upside`: prioritize time extension + keep upside uncapped (no vertical cap suggestions by default).
  - `reduce-theta`: minimize theta per $.
  - `increase-delta`: maximize delta per $.
- Output should show “why this ranked #1”.

### Outputs
- Table: current contract vs candidates with:
  - mark, roll debit/credit, DTE, delta, theta/day, IV, OI, volume, bid/ask spread.
- Recommendation bullets + risks (e.g., “adds time but costs X”, “more OTM reduces delta”, “illiquid spread”).

### Milestones
- **MVP:** single-position roll table from snapshots (same symbol) + basic scoring + `max-upside`.
- **v1:** multi-intent support + `--use-live-chain` + better delta-target selection.
- **v2:** portfolio-aware roll planning (budget checks, sequencing, partial closes/rolls).

### Dependencies
- Snapshot metrics helpers (mark, liquidity checks).
- Existing Black–Scholes code for consistent delta/theta calculations where needed.
- Portfolio loader and position ID lookup (already exists).

### Acceptance criteria
- For an existing portfolio position, prints a ranked list with rationale and passes liquidity gates.
- Provides at least one candidate that extends DTE to match horizon when expiries exist in snapshots.

### Testing
- Fixture-based tests for ranking logic and table output values (no network).

---

## F-004 — `flow` upgrade (Aggregations + Multi-day Netting)

### Summary
Extend `options-helper flow` to:
- net across multiple snapshot pairs (`--window N`),
- aggregate by strike and/or expiry,
- include delta-notional alongside premium-notional,
- surface “hot zones” not just individual contracts.

### Problem
Contract-level flow lists are noisy; we need strike/expiry aggregation and persistence across days to improve signal.

### Goals
- Add group-by views that answer “where is positioning building?”.
- Make it easy to look at last 2/5/10 snapshots for persistence.

### Non-goals
- Tape classification (sweeps/blocks) and intraday flow.

### CLI / UX
```bash
options-helper flow portfolio.json --symbol CVX --window 5 --group-by strike
options-helper flow portfolio.json --symbol CVX --window 5 --group-by expiry
options-helper flow portfolio.json --symbol CVX --window 5 --group-by expiry-strike --top 20
```

### Metrics
- Existing per-contract:
  - ΔOI, premium-notional (ΔOI * mark * 100)
- Add delta-notional:
  - ΔOI * delta * spot * 100 (best-effort; delta from stored BS delta)

### Milestones
- **MVP:** `--window`, `--group-by`, delta-notional, aggregated top lists.
- **v1:** persistence scoring (e.g., days building vs unwinding), strike-zone bucketing (e.g., $2.50 increments).
- **v2:** output artifacts per symbol/day for the briefing and derived store.

### Dependencies
- Existing `compute_flow` logic (or a refactor to share “flow row builder”).
- Multi-date snapshot loading + consistent spot extraction (from `meta.json`).

### Acceptance criteria
- For symbols with enough snapshots, produces aggregated net flows by strike and expiry with deterministic ordering.

### Testing
- Snapshot fixtures with known deltas; unit tests for aggregation math and windowing.

---

## F-005 — `briefing` (Daily Report Generator)

### Summary
Add a cron-friendly command to generate a daily Markdown briefing for portfolio + watchlists, combining chain report,
compare highlights, and flow summaries.

### Problem
Analysis becomes regular only when it’s automated and saved as an artifact you can review later.

### Goals
- One command produces `data/reports/daily/YYYY-MM-DD.md`.
- Minimal noise: each symbol gets a concise “state + change + flow” section.

### Non-goals
- Web dashboards (keep to files + console for now).

### CLI / UX
```bash
options-helper briefing portfolio.json --watchlists-path data/watchlists.json --watchlist positions --watchlist monitor --as-of latest --compare -1
```

Options:
- `--as-of YYYY-MM-DD|latest`
- `--compare -1|-5|YYYY-MM-DD` (previous snapshot by default)
- `--out PATH` (default `data/reports/daily/{ASOF}.md`)
- `--symbol CVX` (optional filter)

### Contents (v1)
- Portfolio summary + per-position note (reuse `analyze` metrics where possible).
- Per symbol:
  - chain “key levels” bullets (walls, expected move, IV/skew)
  - compare “what changed” bullets
  - flow “net building/unwinding zones” bullets (if available)

### Milestones
- **MVP:** markdown output for portfolio symbols; optional watchlists.
- **v1:** add `--all-watchlists` and symbol filters; attach JSON artifacts for machine use.
- **v2:** templates + customization (what sections to include).

### Dependencies
- F-001 `chain-report` (metrics and/or rendered output)
- F-002 `compare`
- F-004 flow aggregations
- Watchlists loader (`options_helper/watchlists.py`)

### Acceptance criteria
- Produces a deterministic Markdown file for fixture inputs and doesn’t require network access.

### Testing
- Golden-file Markdown tests using fixture snapshots and a fixture portfolio/watchlists file.

---

## F-006 — `derived` (Derived Metrics Store)

### Summary
Persist a compact per-symbol, per-day time series of derived metrics computed from snapshots so we can compute
rolling percentiles and trends without re-parsing full chains.

### Problem
We can’t easily answer “is IV high vs the last 60 days?” or “are walls drifting higher for a week?” without a stored series.

### Goals
- Append/idempotently upsert a daily row per symbol.
- Provide “show last N rows” and “percentile vs last N” helpers.

### Non-goals
- Replacing snapshots (snapshots remain source-of-truth).

### Data model
File per symbol:
- `data/derived/{SYMBOL}.csv` (or JSONL if easier to extend; choose one and keep stable)

Recommended columns (initial):
- `date,spot,pc_oi,pc_vol,call_wall,put_wall,gamma_peak_strike,atm_iv_near,em_near_pct,skew_near_pp`

### CLI / UX
```bash
options-helper derived update --symbol CVX --as-of 2026-01-30
options-helper derived show --symbol CVX --last 30
```

### Milestones
- **MVP:** update + show, idempotent per day, schema stability.
- **v1:** percentile ranks and trend flags (e.g., IV 80th percentile of last 60 snapshots).
- **v2:** derived metrics for multiple expiries (front weekly + next monthly + quarterly).

### Dependencies
- F-001 metric computation (source of derived fields).
- Optional: F-002 for storing deltas as separate fields.

### Acceptance criteria
- Running `derived update` twice for the same symbol/day overwrites that day’s row (no duplicates).

### Testing
- Idempotency tests and schema tests using fixture snapshots.
