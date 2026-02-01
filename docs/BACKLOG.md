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
| 2 | F-008 | RSI divergence enrichment for extension | Adds a momentum-confirmation layer: elevated extension + bearish RSI divergence can flag “take profit / reversal risk” zones; symmetric bullish divergence helps with “capitulation” context. | technicals_backtesting + extension percentiles |
| 3 | F-009 | Max-upside (High-based) + weekly context for extension tails | Aligns tail-event stats with longer-dated options positioning by summarizing max upside over 1w/4w/3m/6m/1y and adding weekly “context” (weekly extension/RSI/divergence). | extension percentiles + RSI divergence |

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

### Milestone M7 — RSI divergence enrichment for extension
**Goal:** enrich extension “tail context” with RSI(14) divergence detection so reports can distinguish
“strong trend (momentum confirms)” from “fragile extension (momentum diverges)”.

Deliverables:
- F-008 MVP: compute bearish/bullish RSI divergence flags on cached candles and surface them in `technicals extension-stats` artifacts.
- F-008 v1: conditional forward-return / forward-percentile tables for tail events **with** vs **without** divergence; add a concise “current divergence” callout to the daily briefing.

### Milestone M8 — Max-upside + weekly context for extension tails
**Goal:** add long-horizon, call-friendly “max upside” statistics for downside extension setups and
surface weekly extension/RSI/divergence context alongside daily tail events.

Deliverables:
- F-009 MVP: High-based max-upside summary for daily low-tail events over 1w/4w/3m/6m/1y; add +15D to daily forward extension percentiles.
- F-009 v1: weekly RSI divergence + weekly RSI regime tagging; add weekly context columns to daily tail-event tables.

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

## F-008 — RSI divergence enrichment for extension

### Summary
Add RSI(14) divergence detection as an enrichment layer to extension percentile reports, so “extended” conditions can be
qualified by whether momentum is confirming (no divergence) or weakening (bearish divergence).

This project is **not financial advice**. These signals are descriptive context for position management workflows.

### Problem
Extension tails are common in trending markets. What’s missing is a lightweight way to identify “extension is still high,
but momentum is weakening” (classic divergence), which often corresponds to “take profit / tighten risk” regions.

### Goals
- Detect **bearish RSI divergence** during elevated extension:
  - within a rolling lookback window (default 14 trading days),
  - Close makes a **higher high** (closing basis),
  - RSI(14) makes a **lower high**.
- Detect **bullish RSI divergence** during depressed extension:
  - Close makes a **lower low** while RSI makes a **higher low**.
- Flag whether a divergence occurs at **overbought/oversold** RSI conditions (configurable thresholds), since
  divergences at RSI extremes are often treated differently than divergences in the middle of the range.
- Quantify whether divergence changes the typical tail outcomes:
  - forward returns (1/3/5/10D),
  - forward extension percentile reversion.

### Non-goals
- “Best” swing-point detection or complex pattern libraries (keep logic simple + explainable).
- Intraday divergence (keep to daily/weekly candles only).
- Predictive claims; this is research context only.

### Definitions (v1)
- RSI: RSI(14) computed on Close (Wilder/EMA smoothing; consistent with existing indicator tooling).
- Lookback window: `divergence_window_days` (default 14).
- RSI regime thresholds:
  - `rsi_overbought` (default 70)
  - `rsi_oversold` (default 30)
  - A divergence is tagged as:
    - `overbought` if the newer swing RSI >= `rsi_overbought`
    - `oversold` if the newer swing RSI <= `rsi_oversold`
    - otherwise `neutral`
- Swing highs/lows (simple, deterministic):
  - A swing high at index `i` if `Close[i] >= Close[i-1]` and `Close[i] >= Close[i+1]` (same for RSI if needed).
  - Use the **two most recent** swing highs (or lows) within the window; require a minimum separation (e.g., 2 bars)
    to avoid duplicates.
- Bearish divergence event (daily):
  - Close swing high #2 > Close swing high #1 (optionally by `min_price_delta_pct`),
  - RSI at swing high #2 < RSI at swing high #1 (optionally by `min_rsi_delta`),
  - Extension is “elevated” over the window (choose one):
    - percentile-based: extension percentile >= `min_extension_percentile` for at least `min_extension_days`, OR
    - level-based: extension_atr >= `min_extension_atr` for at least `min_extension_days`.
- Bullish divergence event (daily):
  - Close swing low #2 < Close swing low #1 (optionally by `min_price_delta_pct`),
  - RSI at swing low #2 > RSI at swing low #1 (optionally by `min_rsi_delta`),
  - Extension is “depressed” over the window (choose one):
    - percentile-based: extension percentile <= `max_extension_percentile` for at least `min_extension_days`, OR
    - level-based: extension_atr <= `max_extension_atr` for at least `min_extension_days`.

### Outputs
- Extend `technicals extension-stats` report artifacts (`.json` + `.md`) with a new section:
  - “Current divergence status” (present/absent, dates of the two swing points, price/RSI deltas).
  - “Tail-event table enrichment” (per tail event day, include `bearish_divergence_14d` / `bullish_divergence_14d`).
  - “RSI regime tag” for each divergence (`overbought` / `oversold` / `neutral`) based on the newer swing RSI.
  - Conditional summary stats:
    - tail events with divergence vs without divergence: median fwd returns (1/3/5/10D),
      median forward extension percentile, and hit-rate-style counts (e.g., reversion below p95 by +10D).
- Briefing (v1): if extension percentile is in the high tail (or low tail), add a one-liner:
  - “RSI divergence: bearish (14D)” or “RSI divergence: none”.

### CLI / UX
- Reuse existing command, no new command needed initially:
  - `options-helper technicals extension-stats --symbol CVX ...`
- Add flags/config (either CLI or config file) to control:
  - `--divergence-window-days` (default 14)
  - `--min-extension-percentile` (default 95 for high, 5 for low; aligned to tail thresholds)
  - `--min-extension-days` (default 5)
  - `--min-price-delta-pct` (default 0)
  - `--min-rsi-delta` (default 0)
  - `--rsi-overbought` (default 70)
  - `--rsi-oversold` (default 30)
  - Optional: `--require-rsi-extreme` (default false). If true, only emit divergences tagged `overbought`/`oversold`.

### Implementation notes
- Keep divergence logic pure and testable (no network calls); compute from cached candles + indicator series.
- Prefer percentile-based “extended” gating so it is per-ticker normalized.
- Make output schema explicit and stable (add `schema_version` bump for extension-stats artifacts).

### Testing
- Unit tests with synthetic Close/RSI series:
  - detects bearish divergence correctly,
  - does not false-positive on flat/noisy series,
  - handles missing RSI early-window NaNs gracefully.
- Golden tests for `extension-stats` artifact JSON keys (ensure deterministic ordering and stable schema).

### Dependencies
- F-007 extension percentiles (percentile series + tail-event framing).
- Existing RSI computation (indicator provider or `options_helper/analysis/indicators.py`), plus shared date alignment.

### Acceptance criteria
- `technicals extension-stats --symbol CVX` produces artifacts that include divergence fields and conditional summaries.
- Divergence artifacts include an RSI regime tag (`overbought`/`oversold`/`neutral`) and can be gated by RSI extremes.
- Offline tests pass; no new network dependency added.

---

## F-009 — Max-upside (High-based) + weekly context for extension tails

### Summary
Enrich `technicals extension-stats` with long-horizon, **call-friendly** follow-through metrics:
- replace short-horizon “fwd ret%” displays with **max-upside return** (MFE-style) using High vs entry Close
- add a **Max Upside** summary table over longer horizons (1w/4w/3m/6m/1y) focused on downside-extension bullish setups
- surface **weekly context** (weekly extension percentile + weekly RSI regime + weekly RSI divergence) alongside daily tail events

This project is **not financial advice**; outputs are descriptive and intended for research/decision support.

### Goals
- Answer: “When a ticker is extremely extended to the downside and RSI is oversold (optionally with bullish divergence),
  what upside tends to be reachable within common options horizons?”
- Keep the workflow **offline**, **deterministic**, and **per‑ticker normalized** (percentile-based gating).
- Make daily and weekly context easy to scan in a single report.

### Non-goals
- Options pricing/recommendations (strike/expiry selection is user-driven and separate).
- Intraday signals.
- Predictive claims.

### Definitions
- **Max upside (daily)** at horizon `H`: `max(High[t+1:t+H]) / Close[t] − 1`
- Horizons (daily bars):
  - 1w: 5
  - 4w: 20
  - 3m: 63
  - 6m: 126
  - 1y: 252
- Weekly context is computed on weekly candles (resampled) and forward-filled onto daily dates for display.

### Outputs
- `technicals extension-stats` artifacts (`.json` + `.md`):
  - Daily: add “Max Upside (Daily, High-based)” summary for low-tail setups.
  - Daily tail-event table: add RSI-at-event regime + weekly context columns; add +15D forward percentile; show “Max ret%” instead of “Fwd ret%”.
  - Weekly: add “RSI Divergence (Weekly)” section and include RSI regime in weekly tail-event table.
- JSON schema bump for extension-stats artifacts.

### Testing
- Unit tests for max-upside computation on synthetic series.
- CLI integration test for `technicals extension-stats` output shape (synthetic OHLC input).

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
