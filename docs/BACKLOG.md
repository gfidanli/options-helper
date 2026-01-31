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
| 1 | F-001 | `chain-report` (options chain dashboard) | Productizes the most common analysis (walls, term structure, expected move, gamma concentration) into a single repeatable command. | Snapshot loader + shared pricing/filters |
| 2 | F-002 | `compare` (snapshot diff engine) | Most signal is in change (IV crush, wall shifts, OI builds). Enables “diff-first” workflow across reports. | F-001 metrics library |
| 3 | F-003 | `roll-plan` (roll/position planner) | Turns winners into planned holds/rolls aligned with horizon (“maximize upside”, “extend time”, etc.). | F-001 mark/liquidity + optional live chain |
| 4 | F-004 | `flow` upgrade (aggregations + multi-day netting) | Moves from noisy contract lists to “where is positioning building by strike/expiry (and is it persistent)?” | Existing `compute_flow` + multi-snapshot loader |
| 5 | F-005 | `briefing` (daily report generator) | Makes analysis regular: one scheduled artifact combining chain + change + flow into a daily briefing. | F-001 + F-002 + F-004 |
| 6 | F-006 | `derived` (derived-metrics time series store) | Unlocks percentiles/trends and makes “is IV high vs 60d?” easy, but depends on stable metric definitions. | F-001 metrics (optionally F-002) |

---

## Milestones (delivery plan)

### Milestone M1 — Offline chain intelligence (core)
**Goal:** make chain analysis and “what changed” a first-class, repeatable workflow.

Deliverables:
- F-001 MVP: `options-helper chain-report` from snapshots (console + JSON output).
- F-002 MVP: `options-helper compare` diffing two snapshot dates (console + JSON output).

Definition of done:
- Runs without network access and produces deterministic JSON for fixture snapshots.
- CVX-like snapshots produce: P/C ratios, key walls, expected move, ATM IV term structure, skew (where possible), and top gamma strikes.

### Milestone M2 — Portfolio decision tooling
**Goal:** make “what should I do with this position?” systematic and explainable.

Deliverables:
- F-003 MVP: `options-helper roll-plan` for a single position (from portfolio) with ranked candidates.

Definition of done:
- For a position, prints a ranked table with roll cost, new DTE, delta/theta, liquidity checks, and rationale.
- Supports `--intent max-upside` and `--horizon-months`.

### Milestone M3 — Flow power-up
**Goal:** convert daily snapshot deltas into higher-signal strike/expiry views.

Deliverables:
- F-004 MVP: `flow --window N` + `--group-by strike|expiry|expiry-strike` with premium-notional and delta-notional.

Definition of done:
- For symbols with ≥N+1 snapshots, prints “net building/unwinding zones” and persists a JSON artifact.

### Milestone M4 — Automation + history
**Goal:** make the workflow regular and measurable over time.

Deliverables:
- F-005 MVP: `briefing` generates daily Markdown for portfolio + watchlists, includes compare highlights.
- F-006 MVP: `derived update` appends derived metrics per symbol/day; `derived show` prints last N rows.

Definition of done:
- One cron-friendly command produces a daily report artifact and updates derived series.

---

## Dependency map (summary)
- F-001 is foundational and should expose a reusable “metrics from snapshot” API.
- F-002 should be implemented as “compute metrics for A and B → diff”, reusing F-001 metrics.
- F-003 should reuse the shared mark/liquidity filters and (optionally) the Greeks code already in `options_helper/analysis/greeks.py`.
- F-004 builds on existing `options_helper/analysis/flow.py` but adds aggregation + multi-day netting.
- F-005 composes F-001 + F-002 + F-004 into a single daily artifact.
- F-006 consumes F-001 outputs (and optionally F-002 deltas) to persist daily derived metrics.

---

# PRDs

## F-001 — `chain-report` (Options Chain Dashboard)

### Summary
Add `options-helper chain-report` to produce a standardized options chain dashboard from local snapshot files:
walls, P/C ratios, IV term structure, skew, expected moves, and gamma concentration.

### Problem
We can snapshot chains and compute per-contract flow, but we can’t quickly answer “what levels matter today?” and
“what is the market pricing?” without ad-hoc scripts.

### Goals
- **Offline-first** report from `data/options_snapshots` (no fetch required).
- Clear “levels” output: top OI strikes (overall + per expiry), expected move bands.
- Stable machine-readable output (JSON) + readable console/Markdown output.

### Non-goals
- Dealer positioning inference (net long/short gamma) beyond gross, model-based measures.
- Intraday tape/prints.

### Users / use-cases
- Daily chain check for watchlist symbols after snapshots run.
- Quick “what’s priced into next week / next monthly?” before planning a trade/roll.

### CLI / UX
Proposed:
```bash
options-helper chain-report --symbol CVX --as-of 2026-01-30
options-helper chain-report --symbol CVX --as-of latest --out data/reports
```

Options:
- `--cache-dir PATH` (default `data/options_snapshots`)
- `--format console|md|json` (default `console`)
- `--out PATH` (write artifacts; default none)
- `--top INT` (default 10)
- `--include-expiry YYYY-MM-DD` (repeatable)
- `--expiries near|monthly|all` (default `near`)
- `--best-effort` (don’t fail hard on missing fields; emit warnings)

### Data inputs
- Snapshot day directory: `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/`
  - `{EXPIRY}.csv` (calls+puts)
  - `meta.json` (spot, snapshot_date, etc.)

### Core computations (v1 scope)
- Mark price per contract (mid if valid bid/ask else last).
- Aggregate:
  - call/put totals: OI, volume, premium-volume notional.
  - per-expiry totals: OI and volume with P/C ratios.
- “Walls”:
  - top call OI strikes (at/above spot), top put OI strikes (at/below spot), overall + selected expiries.
- Expected move:
  - per expiry, approximate ATM straddle (nearest strike to spot with both call+put), using mark.
- IV term structure:
  - per expiry, approximate ATM IV (avg call/put IV at nearest strike).
- Skew:
  - per expiry, 25-delta put IV minus 25-delta call IV using stored BS deltas (best-effort).
- Gamma concentration (gross):
  - rank strikes by OI-weighted BS gamma, converted to a dollarized “gamma per 1% move” proxy.
- Optional “max pain”:
  - compute on strike grid per expiry (explicitly labeled heuristic).

### Outputs
- Console: rich tables + 5–10 “takeaways” bullets.
- Optional artifacts:
  - `data/reports/chains/{SYMBOL}/{YYYY-MM-DD}.md`
  - `data/reports/chains/{SYMBOL}/{YYYY-MM-DD}.json` (stable schema)

### Milestones
- **MVP:** P/C ratios, expiry totals, walls, expected move (ATM straddle), IV term structure, gamma strike ranks; JSON schema.
- **v1:** skew, max pain, configurable expiry selection, “liquidity filters” toggles.
- **v2:** simple plots (ASCII sparkline or exported CSV), richer “key takeaways” engine.

### Dependencies
- Snapshot loader (`OptionsSnapshotStore`) and shared parsing utilities.
- Shared helpers to:
  - compute mark price consistently
  - filter illiquid quotes (0 bid/ask, extreme spreads)

### Acceptance criteria
- Running on an existing snapshot day produces deterministic JSON and readable console output.
- For CVX-like snapshots, the report surfaces:
  - top call strikes near/above spot
  - top put strikes near/below spot
  - near-term expected move % for at least one expiry
  - top gamma concentration strikes

### Testing
- No-network unit tests using fixture snapshots under `tests/fixtures/`.
- Golden JSON tests (schema stability).

---

## F-002 — `compare` (Snapshot Diff Engine)

### Summary
Add `options-helper compare` to diff two snapshot dates for a symbol using the same metrics as `chain-report`.

### Problem
The biggest insights often come from **change** (IV crush, wall migration, OI builds/unwinds), but today it’s manual.

### Goals
- Provide a consistent “diff-first” summary.
- Enable composition: `chain-report --compare`, `briefing`, and flow summaries should reuse this diff output.

### Non-goals
- Forecasting or probabilistic predictions.

### CLI / UX
```bash
options-helper compare --symbol CVX --from 2026-01-29 --to 2026-01-30
options-helper compare --symbol CVX --to latest --from -1
```

Options:
- `--cache-dir PATH`
- `--top INT`
- `--out PATH` (write JSON artifact)

### Outputs (v1)
- Spot change and % change.
- Changes in:
  - P/C OI and P/C volume
  - walls (strike + OI) for key expiries and overall
  - ATM IV per expiry (pp)
  - expected move per expiry (pp and $)
  - gamma concentration top strikes
- Link to per-contract ΔOI (reuse flow computation where prior OI exists):
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

