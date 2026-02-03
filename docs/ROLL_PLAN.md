# `roll-plan` — Roll / Position Planner (Offline)

`roll-plan` proposes and ranks roll candidates for a **single existing portfolio position** using locally stored
snapshot files under `data/options_snapshots/`.

It is designed to make “what should I roll into?” more systematic:

- aligns candidate DTE to a thesis horizon (`--horizon-months`)
- shows roll debit/credit, delta/theta, and basic liquidity checks
- includes quote quality labels + stale indicators (best-effort)
- prints a short “why this is #1” rationale

This command is **offline-first**: it does not call Yahoo.

## Usage

```bash
options-helper roll-plan portfolio.json --id cvx-2026-06-18-190c --intent max-upside --horizon-months 12
```

Pick a specific snapshot date:

```bash
options-helper roll-plan portfolio.json --id cvx-2026-06-18-190c --as-of 2026-01-30 --horizon-months 12
```

Constrain roll cost:

```bash
options-helper roll-plan portfolio.json --id cvx-2026-06-18-190c --horizon-months 12 --max-debit 600
```

## Snapshot inputs

`roll-plan` reads:

- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/meta.json` (must include a usable spot, typically `spot`)
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` (calls+puts for each expiry)

If longer-dated expiries are missing from your snapshots, re-run `snapshot-options` with flags that include them
(e.g. `--all-expiries`, and/or increase `--window-pct`).

## CLI flags

- `--id TEXT`: position id from the portfolio JSON (required)
- `--as-of YYYY-MM-DD|latest`: snapshot date (default `latest`)
- `--cache-dir PATH`: snapshot root (default `data/options_snapshots`)
- `--intent max-upside|reduce-theta|increase-delta|de-risk`: planning intent (default `max-upside`)
- `--horizon-months INT`: thesis horizon in months (required)
- `--shape out-same-strike|out-up|out-down`: strike direction constraint (default `out-same-strike`)
- `--top N`: number of candidates to display (default `10`)
- `--max-debit FLOAT`: max roll debit in dollars (total for position size)
- `--min-credit FLOAT`: min roll credit in dollars (total for position size)
- `--min-open-interest INT`: override minimum OI liquidity gate (default from portfolio `risk_profile`)
- `--min-volume INT`: override minimum volume liquidity gate (default from portfolio `risk_profile`)
- `--include-bad-quotes`: include candidates with bad quote quality (best-effort)

## Caveats (best-effort)

- Marks use mid when bid/ask are present, otherwise fall back to last/ask/bid (same rule as other snapshot reports).
- Greeks are best-effort:
  - prefers `bs_delta` / `bs_theta_per_day` from snapshots (computed at snapshot time)
  - falls back to local Black–Scholes when enough fields exist (spot/IV/expiry/strike)
- Windowed snapshots can omit candidate strikes/expiries (common for far-dated LEAPS).
