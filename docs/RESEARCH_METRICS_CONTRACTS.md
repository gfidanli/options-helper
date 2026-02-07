# Research Metrics MVP Contracts (T1)

This document locks MVP naming and calculation conventions for:

- IV surface and IV regime context
- Exposure levels ("dealer-style" heuristic levels)
- Intraday options flow (quotes + trades)
- Underlying levels (anchored VWAP, volume profile, gap/RS/beta)
- Position scenario grids

This tooling is for informational/educational use only and is not financial advice.

## Global Conventions

- Time:
  - `as_of` is a market data date in `YYYY-MM-DD`.
  - `generated_at` is UTC ISO-8601 timestamp.
  - Intraday timestamps are UTC and partitioned by market date.
- Naming:
  - All stable field names are `snake_case`.
  - Percent values use `_pct` (ratio, e.g. `0.025` = 2.5%).
  - Percentage-point values use `_pp` (absolute vol points, e.g. `2.5`).
- Determinism:
  - Sort inputs before grouping (`timestamp` asc for intraday, `strike` asc for ladders).
  - Resolve ties deterministically (earlier expiry first, lower strike first).
  - Missing/invalid inputs do not raise when avoidable; output `None` and add warnings.
- Contract shape:
  - Artifact rows use a stable ordered field set from
    `options_helper.schemas.research_metrics_contracts`.
  - Unknown/unsupported values remain explicit (`None`, `unknown`) rather than inferred.

## Locked Decisions

### Signed Exposure Convention

- Convention id: `calls_positive_puts_negative`
- Formula basis:
  - `call_gex = bs_gamma * open_interest * spot^2 * 0.01 * 100`
  - `put_gex = bs_gamma * open_interest * spot^2 * 0.01 * 100`
  - `net_gex = call_gex - put_gex`
- Interpretation:
  - Calls contribute positive signed gamma exposure.
  - Puts contribute negative signed gamma exposure through subtraction in `net_gex`.

### IV Surface Tenors (MVP)

Target DTE set (stable order):

- `7, 14, 30, 60, 90, 180`

Tenor selection:

- For each target DTE, pick the expiry with minimum `abs(dte - target_dte)`.
- Tie-breaker: earlier expiry date.
- If no valid expiry exists (`dte <= 0` only or empty chain), emit row with null metrics + warning.
- Output `tenor_gap_dte = dte - tenor_target_dte` so consumers can judge fit quality.

### Delta Buckets (Shared by IV + Intraday Flow)

Buckets use absolute delta (`abs(bs_delta)`):

- `d00_20`: `0.00 <= abs_delta < 0.20`
- `d20_40`: `0.20 <= abs_delta < 0.40`
- `d40_60`: `0.40 <= abs_delta < 0.60`
- `d60_80`: `0.60 <= abs_delta < 0.80`
- `d80_100`: `0.80 <= abs_delta <= 1.00`

Invalid (`NaN`, `<0`, `>1`) or missing deltas map to null bucket + warning.

## Artifact Boundaries

- Snapshot-derived metrics:
  - `iv_surface`, `exposure`
  - Inputs: chain snapshot rows (+ spot, + optional prior-day rows for change context).
- Intraday-partition metrics:
  - `intraday_flow`
  - Inputs: intraday option quotes/trades partitions (+ optional snapshot map for strikes/delta).
- Candle-derived metrics:
  - `levels`
  - Inputs: underlying daily candles, optional intraday bars, optional benchmark (SPY) candles.
- Position-derived metrics:
  - `scenarios`
  - Inputs: portfolio positions + current spot/iv/mark/time-to-expiry.

No network calls are part of these analysis contracts; loading stays in `options_helper/data/`.

## Feature Specs (Inputs, Outputs, Edge Cases)

### IV Surface + Regime

- Inputs:
  - Chain snapshot columns: `expiry`, `optionType`, `strike`, `impliedVolatility`,
    `bs_delta`, `bid`, `ask`, `lastPrice`
  - Scalars: `symbol`, `as_of`, `spot`
- Stable outputs:
  - Tenor table fields: `IV_SURFACE_TENOR_FIELDS`
  - Delta bucket table fields: `IV_SURFACE_DELTA_BUCKET_FIELDS`
- Key edge cases:
  - Missing IV/quotes: drop invalid contracts from aggregates; keep tenor row with warning.
  - Sparse deltas around 25d/10d: `skew_25d_pp` or `skew_10d_pp` can be null with warning.
  - Negative/zero mark from bad quotes: exclude from straddle/expected move.

### Exposure Levels

- Inputs:
  - Chain snapshot columns: `expiry`, `strike`, `optionType`, `openInterest`, `bs_gamma`
  - Scalars: `symbol`, `as_of`, `spot`
- Stable outputs:
  - Strike table fields: `EXPOSURE_STRIKE_FIELDS`
  - Summary fields: `EXPOSURE_SUMMARY_FIELDS`
- Key edge cases:
  - Missing OI or gamma: treat as `0` contribution and warn.
  - Non-positive spot: skip gex calculations, return null summary values + warning.
  - No sign flip in cumulative ladder: `flip_strike` is null.

### Intraday Options Flow (Trades + Quotes)

- Inputs:
  - Trades: `timestamp`, `price`, `size`, `contract_symbol`
  - Quotes: `timestamp`, `bid`, `ask`, `contract_symbol`
  - Optional enrichers: `expiry`, `strike`, `option_type`, `bs_delta`
- Stable outputs:
  - Per-contract/day fields: `INTRADAY_FLOW_CONTRACT_FIELDS`
  - Time-bucket fields: `INTRADAY_FLOW_TIME_BUCKET_FIELDS`
- Classification rules:
  - `price >= ask` -> `buy`
  - `price <= bid` -> `sell`
  - otherwise `unknown`
- Key edge cases:
  - Out-of-order ticks: sort and de-dup before `merge_asof`.
  - Missing/zero bid-ask: classify as `unknown`.
  - Missing/zero size: excluded from volume/notional with warning counters.

### Underlying Levels

- Inputs:
  - Intraday bars: `timestamp`, `open`, `high`, `low`, `close`, `volume`, optional `vwap`
  - Daily candles: `date`, `open`, `high`, `low`, `close`, `volume`
  - Optional benchmark candles for RS/beta (SPY)
- Stable outputs:
  - Summary fields: `LEVELS_SUMMARY_FIELDS`
  - Anchored VWAP rows: `LEVELS_ANCHORED_VWAP_FIELDS`
  - Volume profile rows: `LEVELS_VOLUME_PROFILE_FIELDS`
- Key edge cases:
  - `sum(volume) == 0`: return null VWAP/profile values + warning (no divide-by-zero).
  - Missing prices: skip rows/bins with NaN prices.
  - Not enough return history for beta/corr: output null values + warning.

### Scenario Grids

- Inputs:
  - Position row: `symbol`, `option_type`, `side`, `contracts`, `strike`, `expiry`,
    `basis`, optional `mark`, optional `iv`
  - Scalars: `spot`, `as_of`
- Stable outputs:
  - Summary fields: `SCENARIO_SUMMARY_FIELDS`
  - Grid fields: `SCENARIO_GRID_FIELDS`
- Grid axes (defaults):
  - Spot moves: `-0.20, -0.10, -0.05, 0.00, +0.05, +0.10, +0.20`
  - IV shifts (pp): `-10, -5, 0, +5, +10`
  - Days forward: `0, 7, 14, 30`
- Key edge cases:
  - Past expiry: empty scenario grid + warning.
  - IV <= 0 or missing IV/spot/mark: partial summary only; no invalid BS calls.
  - Time-forward beyond expiry: clamp `days_to_expiry` at `0`.
