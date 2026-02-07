# `scenarios` — Position Scenario Grids (Offline)

`scenarios` computes Black-Scholes-based scenario grids for each option position in your portfolio using local snapshot files.

This command is informational only and **not financial advice**.

## Usage

```bash
options-helper scenarios portfolio.json
```

Pick a specific snapshot date:

```bash
options-helper scenarios portfolio.json --as-of 2026-02-07
```

Write JSON artifacts:

```bash
options-helper scenarios portfolio.json --out data/reports
```

## Inputs

`scenarios` reads:

- `portfolio.json`
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/meta.json` (spot, best-effort)
- `data/options_snapshots/{SYMBOL}/{YYYY-MM-DD}/{EXPIRY}.csv` (contract rows, IV/marks)

By default, it resolves `--as-of latest` per symbol from available snapshot folders.

## Console output

The command prints:

- A compact per-position summary table:
  - mark, IV, best/worst scenario PnL, row count, warnings
- A compact spot-shock slice (`iv=0pp`, `t+0d`) for each position with a non-empty grid

Missing data does not hard-fail the run. Instead, warnings are surfaced in console output and artifacts.

Common warning tokens:

- `missing_snapshot_date`
- `missing_snapshot_day`
- `missing_snapshot_row`
- `missing_spot`
- `missing_iv`
- `missing_mark`
- `past_expiry`

## Artifact output

When `--out` is provided, files are written under:

- `{out}/scenarios/{PORTFOLIO_DATE}/`

`PORTFOLIO_DATE` is derived from the resolved scenario `as_of` date(s).

Each JSON file validates against:

- `options_helper.schemas.scenarios.ScenariosArtifact`

Use `--strict` to validate artifacts before write.
