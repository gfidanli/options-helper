# `dashboard` — Daily Briefing Dashboard (Read-Only)

`dashboard` renders a **read-only** CLI dashboard from the daily briefing JSON plus any related
offline artifacts under `data/reports/`. No network calls are made.

This tool is for informational/educational use only and is not financial advice.

## Inputs

- Daily briefing JSON: `data/reports/daily/{YYYY-MM-DD}.json`
- Report-pack artifacts (optional):
  - `data/reports/chains/{SYMBOL}/`
  - `data/reports/compare/{SYMBOL}/`
  - `data/reports/flow/{SYMBOL}/`
  - `data/reports/derived/{SYMBOL}/`
  - `data/reports/technicals/extension/{SYMBOL}/`
- Scanner shortlist (optional): `data/scanner/runs/{RUN_ID}/shortlist.json`

## Usage

Latest daily briefing:

```bash
options-helper dashboard --date latest
```

Specific date:

```bash
options-helper dashboard --date 2026-01-30
```

Specify report/scanner locations:

```bash
options-helper dashboard \
  --date 2026-01-30 \
  --reports-dir data/reports \
  --scanner-run-dir data/scanner/runs
```

Use a specific scanner run:

```bash
options-helper dashboard --date 2026-01-30 --scanner-run-id 2026-01-30_190000
```

## Output

Printed to the console (no files written):

- Portfolio positions table (from briefing JSON)
- Symbol summary (earnings, confluence, warnings)
- Watchlist sections with artifact path hints
- Scanner shortlist (when available)

## CLI flags

- `--date YYYY-MM-DD|latest`
- `--reports-dir PATH`
- `--scanner-run-dir PATH`
- `--scanner-run-id RUN_ID`
- `--max-shortlist-rows N`

## Caveats

- Requires a briefing JSON for the selected date.
- Related artifacts are **best effort**; missing files are shown as directory hints.
