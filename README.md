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

Without installing an entrypoint script:

```bash
python3 -m options_helper init portfolio.json
```

## Portfolio format
See `PRD.md` for the full MVP spec and an example JSON structure.

## Candle cache (technical analysis)
`analyze` maintains a local daily OHLCV cache per symbol and resamples it for 3‑day and weekly indicators.

- Design/usage: `docs/CANDLE_CACHE.md`

## Disclaimer
This tool is for informational/educational use only and is not financial advice.
