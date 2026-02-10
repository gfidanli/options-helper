# Strategy Modeling Dashboard Export

Use the Streamlit Strategy Modeling page to write the same report bundle that the CLI produces, then hand those artifacts to an LLM for follow-up analysis.

Not financial advice.

## What It Exports

After a successful dashboard run, click **Export Reports**. The page writes:

- `summary.json`
- `summary.md`
- `llm_analysis_prompt.md`
- `trades.csv`
- `r_ladder.csv`
- `segments.csv`

## Default Export Settings

- Export reports dir: `data/reports/technicals/strategy_modeling`
- Export output timezone: `America/Chicago`
- Folder layout: `<export_dir>/<strategy>/<as_of>/...`
  - Example: `data/reports/technicals/strategy_modeling/sfp/2026-01-31/summary.json`

`as_of` is resolved the same way as CLI artifacts:
- `run_result.as_of` when present,
- otherwise request `end_date`,
- otherwise current date.

## LLM Analysis Workflow

1. Run Strategy Modeling in the dashboard.
2. Click **Export Reports**.
3. Point your LLM workflow at the exported run folder.
4. Start with `llm_analysis_prompt.md` to bootstrap the analysis prompt and expected output format.
5. Ask focused follow-up questions against `summary.json`, `segments.csv`, and `trades.csv` (for example: regime/symbol weaknesses, stop-policy changes, hold-time opportunities, reject-code patterns).

## Notes

- Export is explicit; the page does not write report files on load.
- Exported files are informational/decision-support artifacts only.
