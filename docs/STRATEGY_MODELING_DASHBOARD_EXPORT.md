# Strategy Modeling Dashboard Export

Use the Streamlit Strategy Modeling page to write the same report bundle that the CLI produces, then hand those artifacts to an LLM for follow-up analysis.

Not financial advice.

## Dashboard Trade Review + Drilldown (Read-Only)

After a run, the Trade Log area renders:
- `Top 20 Best Trades (Realized R)`
- `Top 20 Worst Trades (Realized R)`
- `Full Trade Log`

Selection + drilldown behavior:
- Each table is row-selectable (`on_select="rerun"`, `selection_mode="single-row"`).
- If multiple tables have selections, drilldown selection precedence is deterministic: best -> worst -> full log.
- Drilldown loads local intraday bars for the selected trade and renders entry/stop/target/exit markers.
- Base bars prefer `1Min`, then fallback to the run timeframe (or `5Min`) when `1Min` is unavailable.
- Chart timeframe options are supported multiples of the chosen base timeframe (`1Min/5Min/15Min/30Min/60Min`).
- Large windows use a guardrail: auto-upsample timeframe to keep bars <= 5000, otherwise skip chart with warning.
- Missing selection, missing timestamps/symbol, or missing bars are warning-first paths (no crash).

## What It Exports

After a successful dashboard run, click **Export Reports**. The page writes:

- `summary.json`
- `summary.md`
- `llm_analysis_prompt.md`
- `trades.csv`
- `r_ladder.csv`
- `segments.csv`
- `top_20_best_trades.csv`
- `top_20_worst_trades.csv`

## `summary.json.trade_review` (Additive Contract)

`summary.json` now includes additive key `trade_review`:
- `metric`: ranking metric string (`realized_r`).
- `scope`:
  - `accepted_closed_trades` when `accepted_trade_ids` exists on the run result (including present-but-empty, which stays authoritative).
  - `closed_nonrejected_trades` only when `accepted_trade_ids` is missing (legacy fallback scope).
- `candidate_trade_count`: candidate closed trades in scope (finite `realized_r`, non-rejected).
- `top_best_count`, `top_worst_count`.
- `top_best` and `top_worst`: highlight rows (not full log rows) with:
  - `rank`, `trade_id`, `symbol`, `direction`, `entry_ts`, `exit_ts`, `realized_r`, `exit_reason`.

CSV contracts:
- `top_20_best_trades.csv`: up to 20 rows sorted by `realized_r desc`, `entry_ts asc`, `trade_id asc`.
- `top_20_worst_trades.csv`: up to 20 rows sorted by `realized_r asc`, `entry_ts asc`, `trade_id asc`.
- Both CSVs use `rank` followed by the same field order as `trades.csv`.

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
5. Ask focused follow-up questions against `summary.json` (including `trade_review`), `top_20_best_trades.csv`, `top_20_worst_trades.csv`, `segments.csv`, and `trades.csv` (for example: regime/symbol weaknesses, stop-policy changes, hold-time opportunities, reject-code patterns).

## Notes

- Export is explicit; the page does not write report files on load.
- Exported files are informational/decision-support artifacts only.
