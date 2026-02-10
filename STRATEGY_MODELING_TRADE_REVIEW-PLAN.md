# Strategy Modeling Trade Review + Click-Through Drilldown (Top/Bottom 20 + Candlestick)

## Summary

Implement an inline trade-review workflow on the Strategy Modeling Streamlit page that:

1. Shows `20 best` and `20 worst` trades ranked by `realized_r`.
2. Lets the user click a trade row and open an inline drilldown section.
3. Renders a candlestick chart with entry, stop, and exit markers.
4. Supports chart timeframe filtering (`1Min`, `5Min`, `15Min`, `30Min`, `60Min`) from `1Min` base candles.
5. Extends exported strategy-modeling artifacts with top/bottom trade review outputs.

## Locked Decisions

- Drilldown surface: inline section on the same page.
- Ranking scope: accepted closed trades (portfolio-selected subset), with deterministic fallback when accepted IDs are absent.
- Ranking metric: `realized_r`.
- List size: `20 best + 20 worst`.
- Artifact scope: include both dashboard and export artifact updates.
- Chart window: trade span plus configurable pre/post context bars.

## Implementation Plan

### 1) Component-layer logic (read/transform helpers)

Target file: `apps/streamlit/components/strategy_modeling_page.py`

1. Add deterministic ranking helper for trade logs.
2. Add helper to load `1Min` intraday partitions for a symbol across a timestamp window from `IntradayStore`.
3. Add helper to resample loaded `1Min` bars to selected display timeframe.
4. Add helper to normalize dataframe-selection event row extraction safely (`event.selection.rows`) and tolerate `None`/mocked events.
5. Export new helpers via `__all__` so page code stays thin.

Ranking rules (decision-complete):

- Candidate rows must satisfy `status == "closed"` and finite `realized_r`.
- If `accepted_trade_ids` exists and intersects trade IDs, keep only those IDs and `reject_code is null`.
- Else fallback to closed, non-rejected rows.
- Best ordering: `realized_r desc`, then `entry_ts asc`, then `trade_id asc`.
- Worst ordering: `realized_r asc`, then `entry_ts asc`, then `trade_id asc`.
- Keep first 20 for each side.
- Include deterministic `rank` column in each output table.

Timeframe mapping:

- `1Min -> 1min`
- `5Min -> 5min`
- `15Min -> 15min`
- `30Min -> 30min`
- `60Min -> 60min`

### 2) Strategy Modeling page UI changes

Target file: `apps/streamlit/pages/11_Strategy_Modeling.py`

1. Replace the current single trade-log presentation with:
- Existing full trade log table retained.
- New "Top 20 Best Trades (Realized R)" table.
- New "Top 20 Worst Trades (Realized R)" table.
2. Enable row click selection using `st.dataframe(..., on_select="rerun", selection_mode="single-row", key=...)`.
3. Persist selected trade ID in `st.session_state` and validate selection still exists after reruns/new runs.
4. Add "Trade Drilldown" section below rankings:
- Controls: chart timeframe selectbox and pre/post context bars input.
- Data load: fetch `1Min` bars for selected trade window (`entry_ts` to `exit_ts`, plus context).
- Resample to selected display timeframe.
- Render candlestick with marker overlays:
  - Horizontal lines for `entry_price`, `stop_price`, `exit_price` (when present).
  - Vertical lines at `entry_ts` and `exit_ts` (when present).
5. Candlestick rendering strategy:
- Use Altair layered chart when available.
- If Altair unavailable, fallback to line chart of close + explicit textual marker values.
6. Add clear user-facing warnings for:
- Missing drilldown bars for selected window.
- Invalid/missing timestamps in selected trade row.
- No accepted closed trades available for ranking.

### 3) Artifact writer extensions (export parity)

Target file: `options_helper/data/strategy_modeling_artifacts.py`

1. Extend `StrategyModelingArtifactPaths` with:
- `top_best_trades_csv` -> `top_20_best_trades.csv`
- `top_worst_trades_csv` -> `top_20_worst_trades.csv`
2. Compute ranked trade-review slices in writer using same ranking scope/metric logic as dashboard.
3. Write both CSV files when `write_csv=True`.
4. Add `trade_review` block to `summary.json` with:
- `metric: "realized_r"`
- `scope` (for example `accepted_closed_trades` or fallback value)
- `top_best` rows
- `top_worst` rows
- `top_best_count`, `top_worst_count`
5. Add a new `summary.md` section listing top/worst trade highlights (trade ID, symbol, direction, entry/exit timestamps, realized R, exit reason).
6. Update `llm_analysis_prompt.md` "Files To Read" section to include new top/worst CSV files.

### 4) Documentation updates

1. Update `docs/TECHNICAL_STRATEGY_MODELING.md`:
- New dashboard behavior for ranking + click-through drilldown.
- New artifact files and `summary.json` `trade_review` key.
2. Update `docs/STRATEGY_MODELING_DASHBOARD_EXPORT.md`:
- Include new CSV exports.
- Explain how to use top/bottom tables and trade drilldown.

## Public API / Interface Changes

- New component helpers exported from `apps/streamlit/components/strategy_modeling_page.py` for:
  - Ranking trade review rows.
  - Loading drilldown intraday bars by symbol/time window.
  - Resampling drilldown bars to selected timeframe.
- `StrategyModelingArtifactPaths` in `options_helper/data/strategy_modeling_artifacts.py` gains two new path fields.
- `summary.json` artifact contract gains additive top-level key: `trade_review`.
- Export bundle gains two additive files:
  - `top_20_best_trades.csv`
  - `top_20_worst_trades.csv`

## Tests and Scenarios

### A) Component tests

Target file: `tests/portal/test_strategy_modeling_component.py`

1. Ranking helper returns 20 best/20 worst max with deterministic order.
2. Ranking scope prefers `accepted_trade_ids` closed subset.
3. Fallback ranking path works when accepted IDs are absent.
4. Drilldown bar loader reads windowed `1Min` partitions and filters timestamps correctly.
5. Resampler produces correct OHLC aggregation for each supported timeframe.

### B) Page behavior tests

Target file: `tests/portal/test_strategy_modeling_page.py`

1. Selected row in best/worst table updates selected trade in session state.
2. Drilldown section appears after selection and attempts chart render.
3. Missing bars for selected trade shows warning rather than crash.
4. Existing run/export flows remain green with new widgets present.

### C) Artifact tests

Target file: `tests/test_strategy_modeling_artifacts.py`

1. New CSV files are created on export.
2. `summary.json` contains `trade_review` with required keys and expected row counts.
3. `summary.md` includes top/worst section.
4. `llm_analysis_prompt.md` references new CSV files.

### D) Optional performance guard

Target file: `tests/test_strategy_modeling_performance.py`

- Keep existing runtime/memory thresholds; ensure additive ranking/export logic does not regress those tests.

## Acceptance Criteria

1. Strategy Modeling page shows two ranked tables: best 20 and worst 20 by realized R.
2. User can click a ranked trade row and immediately see an inline drilldown chart.
3. Drilldown chart displays candlesticks plus entry/stop/exit markers.
4. User can change chart timeframe and chart updates from `1Min` base data.
5. Exported run directory contains `top_20_best_trades.csv`, `top_20_worst_trades.csv`, and `summary.json` `trade_review`.
6. Existing strategy-modeling tests still pass, plus new tests covering ranking/drilldown/artifacts.

## Assumptions and Defaults

- `1Min` stock-bar partitions exist for symbols in reviewed runs.
- Ranked trade review is informational only and preserves existing "not financial advice" framing.
- No new ingest/backfill writes are triggered by dashboard interactions.
- Feature is additive and backward-compatible with existing artifact consumers (new keys/files only).
