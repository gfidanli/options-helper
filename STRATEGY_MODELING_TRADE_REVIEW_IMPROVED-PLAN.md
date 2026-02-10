# Plan: Strategy Modeling Trade Review + Click-Through Drilldown (Streamlit + Artifacts)

**Generated**: 2026-02-10
**Target plan file**: `STRATEGY_MODELING_TRADE_REVIEW_IMPROVED-PLAN.md`

## Overview

Add an inline trade-review workflow to the Strategy Modeling Streamlit page plus export parity in the strategy-modeling artifact bundle:

- Show **Top 20 Best** and **Top 20 Worst** closed trades ranked by `realized_r`.
- Allow **row click selection** (from either the full trade log or the top/bottom tables).
- Render an inline **trade drilldown** with a candlestick chart and entry/stop/target/exit markers.
- Load drilldown bars from `IntradayStore` and support chart timeframes `1Min/5Min/15Min/30Min/60Min`.
- Extend exports with `top_20_best_trades.csv`, `top_20_worst_trades.csv`, plus additive `summary.json.trade_review`.

Everything remains read-only and clearly not financial advice.

## Locked Decisions

- Drilldown surface: inline section on `apps/streamlit/pages/11_Strategy_Modeling.py`.
- Ranking metric: `realized_r`.
- Ranking scope (authoritative when available): portfolio accepted closed trades; fallback only when accepted IDs are missing, not when they are present-but-empty.
- List size: `20 best + 20 worst`.
- Selection sources: all tables (full trade log + top/bottom tables).
- Context inputs: bars of the currently selected chart timeframe.
- Drilldown base bars: prefer `1Min`, but fallback to the run’s `intraday_timeframe` (or `5Min`) when `1Min` partitions are missing; chart timeframe options are restricted to multiples of the chosen base timeframe.

## Public API / Interface Changes

### Artifacts (export bundle)

- `options_helper/data/strategy_modeling_artifacts.py`
  - `StrategyModelingArtifactPaths` gains:
    - `top_best_trades_csv` -> `top_20_best_trades.csv`
    - `top_worst_trades_csv` -> `top_20_worst_trades.csv`
  - `summary.json` gains additive key: `trade_review`
  - `llm_analysis_prompt.md` "Files To Read" includes the new CSVs

### No changes to CLI flags

- The CLI continues to call `write_strategy_modeling_artifacts`; it now emits additional files/keys.

## Data / Contract Details (Decision-Complete)

### Ranking inputs

- Input rows come from `run_result.trade_simulations` (Streamlit) and from the normalized `trade_rows` already built in the artifact writer (exports).
- Candidate rows must satisfy:
  - `status` (case-insensitive) == `"closed"`
  - `realized_r` is finite (`not None`, not NaN/inf)
  - `reject_code` is null/empty

### Ranking scope resolution

- If `accepted_trade_ids` is present (attribute exists on result payload), scope is authoritative accepted subset:
  - Keep only rows where `trade_id in accepted_trade_ids`
  - If `accepted_trade_ids` is empty -> top/bottom outputs are empty and scope explains why
- If `accepted_trade_ids` is missing (older payloads/stubs), fallback scope:
  - Use all closed, non-rejected rows

### Ordering (deterministic tie-breaks)

- Best: `realized_r desc`, then `entry_ts asc`, then `trade_id asc`
- Worst: `realized_r asc`, then `entry_ts asc`, then `trade_id asc`
- Add `rank` = 1..N in each output

### Exported CSV schema

- `top_20_best_trades.csv` and `top_20_worst_trades.csv` use field order:
  - `rank`, then the same field order as `trades.csv` (`_TRADE_LOG_FIELDS`)
  - This keeps the top/bottom CSVs subset-compatible with the main trade log.

### `summary.json.trade_review` (additive)

```json
"trade_review": {
  "metric": "realized_r",
  "scope": "accepted_closed_trades" | "closed_nonrejected_trades",
  "candidate_trade_count": <int>,
  "top_best_count": <int>,
  "top_worst_count": <int>,
  "top_best": [ { "rank": 1, "trade_id": "...", "symbol": "...", "direction": "...", "entry_ts": "...", "exit_ts": "...", "realized_r": 1.23, "exit_reason": "..." }, ... ],
  "top_worst": [ ... ]
}
```

- The `top_best/top_worst` objects are highlights, not full duplicate trade rows.

## Dependency Graph

```text
T1 ─┬─> T3 ─┐
    │       ├─> T5 ─> T6
T2 ─┘       │
            └─> T4 ─┘
```

## Tasks

### T1: Shared trade-review ranking helper (pure, reusable)

- **id**: T1
- **depends_on**: []
- **location**:
  - `options_helper/analysis/strategy_modeling_trade_review.py` (new)
  - `tests/test_strategy_modeling_trade_review.py` (new)
- **description**:
  - Implement a pure helper returning one structured result so UI + artifacts don’t diverge:
    - `rank_trades_for_review(trade_rows, *, accepted_trade_ids: Sequence[str] | None, top_n=20, metric="realized_r") -> TradeReviewResult`
    - `TradeReviewResult` includes: `metric`, `scope`, `candidate_trade_count`, `top_best_rows`, `top_worst_rows`
  - Robust coercion rules:
    - `status` case-insensitive; missing/unknown => excluded
    - `realized_r` supports float/int/str/NA; exclude non-finite
    - `entry_ts` supports `datetime`/`Timestamp`/ISO string; parse to UTC for sorting; missing => sort last
- **validation**:
  - `./.venv/bin/python -m pytest -k trade_review`

### T2: Streamlit drilldown data helpers (intraday window + resample + selection parsing)

- **id**: T2
- **depends_on**: []
- **location**:
  - `apps/streamlit/components/strategy_modeling_trade_drilldown.py` (new)
  - `tests/portal/test_strategy_modeling_component.py` (extend)
- **description**:
  - Add helpers (component-level; best-effort, read-only):
    - `extract_selected_rows(event) -> list[int]` tolerant of `None` / mocked return values
    - `selected_trade_id_from_event(event, displayed_df) -> str | None` with bounds checks
    - `load_intraday_window(store_root, symbol, timeframe, start_ts, end_ts) -> pd.DataFrame`
      - Uses `IntradayStore(root_dir).load_partition("stocks","bars", timeframe, symbol, day)`
      - Loads across `start_ts.date()..end_ts.date()` inclusive
      - Normalizes timestamp/ts to UTC `timestamp`, coerces OHLC numeric columns, sorts stable
      - Missing/empty partitions are skipped (best-effort)
    - `resample_ohlc(df, freq) -> pd.DataFrame`
      - Explicit resample alignment: `label="left"`, `closed="left"`, timestamps in UTC
      - OHLC: open=first, high=max, low=min, close=last; volume/trade_count=sum
      - VWAP: weighted by volume; handle 0 volume safely
    - `supported_chart_timeframes(base_tf) -> list[str]` (multiples of base)
  - Add a performance guardrail helper:
    - If resampled bars > 5,000, auto-upsample chart timeframe to the smallest supported timeframe yielding <= 5,000 bars (else warn + skip chart).
- **validation**:
  - `./.venv/bin/python -m pytest tests/portal/test_strategy_modeling_component.py`

### T3: Artifact writer extensions (export parity)

- **id**: T3
- **depends_on**: [T1]
- **location**:
  - `options_helper/data/strategy_modeling_artifacts.py`
  - `tests/test_strategy_modeling_artifacts.py`
  - `tests/test_strategy_modeling_cli.py` (if needed)
- **description**:
  - Extend `StrategyModelingArtifactPaths` + `build_strategy_modeling_artifact_paths(...)` to include the two new CSV paths.
  - Compute trade-review slices using T1’s helper:
    - Pass `accepted_trade_ids=None` only when attribute is missing on `run_result`
    - Otherwise pass the (possibly empty) accepted IDs as authoritative
  - When `write_csv=True`, write:
    - `top_20_best_trades.csv`
    - `top_20_worst_trades.csv`
  - Add `trade_review` block to `summary.json`.
  - Update `summary.md` with a Trade Review section (scope + counts + short highlights).
  - Update `llm_analysis_prompt.md` "Files To Read" to include the two new CSVs.
- **validation**:
  - `./.venv/bin/python -m pytest -k strategy_modeling_artifacts`

### T4: Streamlit trade-review table helpers (ranking wrapper for DataFrames)

- **id**: T4
- **depends_on**: [T1]
- **location**:
  - `apps/streamlit/components/strategy_modeling_trade_review.py` (new)
  - `tests/portal/test_strategy_modeling_component.py` (extend)
- **description**:
  - Add a thin wrapper around T1:
    - Accept `trade_df: pd.DataFrame` and `accepted_trade_ids`
    - Return `best_df`, `worst_df`, `scope_label`
    - Ensure displayed frames are `reset_index(drop=True)` before passing to `st.dataframe` so `event.selection.rows` maps correctly via `.iloc`.
- **validation**:
  - `./.venv/bin/python -m pytest tests/portal/test_strategy_modeling_component.py`

### T5: Strategy Modeling page UI updates (top/bottom tables + drilldown chart)

- **id**: T5
- **depends_on**: [T2, T4, T3]
- **location**:
  - `apps/streamlit/pages/11_Strategy_Modeling.py`
  - `tests/portal/test_strategy_modeling_page.py`
- **description**:
  - UI additions:
    - Keep Trade Log section and add:
      - Top 20 Best Trades (Realized R)
      - Top 20 Worst Trades (Realized R)
    - Enable selection on all 3 tables:
      - `st.dataframe(..., on_select="rerun", selection_mode="single-row", key=...)`
      - Always tolerate `st.dataframe` returning `None`
      - Deterministic precedence if multiple tables have selection (best -> worst -> full log)
    - Add Trade Drilldown section:
      - Controls: chart timeframe selectbox (from `supported_chart_timeframes(base_tf)`), pre/post context bars
      - Base bars: try `1Min`; if empty, fallback to run intraday timeframe (or `5Min`)
      - Render:
        - Altair candlestick + marker overlays when Altair is available
        - Fallback `st.line_chart` of close + textual markers when Altair is unavailable
      - Warn (not crash) for missing selection, missing timestamps, or missing bars
  - Export button UX sync:
    - Update export `help=` text to include the two new CSV outputs.
- **validation**:
  - `./.venv/bin/python -m pytest -k strategy_modeling_page`

### T6: Documentation updates

- **id**: T6
- **depends_on**: [T3, T5]
- **location**:
  - `docs/TECHNICAL_STRATEGY_MODELING.md`
  - `docs/STRATEGY_MODELING_DASHBOARD_EXPORT.md`
- **description**:
  - Document:
    - Dashboard top/bottom tables + drilldown behavior (read-only; not financial advice)
    - New export files
    - `summary.json.trade_review` contract (additive)
- **validation**:
  - Optional: `mkdocs build`

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2 | Immediately |
| 2 | T3, T4 | T1 complete |
| 3 | T5 | T2, T4, T3 complete |
| 4 | T6 | T3 and T5 complete |

## Testing Strategy

- Unit/pure logic: `./.venv/bin/python -m pytest -k trade_review`
- Artifacts: `./.venv/bin/python -m pytest -k strategy_modeling_artifacts`
- Portal:
  - `./.venv/bin/python -m pytest tests/portal/test_strategy_modeling_component.py`
  - `./.venv/bin/python -m pytest -k strategy_modeling_page`

## Risks & Mitigations

- Streamlit selection API differences / tests monkeypatch `st.dataframe`: always guard selection extraction; never assume `.selection` exists.
- Missing intraday partitions: best-effort load across days; warn and skip chart rather than crash.
- Large windows cause slow charts: cap rendered bars and auto-upsample chart timeframe.
- Accepted-scope ambiguity: treat present-but-empty accepted IDs as authoritative (empty outputs + explicit scope label), and only fallback when IDs are truly missing.

## Assumptions / Defaults

- Intraday bars are stored under `data/intraday/stocks/bars/<timeframe>/<SYMBOL>/<YYYY-MM-DD>.csv.gz`.
- All outputs are informational decision support only; not financial advice.
- No ingestion/backfill writes are triggered by portal rendering or selection.

## Task Status Updates

### T2 ✅ Complete (2026-02-10)

- Work log:
  - Added new component helper module for drilldown selection parsing, intraday-window loading, OHLC resampling, timeframe support derivation, and chart-bar guardrails.
  - Added deterministic portal tests for selection parsing, trade-id extraction bounds, UTC intraday normalization/windowing, resample semantics, supported timeframe filtering, and guardrail auto-upsample/skip behavior.
  - Validated with the required pytest target.
- Files changed:
  - `apps/streamlit/components/strategy_modeling_trade_drilldown.py` (new)
  - `tests/portal/test_strategy_modeling_component.py`
  - `STRATEGY_MODELING_TRADE_REVIEW_IMPROVED-PLAN.md`
- Gotchas / errors:
  - During test validation, resample VWAP fallback initially mutated the close column through a shared numpy view. Fixed by copying the close array before VWAP assignment.

### T4 ✅ Complete (2026-02-10)

- Work log:
  - Added a thin Streamlit component wrapper that delegates trade-review ranking to `rank_trades_for_review(...)` and returns `best_df`, `worst_df`, and a user-facing scope label.
  - Ensured returned DataFrames are display-ready by enforcing deterministic column order (`rank` first, source trade columns next, ranked extras last) and `reset_index(drop=True)` for Streamlit selection/`iloc` row mapping.
  - Extended portal component tests to validate delegation inputs, scope label mapping, deterministic trade ordering, and index reset behavior for both populated and empty accepted-scope outputs.
- Files changed:
  - `apps/streamlit/components/strategy_modeling_trade_review.py` (new)
  - `tests/portal/test_strategy_modeling_component.py`
  - `STRATEGY_MODELING_TRADE_REVIEW_IMPROVED-PLAN.md`
- Gotchas / errors:
  - None.

## Task Completion Log

### T1 (Completed 2026-02-10)

- **status**: Complete
- **work log**:
  - Added pure reusable helper `rank_trades_for_review(...)` with deterministic ordering, authoritative accepted-scope handling, robust status/`realized_r` coercion, UTC `entry_ts` parsing, and rank assignment.
  - Added deterministic unit coverage for fallback vs accepted scope, malformed/mixed input filtering, tie-break ordering, and rank output.
- **files changed**:
  - `options_helper/analysis/strategy_modeling_trade_review.py` (new)
  - `tests/test_strategy_modeling_trade_review.py` (new)
  - `STRATEGY_MODELING_TRADE_REVIEW_IMPROVED-PLAN.md` (updated)
- **gotchas/errors**:
  - None beyond expected coercion edge handling (invalid timestamps and non-finite numeric values are safely excluded).
