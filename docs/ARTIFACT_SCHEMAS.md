# Artifact Schemas (JSON)

This repo produces **schema-versioned JSON artifacts** intended for agents and downstream tooling.
All artifacts include:

- `schema_version` — integer schema version (start at `1`).
- `generated_at` — UTC ISO-8601 timestamp.
- `as_of` — data date (YYYY-MM-DD) for the artifact.

> Not financial advice. For informational/educational use only.

## Artifacts

### Daily briefing
- Path: `data/reports/daily/{YYYY-MM-DD}.json`
- Source: `options-helper briefing ... --write-json`
- Schema: `options_helper.schemas.briefing.BriefingArtifact`
- Notes: `sections[]` contains per-symbol summaries; nested objects are best-effort.

### Chain report
- Path: `data/reports/chains/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper chain-report ... --out ...`
- Schema: `options_helper.schemas.chain_report.ChainReportArtifact`

### Compare report
- Path: `data/reports/compare/{SYMBOL}/{FROM}_to_{TO}.json`
- Source: `options-helper compare ... --out ...`
- Schema: `options_helper.schemas.compare.CompareArtifact`

### Flow report
- Path: `data/reports/flow/{SYMBOL}/{FROM}_to_{TO}_w{N}_{group_by}.json`
- Source: `options-helper flow ... --out ...`
- Schema: `options_helper.schemas.flow.FlowArtifact`

### IV surface report
- Path: `data/reports/iv_surface/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis iv-surface ... --out ...`
- Schema: `options_helper.schemas.iv_surface.IvSurfaceArtifact`

### Exposure report
- Path: `data/reports/exposure/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis exposure ... --out ...`
- Schema: `options_helper.schemas.exposure.ExposureArtifact`

### Intraday flow report
- Path: `data/reports/intraday_flow/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper intraday flow ... --out ...`
- Schema: `options_helper.schemas.intraday_flow.IntradayFlowArtifact`

### Levels report
- Path: `data/reports/levels/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: `options-helper market-analysis levels ... --out ...`
- Schema: `options_helper.schemas.levels.LevelsArtifact`

### Scenarios report
- Path: `data/reports/scenarios/{PORTFOLIO_DATE}/{POSITION_KEY}_{CONTRACT}.json`
- Source: `options-helper scenarios ... --out ...`
- Schema: `options_helper.schemas.scenarios.ScenariosArtifact`

### Scanner shortlist
- Path: `data/scanner/runs/{RUN_ID}/shortlist.json`
- Source: `options-helper scanner run ... --write-shortlist`
- Schema: `options_helper.schemas.scanner_shortlist.ScannerShortlistArtifact`

### Strategy modeling run
- Path: `data/reports/strategy_modeling/{STRATEGY}/{RUN_ID}.json` (planned writer in strategy-modeling CLI flow)
- Source: planned `options-helper technicals strategy-model ... --out ...`
- Schema: `options_helper.schemas.strategy_modeling_artifact.StrategyModelingArtifact`
- Required sections:
  - `schema_version`, `generated_at`, `run_id`, `strategy`, `symbols`
  - `policy` (locked modeling policy contract)
  - `portfolio_metrics`, `target_hit_rates`, `segment_records`
  - `equity_curve`, `trade_simulations`, `signal_events`
- Compatibility parser:
  - Use `options_helper.analysis.strategy_modeling_artifact.parse_strategy_modeling_artifact(...)`.
  - Supports schema `v1` and legacy unversioned payloads by explicit key mapping:
    - `metrics -> portfolio_metrics`
    - `r_ladder -> target_hit_rates`
    - `segments -> segment_records`
    - `equity -> equity_curve`
    - `trades -> trade_simulations`
    - `signals -> signal_events`
    - `policy_overrides -> policy`
    - `universe -> symbols`

### Tail risk
- Path: `data/reports/tail_risk/{SYMBOL}/tail_risk_{ASOF}_h{H}_n{N}_seed{SEED}.json` (when `--out` is used)
- Source: `options-helper market-analysis tail-risk ... --out ...`
- Schema: `options_helper.schemas.tail_risk.TailRiskArtifact`

### SPXW 0DTE put study (SPY proxy)
- Path: `data/reports/zero_dte_put_study/{SYMBOL}/{YYYY-MM-DD}.json`
- Source: planned `options-helper market-analysis zero-dte-put-study ... --out ...`
- Schema: `options_helper.schemas.zero_dte_put_study.ZeroDtePutStudyArtifact`
- Locked defaults/assumptions:
  - `assumptions.proxy_underlying_symbol="SPY"`
  - `assumptions.benchmark_decision_mode="fixed_time"`
  - `assumptions.benchmark_fixed_time_et="10:30"`
  - `assumptions.fill_model="bid"`
  - `assumptions.risk_tier_breach_probabilities=[0.005,0.01,0.02,0.05]`
  - `assumptions.exit_modes=["hold_to_close","adaptive_exit"]`
- Required row contracts:
  - `probability_rows[]`: conditional close-tail probability rows keyed by anchor metadata + risk tier + strike return.
  - `strike_ladder_rows[]`: ranked strike candidates with breach probability and premium estimate.
  - `simulation_rows[]`: per-exit-mode trade outcome rows (`hold_to_close`, `adaptive_exit`).
- Anti-lookahead contract:
  - Anchor metadata is explicit via `anchor.{decision_ts,decision_bar_completed_ts,entry_anchor_ts,close_label_ts}`.
  - If `entry_anchor_ts` is missing, `skip_reason` must be `no_entry_anchor`.
- Disclaimer metadata:
  - `disclaimer.not_financial_advice`
  - `disclaimer.informational_use_only`
  - `disclaimer.spy_proxy_caveat`
  - `disclaimer.lookahead_notice`

### SPXW 0DTE model-state + registry contracts
- Recommended root: `data/reports/zero_dte_put_study/{SYMBOL}/`
- Source: `options_helper.data.zero_dte_artifacts.ZeroDteArtifactStore`
- Schemas:
  - `model_states/{MODEL_VERSION}.json`: `ZeroDteModelSnapshotArtifact`
  - `model_registry.json`: `ZeroDteModelRegistryArtifact`
- Required frozen-state metadata:
  - `model_version`
  - `trained_through_session`
  - `assumptions_hash`
  - `snapshot_hash`
  - `compatibility.{artifact_schema_version,feature_contract_version,policy_contract_version}`
- Registry metadata:
  - `active_model_version`
  - `previous_active_model_version` (rollback pointer)
  - `entries[]` with per-model compatibility + snapshot reference
  - `promotion_history[]` with `action in {promote,rollback}`, `from_model_version`, `to_model_version`, `promoted_at`
- Active model resolution is deterministic and fail-closed:
  - Requires one explicit active model in registry.
  - Rejects missing snapshot files.
  - Rejects stale snapshots (`trained_through_session` before required minimum, or not strictly before scoring session).
  - Rejects compatibility/assumption-hash mismatches.

### SPXW 0DTE table artifacts + upsert keys
- Paths (under the same root):
  - `probability_curves.json`: `ZeroDteProbabilityCurveArtifact`
  - `strike_ladders.json`: `ZeroDteStrikeLadderArtifact`
  - `calibration_tables.json`: `ZeroDteCalibrationArtifact`
  - `backtest_summaries.json`: `ZeroDteBacktestSummaryArtifact`
  - `forward_snapshots.json`: `ZeroDteForwardSnapshotArtifact`
  - `trade_ledgers.json`: `ZeroDteTradeLedgerArtifact`
- Canonical forward upsert key (base key, required):
  - `symbol`
  - `session_date`
  - `decision_ts`
  - `risk_tier`
  - `model_version`
  - `assumptions_hash`
- Row-level idempotent upsert extension:
  - Probability curves: base key + `strike_return`
  - Strike ladders: base key + `ladder_rank`
  - Forward snapshots: base key (single row per forward decision/risk tier)
  - Trade ledgers: base key + `exit_mode` + `contract_symbol` + `entry_ts`
  - Calibration tables: `model_version` + `assumptions_hash` + `risk_tier` + `probability_bin`
  - Backtest summaries: `model_version` + `assumptions_hash` + `session_date` + `risk_tier` + `exit_mode`

## Validation

- Offline fixtures live under `tests/fixtures/artifacts/`.
- Tests validate each fixture against its schema in `tests/test_artifact_fixtures.py`.

### Runtime validation (`--strict`)

Most commands that write JSON accept `--strict` to validate the artifact against its schema
before writing. This is useful when you want failures to surface immediately during a run.
