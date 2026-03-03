# Plan: Regime-Tactic CLI + EMA Stop-Trail Rules

**Generated**: 2026-03-03

## Overview
Implement two user-facing improvements:

1) A **regime-aware tactic recommendation** that maps:
- **market regime** (trend vs choppy) and
- **symbol regime** (trend vs sideways)
to a suggested long-entry tactic:
- **breakout / momentum** (aligns with existing MSB-style logic), or
- **undercut/reclaim** (aligns with existing SFP-style logic),
plus the recommended support model (**EMA-dynamic vs static levels**).

2) A **first-class stop-trailing policy** for strategy modeling that can switch trailing stop anchors (e.g. **EMA21 → EMA9**) after a configurable profit threshold (e.g. after **+1.0R**).

This repo is decision-support only; all output and docs must clearly state **not financial advice**.

## Scope / Non-Goals
- **In scope**:
  - New `technicals regime-tactic` CLI command with deterministic offline tests (`--ohlc-path`, `--market-ohlc-path`).
  - New pure analysis helpers for regime classification and tactic mapping.
  - Strategy-modeling policy + simulator support for staged EMA trailing stop rules (including artifact metadata).
- **Out of scope** (for now):
  - Auto “regime-switching backtest” that dynamically changes strategy mid-run.
  - Scanner integration into existing SFP/MSB scan commands (the new CLI will be separate).
  - Any live trading/execution guidance.

## Prerequisites
- Local dev env created and editable install (`pip install -e ".[dev]"`).
- Tests run locally via `./.venv/bin/python -m pytest`.
- No network calls in tests (use `--ohlc-path` fixtures).

## Dependency Graph

```
T1 ──┬── T3 ──┬── T4 ──┬── T11
     │        │        │
     │        │        └── T12
     │        │
     │        └────────────┐
     │                     │
T2 ──┴─────────────────────┘

T6 ──┬── T7 ──┬── T9 ──┬── T13
     │        │        │
     │        │        └── T14
     │        │
     │        └── T10 ─────┘
     │
T5 ──┴────────────────────────
```

Legend:
- T1/T2: pure analysis blocks (regime + mapping)
- T3/T4: CLI surface + wiring
- T5: docs + disclaimers
- T6/T7: schema and CLI options for stop trails
- T9/T10: simulator + strategy-modeling wiring
- T11/T12/T13/T14: tests

## Tasks

### T1: Regime classifier (pure analysis)
- **depends_on**: []
- **location**:
  - `options_helper/analysis/price_regime.py` (new)
  - `tests/test_price_regime.py` (new)
- **description**:
  - Add a deterministic classifier: `classify_price_regime(...) -> PriceRegimeTag`.
  - Inputs: daily OHLC + optional precomputed indicator series.
  - Outputs: tag in `{trend_up, trend_down, choppy, mixed, insufficient_data}` plus a compact diagnostics dict.
  - Use robust heuristics:
    - EMA9/EMA21 slope direction + spacing,
    - CHOP(14) thresholds (e.g. trend ≤45, choppy ≥60),
    - EMA21 cross-count window (e.g. 20 sessions).
  - NaN-safe; no I/O; no imports from `options_helper/data/*`.
  - Include guardrails: minimum history length (e.g. ≥60 bars) or `insufficient_data`.
  - Either reuse an existing CHOP implementation if present, or add a small NaN-safe CHOP helper in the same module (explicitly tested).
- **validation**:
  - Unit tests cover at least: obvious uptrend, obvious chop, insufficient history, NaN segments.
  - `pytest -k price_regime`.
- **status**: Completed
- **log**:
- Added `classify_price_regime(...)` + NaN-safe CHOP/EMA/cross diagnostics with insufficient-history guardrail.
  - Added deterministic unit coverage for trend/chop/NaN/insufficient-history cases.
  - Commit: `426e4f3`.
- **files edited/created**:
- `options_helper/analysis/price_regime.py`
  - `tests/test_price_regime.py`

### T2: Regime → tactic mapping (pure analysis)
- **depends_on**: []
- **location**:
  - `options_helper/analysis/regime_tactic.py` (new)
  - `tests/test_regime_tactic.py` (new)
- **description**:
  - Add a pure function mapping `(market_regime, symbol_regime, direction=long)` to:
    - `tactic`: `breakout` | `undercut_reclaim` | `avoid`
    - `support_model`: `ema` | `static`
    - `rationale`: short list of strings
  - Ensure mapping is stable/deterministic, configurable thresholds documented in module constants.
- **validation**:
  - Unit tests for key combinations (trend/trend, choppy/sideways, mixed cases).
  - `pytest -k regime_tactic`.
- **status**: Completed
- **log**:
- Added deterministic regime-to-tactic mapping with typed recommendation output and normalized aliases.
  - Added mapping tests across trend/chop/mixed/short-direction scenarios.
  - Commit: `1c17c14`.
- **files edited/created**:
- `options_helper/analysis/regime_tactic.py`
  - `tests/test_regime_tactic.py`

### T3: CLI command skeleton + deps seam wiring
- **depends_on**: [T1, T2]
- **location**:
  - `options_helper/commands/technicals/regime_tactic.py` (new)
  - `options_helper/commands/technicals/__init__.py` (if needed)
  - `options_helper/cli.py`
  - `options_helper/cli_deps.py` (if a new service seam is needed)
- **description**:
  - Add `technicals regime-tactic` command (Typer) that:
    - loads symbol OHLC (from `--ohlc-path` or cache),
    - loads market proxy OHLC (from `--market-ohlc-path` or `--market-symbol`),
    - classifies both regimes,
    - computes a tactic recommendation and prints + optionally writes an artifact.
  - Keep `cli.py` thin; command implementation under `commands/`.
  - Ensure it can run fully offline when both OHLC paths are provided.
- **validation**:
  - `./.venv/bin/options-helper technicals regime-tactic --help` works.
  - Command runs against test fixtures without network.
- **status**: Completed
- **log**:
- Added `technicals regime-tactic` command wiring under technicals module and offline path execution support.
  - Added command help/smoke coverage for registration and baseline behavior.
  - Commit: `360aa72`.
- **files edited/created**:
- `options_helper/commands/technicals/regime_tactic.py`
  - `options_helper/commands/technicals/__init__.py`
  - `tests/test_regime_tactic_cli.py`

### T4: Artifact output + reporting format
- **depends_on**: [T3]
- **location**:
  - `options_helper/data/reports/...` writer module OR reuse existing report writer conventions
- **description**:
  - Produce a JSON artifact with:
    - `schema_version`
    - `asof_date`
    - `symbol`, `market_symbol`
    - regime tags + diagnostics
    - tactic + rationale
    - disclaimer field
  - Optional: a Markdown summary mirroring JSON keys for easy reading.
  - Follow existing scans’ artifact patterns where possible (dict-based output is acceptable if consistent and tested).
  - Define `asof_date` as the **latest candle date present in the input OHLC**, not wall-clock run time.
- **validation**:
  - Manually run the CLI once against fixtures to ensure artifact files are created where expected.
- **status**: Completed
- **log**:
- Added JSON artifact writer with schema/disclaimer/regime/recommendation payload and `asof_date` from latest OHLC candle.
  - Added CLI output controls (`--out`, `--write-json/--no-write-json`) and fixture-verified artifact creation.
  - Commit: `97aef41`.
- **files edited/created**:
- `options_helper/commands/technicals/regime_tactic.py`
  - `options_helper/data/regime_tactic_artifacts.py`
  - `tests/test_regime_tactic_cli.py`

### T5: Documentation + disclaimers
- **depends_on**: []
- **location**:
  - `docs/technicals_regime_tactic.md` (new)
  - `docs/strategy_modeling_stop_trails.md` (new)
  - `mkdocs.yml` (nav wiring)
- **description**:
  - Document:
    - how regime tags are computed (at a high level),
    - how mapping chooses tactics,
    - how to run the CLI offline using `--ohlc-path` and `--market-ohlc-path`,
    - disclaimer: **not financial advice**; backtests are illustrative; no guarantee.
  - Document stop-trail rules syntax, lookahead-bias guardrails, and limitations.
- **validation**:
  - Docs build (if present) or at minimum `rg` shows docs are referenced in nav (if used).
- **status**: Completed
- **log**:
- Added docs for regime-tactic flow and stop-trail policy behavior, including offline usage and explicit non-advice disclaimers.
  - Wired both docs into MkDocs navigation under Research & Analysis.
  - Commit: `1f9d4dc`.
- **files edited/created**:
- `docs/technicals_regime_tactic.md`
  - `docs/strategy_modeling_stop_trails.md`
  - `mkdocs.yml`

### T6: Stop-trail policy schema extensions
- **depends_on**: []
- **location**:
  - `options_helper/schemas/strategy_modeling_policy.py`
- **description**:
  - Add `StopTrailRule` (start threshold in R, EMA span, optional ATR buffer multiple).
  - Add `stop_trail_rules: list[StopTrailRule]` to policy config with validation:
    - `start_r >= 0`, `ema_span` in a small allowed set (e.g. 9/21/50/200), `buffer_atr_multiple >= 0`.
    - deterministic parsing order: sort by `start_r` ascending.
  - Document precedence with existing stop logic:
    - Treat both stop-move rules and stop-trail rules as proposing a new stop.
    - Use a **tighten-only reducer**: for longs take `max(current_stop, candidates...)`; for shorts take `min(...)`.
- **validation**:
  - Unit-level schema tests (or extend existing) added in T13/T14.
- **status**: Completed
- **log**:
- Added `StopTrailRule` schema and `stop_trail_rules` policy field with validation/sorting/duplicate detection.
  - Documented tighten-only precedence semantics with stop-move rules.
  - Added schema tests for valid/invalid rule sets.
  - Commit: `dee1f8b`.
- **files edited/created**:
- `options_helper/schemas/strategy_modeling_policy.py`
  - `tests/test_strategy_modeling_policy.py`

### T7: CLI profile / option plumbing for stop trails
- **depends_on**: [T6]
- **location**:
  - `options_helper/schemas/strategy_modeling_profile.py`
  - `options_helper/commands/technicals/strategy_model.py` (and/or related command files)
- **description**:
  - Add CLI flags:
    - `--stop-trail` (repeatable): `start_r:ema_span[:buffer_atr_multiple]`
    - `--disable-stop-trails`
  - Respect “parameter source” semantics (CLI overrides profile; profile only fills unspecified).
  - Ensure Typer parsing avoids `datetime.date` annotations (repo guardrail).
- **validation**:
  - CLI help includes new options.
  - A small deterministic test sets stop trails via profile and via CLI and asserts precedence.
- **status**: Completed
- **log**:
- Added strategy-model CLI flags `--stop-trail` (repeatable) and `--disable-stop-trails` with deterministic parser errors.
  - Added profile + parameter-source precedence wiring so CLI overrides profile and disable clears effective rules.
  - Added CLI/profile tests for help output and precedence behavior.
  - Commit: `e4576e4`.
- **files edited/created**:
- `options_helper/commands/technicals/strategy_model.py`
  - `options_helper/commands/technicals/strategy_model_runtime.py`
  - `options_helper/commands/technicals/strategy_model_runtime_profile_values.py`
  - `options_helper/commands/technicals/strategy_model_helpers_legacy.py`
  - `options_helper/schemas/strategy_modeling_profile.py`
  - `tests/test_strategy_modeling_cli.py`

### T8: Simulator support for staged EMA trailing stops
- **depends_on**: [T6]
- **location**:
  - `options_helper/analysis/strategy_simulator_trade_paths.py`
  - (maybe) `options_helper/analysis/strategy_simulator_trade_utils.py`
- **description**:
  - Implement staged stop updates:
    - Use **prior session** daily EMA(span) and ATR14 to compute a candidate stop.
    - Apply stop changes only at **next session start** (no same-bar lookahead).
    - Tighten-only behavior (long: raise stop; short: lower stop).
    - Stage selection: activate highest `start_r` satisfied by max close-to-date in R units.
  - Add optional `daily_ohlc_by_symbol` input for deterministic indicator calculation.
  - Ensure behavior is unchanged when `stop_trail_rules` is empty/disabled.
  - Define session alignment explicitly:
    - Determine each intraday bar’s `session_date` (existing conventions).
    - “Prior session” for a given `session_date` means the latest daily bar strictly **before** that date.
    - If prior-session EMA/ATR is missing/NaN, keep stop unchanged and emit a trace reason.
  - Lookahead bias rule in writing:
    - Stage activation is computed only from **completed bar closes** and applied no earlier than the **next session open**.
- **validation**:
  - Deterministic unit tests added in T13/T14.
- **status**: Completed
- **log**:
- Implemented staged EMA stop-trail logic using prior-session daily EMA/ATR context with next-session-open application and tighten-only updates.
  - Added missing-indicator trace behavior and inert path when stop trails are disabled/empty.
  - Added focused simulator stop-trail tests.
  - Commit: `d8569f5`.
- **files edited/created**:
- `options_helper/analysis/strategy_simulator_trade_paths.py`
  - `options_helper/analysis/strategy_simulator_trade_utils.py`
  - `tests/test_strategy_simulator_stop_trails.py`

### T9: Strategy-modeling wiring + artifact metadata
- **depends_on**: [T7, T8]
- **location**:
  - `options_helper/analysis/strategy_modeling.py`
  - `options_helper/data/strategy_modeling_artifacts.py`
  - `options_helper/schemas/strategy_modeling_artifact.py` (if metadata schema needs extension)
- **description**:
  - Pass `stop_trail_rules` from policy into simulator.
  - Provide daily OHLC context to simulator in a stable way (no network).
  - Include `stop_trail_rules` in written `policy_metadata` for reproducibility.
- **validation**:
  - Existing strategy-modeling tests still pass.
  - New test asserts artifact metadata contains `stop_trail_rules` when configured.
- **status**: Completed
- **log**:
- Wired policy `stop_trail_rules` and deterministic `daily_ohlc_by_symbol` through strategy-modeling simulation flow.
  - Extended artifact writer metadata to include sorted `stop_trail_rules` for reproducibility.
  - Added service/artifact tests for wiring assertions.
  - Commit: `af9b686`.
- **files edited/created**:
- `options_helper/analysis/strategy_modeling.py`
  - `options_helper/data/strategy_modeling_artifacts.py`
  - `tests/test_strategy_modeling_service.py`
  - `tests/test_strategy_modeling_artifacts.py`

### T10: Add stop-update observability (for testing + debugging)
- **depends_on**: [T6]
- **location**:
  - `options_helper/schemas/strategy_modeling_contracts.py`
  - `options_helper/analysis/strategy_simulator_trade_paths.py`
- **description**:
  - Add a minimal stop trace to `StrategyTradeSimulation`, enough to assert staged behavior deterministically:
    - either `stop_updates: list[{ts, stop_price, reason, stage}]`, or
    - a smaller surface such as `stop_trail_stage_final` + `stop_trail_activated_at_ts` + `stop_trail_rule_final`.
  - Ensure schema remains backward compatible (optional fields with defaults).
- **validation**:
  - Simulator unit tests can assert EMA21→EMA9 stage switching without relying on floating-point final stop only.
- **status**: Completed
- **log**:
- Added stop-update observability contract (`StrategyTradeStopUpdate`, `stop_updates`) with backward-compatible defaults.
  - Propagated stop-update traces through simulator trade path and close-trade shaping for deterministic assertions.
  - Commit: `d8569f5`.
- **files edited/created**:
- `options_helper/schemas/strategy_modeling_contracts.py`
  - `options_helper/analysis/strategy_simulator_trade_paths.py`
  - `options_helper/analysis/strategy_simulator_trade_utils.py`
  - `tests/test_strategy_simulator_stop_trails.py`

### T11: CLI tests for `technicals regime-tactic`
- **depends_on**: [T3, T4, T5]
- **location**:
  - `tests/test_cli_regime_tactic.py` (new)
  - `tests/fixtures/...` (new deterministic OHLC fixtures as needed)
- **description**:
  - Use `CliRunner` to run command with `--ohlc-path` and `--market-ohlc-path`.
  - Enforce “offline-only” behavior for tests:
    - if OHLC paths are omitted and cache is missing, fail fast with a clear error (no silent network fallback).
  - Define and validate the OHLC file contract:
    - required columns, date parsing, sorting, duplicate timestamps, timezone assumptions.
  - Assert:
    - exit code success,
    - printed recommendation contains the selected tactic,
    - JSON artifact contains required keys and disclaimer.
- **validation**:
  - `pytest -k regime_tactic`.
- **status**: Completed
- **log**:
- Expanded regime-tactic CLI tests for offline-only behavior and OHLC input contract edge cases (parse/tz validation).
  - Verified recommendation output and artifact disclaimer coverage remain deterministic.
  - Commit: `6cba54c`.
- **files edited/created**:
- `tests/test_regime_tactic_cli.py`

### T12: Regime-tactic CLI error handling / input validation
- **depends_on**: [T3]
- **location**:
  - `options_helper/commands/technicals/regime_tactic.py`
- **description**:
  - Add explicit user-facing errors for:
    - missing OHLC sources (no path, no cache),
    - malformed OHLC inputs (missing columns, unsorted timestamps, duplicates).
  - Keep messages actionable and deterministic.
- **validation**:
  - CLI tests in T11 cover the failure modes.
- **status**: Completed
- **log**:
- Added deterministic user-facing validation errors for missing OHLC sources and malformed OHLC files (columns/sort/duplicates/timezones).
  - Covered failure modes via CLI tests in `test_regime_tactic_cli.py`.
  - Commit: `97aef41`.
- **files edited/created**:
- `options_helper/commands/technicals/regime_tactic.py`
  - `tests/test_regime_tactic_cli.py`

### T13: Simulator unit tests for stop trails (staged EMA)
- **depends_on**: [T8, T10]
- **location**:
  - Extend `tests/test_strategy_simulator.py` or add `tests/test_strategy_simulator_stop_trails.py`
- **description**:
  - Deterministic unit tests asserting:
    - stage switches at the correct R threshold,
    - EMA21 applied early, EMA9 after threshold,
    - no lookahead (uses prior session values only),
    - NaN/insufficient daily indicator inputs keep stop unchanged with a recorded reason/stage.
- **validation**:
  - `pytest -k stop_trails`.
- **status**: Completed
- **log**:
- Added simulator tests for exact stage-threshold activation, EMA21→EMA9 switch timing, and no-lookahead behavior.
  - Added missing-indicator test that asserts unchanged stop and trace reason/stage fields.
  - Commit: `78c559f`.
- **files edited/created**:
- `tests/test_strategy_simulator_stop_trails.py`

### T14: Strategy-modeling integration tests for stop trails end-to-end
- **depends_on**: [T7, T9, T10, T5]
- **location**:
  - Extend `tests/test_strategy_modeling_*` or add `tests/test_strategy_modeling_stop_trails.py`
- **description**:
  - Run a small modeling flow over fixtures with stop trails enabled.
  - Assert:
    - the simulator output records the EMA21→EMA9 stage transition (via the trace fields),
    - artifacts preserve the configured rule set in `policy_metadata`.
- **validation**:
  - `pytest -k stop_trails`.
- **status**: Completed
- **log**:
- Added end-to-end strategy-modeling stop-trail integration test validating stage-transition traces and artifact policy metadata persistence.
  - Added minimal simulator-path wiring to consume forwarded stop-trail kwargs in default simulation path.
  - Commit: `916c185`.
- **files edited/created**:
- `tests/test_strategy_modeling_stop_trails.py`
  - `options_helper/analysis/strategy_simulator.py`

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2, T5, T6 | Immediately |
| 2 | T3 | T1 + T2 done |
| 3 | T4, T12 | T3 done |
| 4 | T7, T8, T10 | T6 done |
| 5 | T9 | T7 + T8 done |
| 6 | T11 | T3 + T4 + T5 done |
| 7 | T13 | T8 + T10 done |
| 8 | T14 | T7 + T9 + T10 + T5 done |

## Testing Strategy
- Fast unit tests first:
  - `pytest -k price_regime`
  - `pytest -k regime_tactic`
  - `pytest -k stop_trails`
- Then run the full suite:
  - `pytest`

## Risks & Mitigations
- **Lookahead bias**: Stop updates must use only prior-session data and apply at next session open; add explicit tests.
- **Indicator NaNs / short history**: Classifier must return `insufficient_data` and remain stable; test NaN cases.
- **Typer parsing quirks**: Avoid `datetime.date` annotations; parse date strings manually with clear `BadParameter` errors.
- **Performance**: Precompute indicator series once per symbol per run; avoid repeated pandas slicing in loops.
