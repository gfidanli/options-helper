# Plan: Modular Strategy Modeling (SFP/MSB/ORB) + Entry Filters + Directional Results (CLI + Streamlit)

**Generated**: 2026-02-08

## Summary
Extend the existing strategy-modeling service to support:
- A new primary strategy: **ORB** (opening-range breakout).
- An optional **ORB confirmation gate** for SFP/MSB that shifts `signal_confirmed_ts`/`entry_ts` deterministically (anti-lookahead safe).
- A modular **entry filter framework** (short toggle, ATR floor, RSI extremes, EMA9 regime, volatility regime).
- **Directional results** (combined vs long-only vs short-only) as counterfactual portfolio reruns.
- Parity updates to **CLI + artifacts + Streamlit Strategy Modeling page**.

All outputs remain explicitly **informational / not financial advice**.

## Locked Decisions (from review + your answers)
- **Scope**: Update **CLI + artifacts + Streamlit** (feature parity).
- **ORB confirmation gate stop policy**: Default **keep base stop**; also support modeling the **ORB-range stop** variant via a config flag (and optional “tighten”).
- **ORB daily-indicator anchoring**: For intraday ORB signals, daily-indicator filters use the **most recent completed daily bar strictly before the ORB signal day** (avoids lookahead; no intraday-indicator compute).
- **Composition model**: Base strategy + optional gates/filters (AND logic); new filters default **OFF**.
- **ORB definition**: 15-minute opening range breakout **by close**; cutoff configurable (default **10:30 ET**).
- **Shorts**: Default **ON**; allow explicit disabling.

## Public Interface / Contract Changes

### CLI (`options-helper technicals strategy-model`)
Additive options (Typer style `--flag/--no-flag`):
- `--strategy sfp|msb|orb`
- `--allow-shorts/--no-allow-shorts` (default `--allow-shorts`)
- `--enable-orb-confirmation/--no-enable-orb-confirmation` (default off)
- `--orb-range-minutes` (default `15`)
- `--orb-confirmation-cutoff-et` (default `"10:30"`, parsed as `HH:MM` ET)
- `--orb-stop-policy base|orb_range|tighten` (default `base`)
- `--enable-atr-stop-floor/--no-enable-atr-stop-floor` (default off)
- `--atr-stop-floor-multiple` (default `0.5`)
- `--enable-rsi-extremes/--no-enable-rsi-extremes` (default off)
- `--enable-ema9-regime/--no-enable-ema9-regime` (default off)
- `--ema9-slope-lookback-bars` (default `3`)
- `--enable-volatility-regime/--no-enable-volatility-regime` (default off)
- `--allowed-volatility-regimes low,normal,high` (default all)

Notes:
- Keep **date/time options as strings + explicit parsing**, consistent with current repo patterns and prior Typer pitfalls.

### Strategy Modeling Contracts / Types
- Extend `StrategyId` to include `"orb"` in `options_helper/schemas/strategy_modeling_contracts.py`.
- Add new schema model `StrategyEntryFilterConfig` in `options_helper/schemas/strategy_modeling_filters.py` (strict, `extra="forbid"`), containing all filter/gate settings above.
- Extend `StrategyModelingRequest` (analysis dataclass) with `filter_config: StrategyEntryFilterConfig | Mapping[str, Any] | None = None`.
- Extend `StrategyModelingRunResult` with:
  - `filter_metadata: dict[str, Any]` (effective settings)
  - `filter_summary: dict[str, Any]` (counts by reject reason, plus base/kept counts)
  - `directional_metrics: dict[str, Any]` (combined/long_only/short_only metric payloads + counts)

### Artifact outputs (`data/reports/technicals/strategy_modeling/.../summary.json|summary.md`)
- Additive fields:
  - `filter_metadata`, `filter_summary`, `directional_metrics`
- Update tests to assert **required keys** (subset), not exact top-level key sets, so future additive fields don’t cause churn.

## Key Behavioral Specs (Decision-Complete)

### ORB strategy (primary)
Per symbol, per session day:
- Use **regular-session** bars only (America/New_York).
- Opening range window: `09:30:00 <= ts < 09:30 + range_minutes`.
- Compute `opening_range_high = max(high)`, `opening_range_low = min(low)` for that window.
- Breakout confirmation (by close, before cutoff ET):
  - Bullish: first bar after range window with `close > opening_range_high`
  - Bearish: first bar after range window with `close < opening_range_low`
- **Anti-lookahead timestamps**:
  - `signal_ts`: breakout bar **open** ts (for traceability)
  - `signal_confirmed_ts`: breakout bar **close** ts (open + bar_duration)
  - `entry_ts`: **next bar open** strictly after `signal_confirmed_ts`
- Stop:
  - Long: `stop_price = opening_range_low`
  - Short: `stop_price = opening_range_high`
- **One event max per session**: choose whichever breakout confirms first in time (tie-break: prefer stop-first conservatism doesn’t apply; just pick earliest confirmed breakout).

### ORB confirmation gate (for SFP/MSB)
For each SFP/MSB base event:
- Evaluate ORB on the **entry session day** (the session day implied by the event’s current `entry_ts` anchor).
- If breakout in event direction confirms by cutoff:
  - Set `signal_confirmed_ts` to ORB breakout bar close ts
  - Set `entry_ts` to next bar open after breakout confirmation
  - Apply stop per `orb_stop_policy`:
    - `base`: keep base stop (default)
    - `orb_range`: replace stop with ORB opening-range stop
    - `tighten`: choose tighter stop (long => max(stop prices), short => min(stop prices))
- If not confirmed (or opening-range bars missing): reject event with a stable reason code (see below).

### Filters (AND logic; evaluated in stable order)
Recommended order (cheap → expensive):
1. `allow_shorts` gate
2. RSI extremes (daily)
3. EMA9 regime (daily)
4. Volatility regime (daily)
5. ORB confirmation gate (intraday; mutates timestamps/stop on success)
6. ATR floor (uses **entry initial risk** / daily ATR at indicator anchor)

Indicator anchoring:
- SFP/MSB events: daily indicator row keyed by the **signal session date**.
- ORB events: daily indicator row keyed by **prior daily bar** (most recent `< session_date`).

### Filter audit + reason codes
Implement a small fixed set of reason codes (snake_case strings) used by CLI/artifacts/tests, e.g.:
- `shorts_disabled`
- `missing_daily_context`
- `rsi_not_extreme`
- `ema9_regime_mismatch`
- `volatility_regime_disallowed`
- `orb_opening_range_missing`
- `orb_breakout_missing`
- `atr_floor_failed`

## Dependency Graph (Swarm-parallel)
```text
T1 ─┬─ T2 ─┬─ T4 ─┬─ T5 ─┬─ T6 ─┬─ T7 ─┬─ T9 ─┬─ T10
    │      │      │      │      │      │      │
    │      │      │      │      └─ T8 ─┘      │
    └─ T3 ─┴──────┘      └───────────────┬────┘
                                         └─ (docs can run in parallel once interfaces are stable)
```

## Tasks

### T1: Update strategy contracts + add filter config schema (Complete)
- **depends_on**: []
- **location**:
  - `options_helper/schemas/strategy_modeling_contracts.py`
  - `options_helper/schemas/strategy_modeling_filters.py` (new)
  - `options_helper/analysis/strategy_modeling.py`
- **description**:
  - Add `"orb"` to `StrategyId`.
  - Create `StrategyEntryFilterConfig` (strict model; defaults OFF; includes `orb_stop_policy`).
  - Extend `StrategyModelingRequest` and `StrategyModelingRunResult` with filter/directional fields.
- **validation**:
  - Update/extend `tests/test_strategy_modeling_contracts.py` for new `StrategyId`.
  - New unit test for filter config validation (unknown fields rejected, domain constraints enforced).
- **work log**:
  - Added `"orb"` to `StrategyId`.
  - Added strict `StrategyEntryFilterConfig` with domain validation (cutoff format, ranges, regime set constraints) and `orb_stop_policy`.
  - Extended `StrategyModelingRequest` (`filter_config`) and `StrategyModelingRunResult` (`filter_metadata`, `filter_summary`, `directional_metrics`) and wired deterministic default population in service results.
  - Added tests for ORB contract acceptance, filter schema validation, and dataclass-field presence checks.
- **files touched**:
  - `options_helper/schemas/strategy_modeling_contracts.py`
  - `options_helper/schemas/strategy_modeling_filters.py`
  - `options_helper/analysis/strategy_modeling.py`
  - `tests/test_strategy_modeling_contracts.py`
  - `tests/test_strategy_modeling_filters.py`
  - `docs/plans/MODULAR-STRATEGY-MODELING-IMPROVED-PLAN.md`
- **gotchas**:
  - Kept `allow_shorts=True` by default while leaving all explicit filter toggles off, so baseline behavior remains unchanged.
  - Added result-level filter/directional placeholders now so downstream T4/T5 can fill logic without another contract break.

### T2: Implement ORB module + strategy_signals adapter (Complete)
- **depends_on**: [T1]
- **location**:
  - `options_helper/analysis/orb.py` (new)
  - `options_helper/analysis/strategy_signals.py`
  - `tests/test_orb.py` (new)
  - `tests/test_strategy_signals.py`
- **description**:
  - Implement ORB session computation with cutoff + anti-lookahead timestamps.
  - Register `orb` adapter in `strategy_signals.py` (adapter accepts intraday bars via kwargs; service passes them when `strategy=="orb"`).
  - Decide/encode “one ORB event per session” rule.
- **validation**:
  - `tests/test_orb.py`: opening range, breakout detection, cutoff handling, missing bars.
  - `tests/test_strategy_signals.py`: registry includes `orb`; ORB event fields match `STRATEGY_SIGNAL_EVENT_FIELDS`.
- **work log**:
  - Added new `analysis/orb.py` module with deterministic intraday normalization (UTC + regular session filter), opening-range computation, breakout-by-close detection, cutoff enforcement, and one-event-per-session selection.
  - Implemented anti-lookahead ORB timestamps (`signal_ts`, close-confirmed `signal_confirmed_ts`, next-bar `entry_ts`) and stop assignment from opening-range boundaries.
  - Added ORB adapter + normalization path in `strategy_signals` and registered `strategy="orb"` in the adapter registry.
  - ORB adapter accepts intraday data via kwargs as either `intraday_bars` or `intraday_bars_by_symbol` mapping for symbol-level resolution.
  - Added deterministic tests for ORB core logic and strategy-signal contract parity.
- **files touched**:
  - `options_helper/analysis/orb.py`
  - `options_helper/analysis/strategy_signals.py`
  - `tests/test_orb.py`
  - `tests/test_strategy_signals.py`
  - `docs/plans/MODULAR-STRATEGY-MODELING-IMPROVED-PLAN.md`
- **gotchas**:
  - Close-confirmed ORB signals were modeled as `bar_open + bar_duration - 1 microsecond` so `entry_ts` (next bar open) remains strictly greater than confirmation and stays simulator-compatible.
  - ORB detection ignores pre/post-market bars by filtering to regular session (`09:30-16:00 ET`) before opening-range and breakout checks.

### T3: Extend strategy feature enrichment for EMA9 + ATR (Complete)
- **depends_on**: []
- **location**:
  - `options_helper/analysis/strategy_features.py`
  - `tests/test_strategy_features.py`
- **description**:
  - Add `atr` (daily ATR) and `ema9` columns to feature output.
  - Provide slope computation helper (from `ema9` + configurable lookback).
- **validation**:
  - Deterministic unit tests for EMA9 and slope, ATR presence, NaN stability.
- **work log**:
  - Added additive feature columns: `atr`, `ema9`, and `ema9_slope`.
  - Added `ema9_slope_lookback_bars` to `StrategyFeatureConfig` (default `3`) and public helper `compute_ema_slope(...)`.
  - Extended `tests/test_strategy_features.py` with deterministic ATR/EMA9/slope assertions and NaN-safe slope coverage.
- **files touched**:
  - `options_helper/analysis/strategy_features.py`
  - `tests/test_strategy_features.py`
  - `docs/plans/MODULAR-STRATEGY-MODELING-IMPROVED-PLAN.md`
- **gotchas**:
  - EMA9 uses `min_periods=9`, so short histories intentionally produce `NaN` for `ema9` and `ema9_slope`.

### T4: Build filter/gate engine (including ORB confirmation gate)
- **depends_on**: [T1, T2, T3]
- **location**:
  - `options_helper/analysis/strategy_modeling_filters.py` (new)
  - `options_helper/analysis/strategy_modeling.py`
- **description**:
  - Implement `apply_entry_filters(...)` producing:
    - filtered events (possibly with mutated timestamps/stops from ORB gate)
    - `filter_summary` counts by reason
    - `filter_metadata` (effective config + parsed cutoff/ranges)
  - Ensure all filter logic is anti-lookahead safe and deterministic.
  - Performance: precompute per-symbol per-session ORB results once and reuse for all events.
- **validation**:
  - `tests/test_strategy_modeling_regression.py`: ORB gate shifts `signal_confirmed_ts` and sets `entry_ts` strictly after confirmation.
  - New tests for each reject reason code and ordering determinism.

### T5: Directional counterfactual results + “portfolio trade subset” clarification
- **depends_on**: [T4]
- **location**:
  - `options_helper/analysis/strategy_modeling.py`
  - `options_helper/analysis/strategy_portfolio.py`
  - `options_helper/analysis/strategy_metrics.py`
- **description**:
  - Define and implement directional reruns (`combined`, `long_only`, `short_only`) using:
    - the same portfolio construction rules
    - the same target selection policy as baseline portfolio
  - Clarify portfolio semantics when `target_ladder` contains multiple targets:
    - Portfolio ledger/portfolio metrics/segmentation should be computed from a single consistent “portfolio target” subset (default: first ladder entry).
    - R-ladder stats remain computed across all simulated targets.
- **validation**:
  - `tests/test_strategy_modeling_service.py`: directional metrics shape + stable values on deterministic fixtures.
  - Regression that `directional_metrics.long_only` differs when shorts exist.

### T6: CLI wiring + validation + output updates
- **depends_on**: [T1, T4, T5]
- **location**:
  - `options_helper/commands/technicals.py`
  - `tests/test_strategy_modeling_cli.py`
- **description**:
  - Add new flags and parsing (cutoff `HH:MM`, CSV regimes, stop policy enum).
  - Construct `filter_config` and pass through request (`SimpleNamespace(**vars(base_request), filter_config=..., ...)`).
  - CLI output: print filter summary counts + directional headline metrics when applicable.
- **validation**:
  - CLI tests for invalid strategy, invalid cutoff, invalid regimes, and `orb` accepted.
  - Ensure defaults preserve current behavior (all new filters OFF).

### T7: Artifacts: summary.json + trades.csv + summary.md enhancements
- **depends_on**: [T5, T6]
- **location**:
  - `options_helper/data/strategy_modeling_artifacts.py`
  - `tests/test_strategy_modeling_artifacts.py`
  - `tests/test_strategy_modeling_performance.py`
- **description**:
  - Add `filter_metadata`, `filter_summary`, `directional_metrics` to JSON payload.
  - Add Markdown sections: Filters + Directional Results (combined/long-only/short-only).
  - Relax tests to assert required keys rather than exact key sets.
- **validation**:
  - Update artifact tests for new fields + markdown content.
  - Ensure performance smoke remains within runtime/memory thresholds (adjust only if justified with measurement).

### T8: Streamlit Strategy Modeling page parity
- **depends_on**: [T4, T5, T7]
- **location**:
  - `apps/streamlit/pages/11_Strategy_Modeling.py`
  - (optional) `apps/streamlit/components/strategy_modeling_page.py`
  - `tests/portal/test_strategy_modeling_page.py`
- **description**:
  - Add UI controls for strategy=orb and entry filters (mirroring CLI flags).
  - Display filter summary + directional results.
  - Ensure page remains “read-only / no ingestion writes”.
- **validation**:
  - Portal page tests continue to pass (blocked-mode run button, payload shape).
  - Add a lightweight new portal test asserting `orb` appears in strategy options and that filter fields don’t crash request build.

### T9: Documentation updates
- **depends_on**: [T6, T7, T8]
- **location**:
  - `docs/TECHNICAL_STRATEGY_MODELING.md`
  - `docs/plans/MODULAR-STRATEGY-MODELING-PLAN.md`
- **description**:
  - Document ORB strategy + ORB confirmation gate semantics (timestamps, cutoff, stop policy).
  - Document each filter formula, anchoring rules, defaults, and reject reasons.
  - Document directional metrics definition and target-ladder portfolio semantics.
- **validation**:
  - Doc examples match actual CLI flags and artifact keys.

### T10: Consolidated regression + performance verification
- **depends_on**: [T2, T4, T5, T7]
- **location**: `tests/`
- **description**:
  - Add/extend deterministic regression tests for lookahead rules and ORB gating.
  - Run targeted suite + performance smoke.
- **validation commands**:
  - `./.venv/bin/python -m pytest tests/test_strategy_signals.py tests/test_orb.py -q`
  - `./.venv/bin/python -m pytest tests/test_strategy_modeling_service.py tests/test_strategy_modeling_regression.py -q`
  - `./.venv/bin/python -m pytest tests/test_strategy_modeling_cli.py tests/test_strategy_modeling_artifacts.py -q`
  - `./.venv/bin/python -m pytest tests/test_strategy_modeling_performance.py -q`

## Parallel Execution Groups
| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T3 | Immediately |
| 2 | T2 | T1 complete |
| 3 | T4 | T1, T2, T3 complete |
| 4 | T5 | T4 complete |
| 5 | T6 | T1, T4, T5 complete |
| 6 | T7, T8, T9 | T5 + (T6 for docs/CLI text) complete |
| 7 | T10 | T2, T4, T5, T7 complete |

## Assumptions / Defaults
- All new filters are **opt-in** (default OFF) to preserve baseline behavior.
- ORB gate cutoff `"10:30"` is interpreted as **ET wall clock**; breakout bar close time must be `<= cutoff`.
- ORB strategy produces **at most one event per session** (first confirmed breakout wins).
- Directional metrics are **counterfactual portfolio reruns**, not slices of the combined run.
- Artifacts remain **schema_version=1** with additive fields; tests updated to be forward-compatible.
