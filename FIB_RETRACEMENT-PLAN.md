# Plan: Add `fib_retracement` Strategy (Fibonacci Retracement w/ MSB + Next Swing Pivot)

This project is **not financial advice**. The goal is to add a new **strategy-modeling** strategy that emits close-confirmed signal events and uses the existing next-bar-open entry semantics.

## Summary (Decisions Locked)
- **Strategy id**: `fib_retracement`
- **Entry model**: **next regular-session open after the fib-touch candle** (no limit fills at fib price)
- **Swing/MSB params**: **fixed defaults** (same defaults as current MSB implementation; no new swing knobs)
- **Fib level default**: `61.8` (percent)
- **Stops**:
  - Long: `stop = range_low` (swing low before MSB)
  - Short: `stop = range_high` (swing high before MSB)

## Public Interfaces / Contracts to Change
1. **Strategy ID union**
   - Update `StrategyId` to include `fib_retracement`.
2. **CLI**
   - `options-helper technicals strategy-model --strategy fib_retracement`
   - New option: `--fib-retracement-pct` (float, default `61.8`)
3. **Profiles**
   - Add `fib_retracement_pct` field to saved profiles (default `61.8`).
4. **Streamlit**
   - Add `fib_retracement` to strategy dropdown.
   - Add a numeric input for `fib_retracement_pct`.

## Strategy Semantics (Decision-Complete Spec)
### Core Definitions
- Use the **existing MSB swing + break logic** by building on `compute_msb_signals(...)` semantics:
  - Swings use left/right confirmation bars (defaults in MSB today).
  - Bullish MSB: close crosses **above** last confirmed swing high (and prev close ≤ level), respecting MSB’s min distance default.
  - Bearish MSB: close crosses **below** last confirmed swing low (and prev close ≥ level).

### Long Setup → Entry Signal
1. **Trigger**: first bullish MSB encountered while no long setup is active.
2. **Range low**: the **last confirmed swing low** at the MSB bar (the “swing low before the MSB”).
3. **Range high**: the **next swing high after the MSB**.
4. **Confirmation guard**: do **not** search for fib touch until the range-high swing is confirmed and **one full bar after confirmation**:
   - If `range_high_idx` is the swing pivot index and `right = swing_right_bars`, confirmation becomes knowable at bar index `range_high_idx + right` **close**.
   - Start fib-touch scan at `scan_start_idx = range_high_idx + right + 1`.
5. **Fib price**:
   - `fib_ratio = (normalize(fib_retracement_pct) / 100.0)` where `normalize` accepts `61.8` or `0.618` and returns **percent**.
   - `entry_level = range_high - (range_high - range_low) * fib_ratio`
6. **Touch condition** (daily/strategy timeframe bar): first bar `i >= scan_start_idx` where `low[i] <= entry_level <= high[i]`.
7. **Emit signal event on bar i**:
   - `signal_ts = index[i]` (close-confirmed)
   - `signal_confirmed_ts = signal_ts`
   - `entry_ts = index[i+1]` (must exist; otherwise drop)
   - `stop_price = range_low`
   - `signal_open/high/low/close` set from bar `i`
   - `notes` includes: MSB broken swing info, range high/low timestamps + levels, `fib_retracement_pct`, computed `entry_level`, and a fixed note that entry anchors to next-bar open.
8. **De-dup / state**: after emitting, clear long setup; ignore additional bullish MSBs while waiting.
   - **Update rule in stage-1**: while waiting for the next swing high, if another bullish MSB occurs, **reset** the long setup to the *latest* MSB (so “swing low before MSB” matches the MSB closest to the eventual range high).

### Short Setup → Entry Signal (Mirror)
- Trigger on bearish MSB.
- Range high = last confirmed swing high at MSB bar.
- Range low = next swing low after MSB.
- After range-low confirmation + 1 bar, compute:
  - `entry_level = range_low + (range_high - range_low) * fib_ratio`
- Touch condition: `low[i] <= entry_level <= high[i]`
- Emit short event with:
  - `stop_price = range_high`
  - `direction="short"`

### Validity Guards
- Skip setup if required swing anchor is missing (no last swing low/high available at MSB).
- Skip if range is degenerate/non-positive:
  - Long requires `range_high > range_low`
  - Short requires `range_high > range_low`
- Optional but recommended guard to match intent:
  - Long requires `range_high > broken_swing_high_level` (from the MSB bar)
  - Short requires `range_low < broken_swing_low_level`

## Implementation Tasks (Dependency-Aware)

### T1: Extend strategy id contract
- **depends_on**: []
- **location**:
  - `/Volumes/develop/options-helper/options_helper/schemas/strategy_modeling_contracts.py`
  - `/Volumes/develop/options-helper/tests/test_strategy_modeling_contracts.py`
- **description**:
  - Add `"fib_retracement"` to `StrategyId = Literal[...]`.
  - Update the paramized test that asserts extended ids are accepted to include `fib_retracement`.
- **validation**:
  - `python -m pytest /Volumes/develop/options-helper/tests/test_strategy_modeling_contracts.py`

### T2: Implement fib retracement signal computation (pure analysis)
- **depends_on**: []
- **location**:
  - New: `/Volumes/develop/options-helper/options_helper/analysis/fib_retracement.py`
- **description**:
  - Implement:
    - `normalize_fib_retracement_pct(value: object) -> float` returning a **percent** in `(0, 100]`:
      - if `0 < v <= 1`: treat as ratio → return `v * 100`
      - if `1 < v <= 100`: treat as percent → return `v`
      - else: raise `ValueError`
    - `compute_fib_retracement_signals(ohlc: pd.DataFrame, *, fib_retracement_pct: float = 61.8, timeframe: str|None=None) -> pd.DataFrame`
      - Build on `compute_msb_signals(...)` with **its defaults** (no new CLI knobs).
      - Run the state machines described above.
      - Add columns (set values only on signal rows; otherwise NaN/None/False):
        - `fib_retracement_long` (bool)
        - `fib_retracement_short` (bool)
        - `fib_entry_level` (float)
        - `fib_retracement_pct` (float percent)
        - `fib_range_high_level`, `fib_range_low_level` (float)
        - `fib_range_high_timestamp`, `fib_range_low_timestamp` (ISO str)
        - `fib_msb_timestamp` (ISO str)
        - `fib_broken_swing_level` (float) + `fib_broken_swing_timestamp` (ISO str)
  - Keep analysis deterministic and free of I/O.
- **validation**:
  - Add and run unit tests in T8.

### T3: Wire `fib_retracement` into strategy signal registry
- **depends_on**: [T1, T2]
- **location**:
  - `/Volumes/develop/options-helper/options_helper/analysis/strategy_signals.py`
- **description**:
  - Add adapter functions mirroring existing patterns:
    - `normalize_fib_retracement_signal_events(...) -> list[StrategySignalEvent]`
    - `adapt_fib_retracement_signal_events(ohlc, *, symbol, timeframe=None, fib_retracement_pct=61.8, entry_price_source=...)`
  - Event emission:
    - `strategy="fib_retracement"`
    - `event_id = f"fib_retracement:{SYMBOL}:{timeframe_label}:{signal_ts_iso}:{direction}"`
    - `entry_ts = next bar timestamp` (drop last-bar touches)
    - `stop_price` per rules
    - `notes` populated from compute columns + fixed anti-lookahead note
  - Register:
    - `register_strategy_signal_adapter("fib_retracement", adapt_fib_retracement_signal_events, replace=True)`
- **validation**:
  - Add and run unit tests in T8 that call `build_strategy_signal_events("fib_retracement", ...)`.

### T4: Add fib pct to profile schema
- **depends_on**: [T1]
- **location**:
  - `/Volumes/develop/options-helper/options_helper/schemas/strategy_modeling_profile.py`
- **description**:
  - Add field: `fib_retracement_pct: float = Field(default=61.8, gt=0.0, le=100.0)`
  - Add a `field_validator(..., mode="before")` to accept `0.618` or `61.8` and normalize to percent (store percent).
- **validation**:
  - `python -m pytest /Volumes/develop/options-helper/tests/test_strategy_modeling_cli.py` (after T5 updates)

### T5: CLI: enable `--strategy fib_retracement` + `--fib-retracement-pct`
- **depends_on**: [T1, T3, T4]
- **location**:
  - `/Volumes/develop/options-helper/options_helper/commands/technicals.py`
  - `/Volumes/develop/options-helper/tests/test_strategy_modeling_cli.py`
- **description**:
  - Update `--strategy` help text to include `fib_retracement`.
  - Update strategy allowlist check to include `fib_retracement` and update error message string.
  - Add Typer option param:
    - `fib_retracement_pct: float = typer.Option(61.8, "--fib-retracement-pct", help="Fib retracement percent (e.g., 61.8; accepts 0.618 as ratio).")`
  - Extend `_build_strategy_signal_kwargs(...)`:
    - accept `fib_retracement_pct` param
    - if `strategy == "fib_retracement"`: validate/normalize and return `{"fib_retracement_pct": normalized_percent}`
  - Ensure `cli_profile_values` includes `"fib_retracement_pct": float(fib_retracement_pct)`
  - Ensure the pre-merge validation call to `_build_strategy_signal_kwargs(...)` passes the new argument.
  - Update CLI tests that assert invalid strategy output contains the allowlist (now including `fib_retracement`).
- **validation**:
  - `python -m pytest /Volumes/develop/options-helper/tests/test_strategy_modeling_cli.py`

### T6: Streamlit: add strategy option + fib pct widget + profile integration
- **depends_on**: [T3, T4]
- **location**:
  - `/Volumes/develop/options-helper/apps/streamlit/pages/11_Strategy_Modeling.py`
- **description**:
  - Add `fib_retracement` to `_STRATEGY_OPTIONS`.
  - Add widget key in `_WIDGET_KEYS`:
    - `"fib_retracement_pct": "strategy_modeling_fib_retracement_pct"`
  - In sidebar inputs, when `strategy == "fib_retracement"` show a `number_input` for fib percent (default `61.8`).
  - Extend `_build_signal_kwargs(...)` to accept `fib_retracement_pct` and return `{"fib_retracement_pct": normalized_percent}` when strategy is fib.
  - Extend `_build_profile_from_inputs(...)` payload to include `fib_retracement_pct`.
  - Extend `_apply_loaded_profile_to_state(...)` to set the widget state from loaded profile.
  - Ensure run disable logic accounts for any new validation errors.
- **validation**:
  - `python -m pytest /Volumes/develop/options-helper/tests/portal/test_streamlit_scaffold.py`

### T7: Documentation + mkdocs nav
- **depends_on**: [T5, T6]
- **location**:
  - New: `/Volumes/develop/options-helper/docs/TECHNICAL_FIB_RETRACEMENT.md`
  - Update: `/Volumes/develop/options-helper/docs/TECHNICAL_STRATEGY_MODELING.md`
  - Update: `/Volumes/develop/options-helper/mkdocs.yml`
- **description**:
  - New doc covers:
    - What the strategy models (MSB → next swing pivot → fib touch)
    - CLI examples including `--fib-retracement-pct`
    - Anti-lookahead statement: signal at bar close, entry at next bar open
    - Limitations: entry is *not* a limit fill at fib level
  - Update strategy-modeling doc to include fib in the strategy list and briefly summarize semantics + link to the new doc.
  - Add the new doc to mkdocs nav under “Research & Analysis”.
- **validation**:
  - (Optional) `mkdocs build` if available in this repo’s tooling; otherwise ensure links/paths are correct.

### T8: Add deterministic unit tests for fib retracement strategy
- **depends_on**: [T1, T2, T3, T5, T6]
- **location**:
  - New: `/Volumes/develop/options-helper/tests/test_fib_retracement.py`
  - New (or extend): `/Volumes/develop/options-helper/tests/test_strategy_signals_fib_retracement.py`
  - Update: `/Volumes/develop/options-helper/tests/test_strategy_modeling_contracts.py`
  - Update: `/Volumes/develop/options-helper/tests/test_strategy_modeling_cli.py`
- **description**:
  - Unit test `compute_fib_retracement_signals` with a synthetic OHLC series that:
    - creates a bullish MSB
    - produces a next swing high
    - confirms it with right-bars lag
    - then touches the computed fib level
    - asserts:
      - exactly one `fib_retracement_long` flag
      - `fib_entry_level` matches expected within tolerance
      - stop anchor equals the intended swing low/high
      - no signal emitted without `i+1` bar available (end-of-series guard)
  - Adapter test: `build_strategy_signal_events("fib_retracement", ...)` returns correct `StrategySignalEvent` fields:
    - `direction`, `stop_price`, `entry_ts == next index`, `strategy == fib_retracement`
  - CLI regression: invalid strategy error output includes `fib_retracement`.
- **validation**:
  - `python -m pytest /Volumes/develop/options-helper/tests/test_fib_retracement.py`
  - `python -m pytest /Volumes/develop/options-helper/tests/test_strategy_signals_fib_retracement.py`
  - `python -m pytest /Volumes/develop/options-helper/tests/test_strategy_modeling_cli.py`

## Dependency Graph

```
T1 ──┬─────────┬── T3 ──┬── T5 ──┬── T7 ──┐
     │         │        │        │        │
     │         └── T4 ──┘        └────────┤
     │                                   │
T2 ──┘                                   └── T8
                 T3 ──┬── T6 ───────────────┘
                      └──────────────────────
```

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2, T4 | Immediately |
| 2 | T3 | T1 + T2 complete |
| 3 | T5, T6 | T3 + T4 complete |
| 4 | T7 | T5 + T6 complete |
| 5 | T8 | T1–T6 complete |

## Acceptance Criteria
- `./.venv/bin/options-helper technicals strategy-model --strategy fib_retracement ...` runs and writes artifacts like other strategies.
- Strategy allowlists (schema + CLI + Streamlit) include `fib_retracement`.
- Unit tests demonstrate:
  - correct MSB → range → fib-touch detection
  - no lookahead: entry anchor is always next bar open after signal confirmation

## Risks & Mitigations
- **Lookahead bias** (range pivot uses right-side confirmation): mitigate by enforcing `scan_start_idx = pivot_idx + right + 1` and emitting entry only on subsequent bars.
- **Percent vs ratio confusion**: mitigate by normalization accepting `0.618` or `61.8` and storing **percent** in profiles.
- **Duplicate/overlapping setups**: mitigate with explicit long/short state machines and “latest MSB wins while waiting for next pivot” rule.

## Task Completion Log
### T1: Extend strategy id contract (completed 2026-02-10)
- Work log: Added `fib_retracement` to `StrategyId` and extended the strategy-id acceptance contract test to include it.
- Files modified: `options_helper/schemas/strategy_modeling_contracts.py`, `tests/test_strategy_modeling_contracts.py`.
- Validation: `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/test_strategy_modeling_contracts.py` passed (`15 passed`).
- Errors/gotchas: Initial validation failed because `.venv` and `pytest` were absent in this checkout; resolved by creating `.venv` and installing `-e .[dev]`.

### T4: Add fib pct to profile schema (completed 2026-02-10)
- Work log: Added `fib_retracement_pct` to `StrategyModelingProfile` and normalized input with a before-validator that accepts ratio (`0.618`) or percent (`61.8`) using the shared fib normalization helper.
- Files modified: `options_helper/schemas/strategy_modeling_profile.py`.
- Validation: `./.venv/bin/python -m py_compile /Volumes/develop/options-helper-fib-retracement/options_helper/schemas/strategy_modeling_profile.py` passed.
- Errors/gotchas: None.

### T2: Implement fib retracement signal computation (completed 2026-02-10)
- Work log:
  - Added new pure analysis module `fib_retracement.py` with `normalize_fib_retracement_pct(...)` accepting ratio (`0.618`) and percent (`61.8`) forms and normalizing to percent.
  - Implemented `compute_fib_retracement_signals(...)` on top of `compute_msb_signals(...)` defaults, with deterministic long/short setup state machines, pivot-confirmation guard (`scan_start_idx = pivot_idx + right + 1`), and first-touch signal emission.
  - Added required fib signal metadata columns and end-of-series guard (drop touch on final bar where next-bar entry is unavailable).
- Files modified: `options_helper/analysis/fib_retracement.py`.
- Validation: `./.venv/bin/python -m py_compile /Volumes/develop/options-helper-fib-retracement/options_helper/analysis/fib_retracement.py` passed.
- Errors/gotchas: None.

### T3: Wire `fib_retracement` into strategy signal registry (completed 2026-02-10)
- Work log:
  - Added `normalize_fib_retracement_signal_events(...)` to convert fib signal rows into `StrategySignalEvent` objects with close-confirmed signal timestamps and next-bar-open entry anchors.
  - Added `adapt_fib_retracement_signal_events(...)` to call `compute_fib_retracement_signals(...)` and normalize the results for `build_strategy_signal_events(...)`.
  - Registered the strategy adapter for `fib_retracement` and exported the new adapter/normalizer in module `__all__`.
- Files modified: `options_helper/analysis/strategy_signals.py`.
- Validation: `./.venv/bin/python -m py_compile /Volumes/develop/options-helper-fib-retracement/options_helper/analysis/strategy_signals.py` passed.
- Errors/gotchas: None.

### T6: Streamlit add strategy option + fib pct widget + profile integration (completed 2026-02-10)
- Work log:
  - Added `fib_retracement` to Streamlit strategy options and added `_WIDGET_KEYS["fib_retracement_pct"]` with sidebar default state `61.8`.
  - Added a conditional sidebar `number_input` for fib percent shown only when strategy is `fib_retracement`.
  - Extended `_build_signal_kwargs(...)` to accept `fib_retracement_pct`, normalize it via `normalize_fib_retracement_pct(...)`, and return fib signal kwargs for the fib strategy.
  - Extended profile save/load wiring to persist and hydrate `fib_retracement_pct` via `_build_profile_from_inputs(...)` and `_apply_loaded_profile_to_state(...)`.
- Files modified: `apps/streamlit/pages/11_Strategy_Modeling.py`.
- Validation: `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/portal/test_streamlit_scaffold.py` passed (`1 passed, 3 skipped`); `./.venv/bin/python -m py_compile /Volumes/develop/options-helper-fib-retracement/apps/streamlit/pages/11_Strategy_Modeling.py` passed.
- Errors/gotchas: None.

### T5: CLI enable `fib_retracement` strategy + fib pct option (completed 2026-02-10)
- Work log:
  - Enabled `fib_retracement` in `technicals strategy-model` strategy help/allowlist and invalid-value error messaging.
  - Added `--fib-retracement-pct` option (default `61.8`) and wired it through CLI pre-validation, profile-merge payload, and final strategy signal kwargs construction.
  - Extended `_build_strategy_signal_kwargs(...)` to normalize/validate fib percent and emit `{"fib_retracement_pct": normalized_percent}` for fib strategy.
  - Added CLI regression tests for updated allowlist messaging, fib kwargs normalization from ratio input, and invalid fib pct rejection.
- Files modified: `options_helper/commands/technicals.py`, `tests/test_strategy_modeling_cli.py`, `FIB_RETRACEMENT-PLAN.md`.
- Validation: `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/test_strategy_modeling_cli.py` passed.
- Errors/gotchas: None.

### T7: Documentation + mkdocs nav (completed 2026-02-10)
- Work log:
  - Added `docs/TECHNICAL_FIB_RETRACEMENT.md` with strategy semantics (`MSB -> next swing pivot -> fib touch`), CLI usage with `--fib-retracement-pct`, anti-lookahead entry anchoring, and explicit fill limitations.
  - Updated `docs/TECHNICAL_STRATEGY_MODELING.md` to include `fib_retracement` in strategy docs, added fib option/example text, and linked to the dedicated fib strategy document.
  - Added the fib doc to `mkdocs.yml` nav under `Research & Analysis`.
- Files modified: `docs/TECHNICAL_FIB_RETRACEMENT.md`, `docs/TECHNICAL_STRATEGY_MODELING.md`, `mkdocs.yml`, `FIB_RETRACEMENT-PLAN.md`.
- Validation: `./.venv/bin/mkdocs build` unavailable in this checkout (`no such file or directory: ./.venv/bin/mkdocs`); manually verified doc file presence and link/nav references.
- Errors/gotchas: Local `.venv` does not include `mkdocs` binary.

### T8: Add deterministic unit tests for fib retracement strategy (completed 2026-02-10)
- Work log:
  - Added deterministic compute coverage for `compute_fib_retracement_signals(...)` with synthetic OHLC fixtures that explicitly assert bullish MSB formation, next swing-high selection, right-bars confirmation lag behavior, fib-touch triggering, expected entry level tolerance, and long stop-anchor metadata (`fib_range_low_level`).
  - Added an explicit i+1 anti-lookahead guard regression where a valid fib touch on the final bar emits no signal due to missing next-bar entry anchor.
  - Added adapter coverage through `build_strategy_signal_events("fib_retracement", ...)` asserting normalized `strategy`, `direction`, `stop_price`, and `entry_ts == next index`, plus final-bar touch suppression.
  - Kept CLI regression coverage in `tests/test_strategy_modeling_cli.py` that already asserts invalid strategy output includes `fib_retracement`.
- Files modified: `tests/test_fib_retracement.py`, `tests/test_strategy_signals_fib_retracement.py`, `FIB_RETRACEMENT-PLAN.md`.
- Validation:
  - `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/test_fib_retracement.py` passed (`2 passed`).
  - `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/test_strategy_signals_fib_retracement.py` passed (`2 passed`).
  - `./.venv/bin/python -m pytest /Volumes/develop/options-helper-fib-retracement/tests/test_strategy_modeling_cli.py` passed (`24 passed`).
- Errors/gotchas: No implementation fixes were required in analysis/adapter/CLI code for T8; adapter test run emits an existing pandas resample deprecation warning from `options_helper/analysis/sfp.py`.
