# Strategy Modeling Profiles (CLI + Dashboard) — Decision-Complete Plan

## Summary
Implement a shared, local profile system so users can save a named set of strategy-modeling inputs once and load it from either the CLI or the Streamlit Strategy Modeling page.

Chosen product decisions:
- Profile scope: full modeling run inputs (not export/report settings).
- CLI UX: inline flags on `technicals strategy-model`.
- Precedence: explicit CLI flags / dashboard edits override loaded profile values.
- Overwrite safety: saving to an existing profile name fails unless explicit overwrite is set.

## Public API / Interface Changes

1. Add shared profile storage + contract
- New schema module: `options_helper/schemas/strategy_modeling_profile.py`
- New data I/O module: `options_helper/data/strategy_modeling_profiles.py`
- Default profile store path: `config/strategy_modeling_profiles.json` (local state)

2. Profile JSON file contract (v1)
- Top-level keys:
  - `schema_version` (int, `1`)
  - `updated_at` (UTC ISO timestamp)
  - `profiles` (mapping of profile name -> profile payload)
- Profile payload fields (normalized, validated):
  - `strategy`
  - `symbols`
  - `start_date`, `end_date`
  - `intraday_timeframe`, `intraday_source`
  - `starting_capital`, `risk_per_trade_pct`, `gap_fill_policy`, `max_hold_bars`, `one_open_per_symbol`
  - `r_ladder_min_tenths`, `r_ladder_max_tenths`, `r_ladder_step_tenths`
  - filter/gate fields (`allow_shorts`, ORB fields, ATR/RSI/EMA9/volatility settings)
  - MA/trend signal fields (`ma_*`, `trend_slope_lookback_bars`, `atr_window`, `atr_stop_multiple`)
- Excluded intentionally (cross-surface non-core run settings): `out`, `write_json/csv/md`, `show_progress`, `segment_*`, `exclude_symbols`, `universe_limit`, `signal_confirmation_lag_bars`, export directory/timezone.

3. CLI additions (`options_helper/commands/technicals.py`)
- Add options to `technicals strategy-model`:
  - `--profile TEXT` (load profile by name)
  - `--save-profile TEXT` (save effective inputs under this name)
  - `--overwrite-profile/--no-overwrite-profile` (default no-overwrite)
  - `--profile-path PATH` (default `config/strategy_modeling_profiles.json`)
- Behavior:
  - Load profile first, then apply explicit CLI arguments on top.
  - Detect explicit arguments using `typer.Context.get_parameter_source(...)` so defaulted flags do not unintentionally override profile values.
  - If `--save-profile` is set, persist normalized effective inputs (after validation) using atomic file replace.
  - Existing profile name without overwrite flag -> `typer.BadParameter` with clear message.

4. Dashboard additions (`apps/streamlit/pages/11_Strategy_Modeling.py`)
- Add a **Profiles** section in sidebar:
  - profile store path input (default `config/strategy_modeling_profiles.json`)
  - profile select dropdown (from file)
  - profile name input for save
  - overwrite checkbox
  - `Load Profile` button (applies values into widget-backed `st.session_state`)
  - `Save Profile` button (saves current effective settings)
- Add missing ladder controls in dashboard so profile fields are fully loadable/run-able there:
  - `r_ladder_min_tenths`, `r_ladder_max_tenths`, `r_ladder_step_tenths`
- Keep portal write semantics explicit-only (no writes on load/render).

5. Local state ignore
- Update `.gitignore` to include:
  - `config/strategy_modeling_profiles.json`

## Implementation Steps

1. Build profile schema + normalization
- Create strict Pydantic models with `extra="forbid"` and field bounds mirroring current CLI/dashboard validation rules.
- Add cross-field validation (for example date ordering, MA fast/slow relationship when strategy is `ma_crossover`).

2. Build profile persistence module
- Implement:
  - `list_strategy_modeling_profiles(path) -> list[str]`
  - `load_strategy_modeling_profile(path, name) -> StrategyModelingProfile`
  - `save_strategy_modeling_profile(path, name, profile, overwrite=False) -> None`
- Behavior:
  - Missing file => empty store for listing; load-by-name returns clear error.
  - Malformed/unsupported schema version => explicit, actionable error.
  - Atomic writes via temp file + replace.
  - Deterministic output ordering (`sort_keys=True`) for stable diffs/debugging.

3. CLI wiring
- Add `ctx: typer.Context` to `technicals_strategy_model`.
- Build an “effective option payload” layer:
  - Start with loaded profile values if `--profile`.
  - Replace with explicit command-line values only when source is `COMMANDLINE`.
  - Reuse existing helper validators (`_build_strategy_signal_kwargs`, `_build_strategy_filter_config`, ladder builder, etc.) on effective values.
- Save profile on `--save-profile` using effective normalized input payload.
- Keep existing behavior unchanged when no profile flags are used.

4. Dashboard wiring
- Add profile UI controls and helper functions in-page (or thin wrappers delegating to data module).
- Move all run-setting widgets to stable keys in `st.session_state`.
- On `Load Profile`, populate session keys and rerun.
- On `Save Profile`, serialize current widget state to shared profile model and persist through data module.
- Ensure existing run/export flow remains intact.

5. Tests
- New tests: `tests/test_strategy_modeling_profiles.py`
  - empty/missing store behavior
  - save/load roundtrip
  - overwrite guard
  - malformed JSON fallback/error behavior
  - unsupported `schema_version` rejection
- Extend `tests/test_strategy_modeling_cli.py`
  - `--profile` applies stored values
  - explicit CLI values override loaded profile values
  - `--save-profile` writes file with normalized payload
  - existing-name save fails without `--overwrite-profile`
- Extend `tests/portal/test_strategy_modeling_page.py`
  - save profile from dashboard controls writes to profile store
  - load profile repopulates controls and resulting request reflects loaded values
  - dashboard/CLI interop smoke: profile written by one surface can be loaded by the other
- Regression check: existing non-profile strategy-model tests still pass unchanged.

6. Docs
- New doc: `docs/STRATEGY_MODELING_PROFILES.md`
  - purpose, not-financial-advice note
  - file path and schema summary
  - CLI usage examples (`--profile`, `--save-profile`, overwrite)
  - dashboard workflow (load/save)
  - precedence and overwrite rules
- Update links:
  - `docs/index.md`
  - `mkdocs.yml`
  - `docs/TECHNICAL_STRATEGY_MODELING.md` (new flags + behavior)

## Test Scenarios / Acceptance Criteria

1. CLI run with `--save-profile my_fav` creates `config/strategy_modeling_profiles.json` and stores normalized modeling inputs.
2. CLI run with `--profile my_fav` reproduces stored settings when no overriding flags are passed.
3. CLI run with `--profile my_fav --strategy orb` uses `orb` while retaining other profile values.
4. CLI save to existing profile name fails unless `--overwrite-profile` is set.
5. Dashboard can list profiles, load one, and run with loaded values.
6. Dashboard can save current settings to a named profile and enforce overwrite checkbox rule.
7. Profile created in CLI can be loaded in dashboard, and profile created in dashboard can be loaded in CLI.
8. Existing strategy-model workflows without any profile flags/widgets continue to behave exactly as before.

## Assumptions and Defaults
- Shared profile store path is `config/strategy_modeling_profiles.json`.
- Profiles are local-state artifacts and intentionally gitignored.
- Profile payload is run-focused; export/report output preferences stay outside profile scope.
- Overwrite is explicit-only on both surfaces to prevent accidental loss.
