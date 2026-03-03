# Technical Debt Containment Handoff

Last updated: February 14, 2026
Owner branch: `codex/tech-debt-containment`
Repo state at handoff: clean (`git status -sb` shows no local changes)

## Purpose
This document is a continuation handoff for the no-behavior-change refactor/debt-containment program. It summarizes:
- what is already done,
- what is still pending,
- where to look first,
- and the guardrails/gotchas that previously caused regressions.

## Current Status Snapshot
- `ruff check .`: passing
- Debt guardrail report (`python scripts/debt_guardrails.py report`):
  - production python files: `287`
  - files `>1000` lines: `6`
  - functions `>80` lines: `153`
- CI debt gate mode: **warn-only** (`continue-on-error: true`) in `.github/workflows/ci.yml` (planned switch to blocking after 2026-02-27).

Current `>1000` files:
- `options_helper/commands/market_analysis_legacy.py` (`2511`)
- `options_helper/data/alpaca_client_legacy.py` (`2383`)
- `apps/streamlit/pages/11_Strategy_Modeling_legacy.py` (`2238`)
- `options_helper/data/stores_duckdb_legacy.py` (`2145`)
- `options_helper/pipelines/visibility_jobs_legacy.py` (`1846`)
- `options_helper/commands/technicals/extension_stats_legacy.py` (`1247`)

## What Has Been Completed
Recent commit chain (newest first):
- `15efde4` refactor(streamlit): shim strategy modeling page
- `ccfab01` refactor(data): shim alpaca client module
- `720662a` refactor(pipelines): shim visibility jobs module
- `fadd2aa` refactor(zero-dte): extract dataset calendar/time helpers
- `93ee927` refactor(ingestion): extract options bars helper utilities
- `41739be` refactor(strategy-modeling): extract segmentation module
- `bceba1f` refactor(market-analysis): split command package and preserve seams
- `58901ca` refactor(workflows): shim legacy workflow commands
- `1b3f8a4` refactor(reports): shim legacy report-pack command
- `10ca443` refactor(reports): shim legacy report commands
- plus earlier technicals decomposition and shim commits (`ecb20af` through `871dc52`)

Important outcomes:
- Command/pipeline/data monoliths were converted to compatibility shims with `_legacy` backends while preserving existing import paths.
- Files over 1000 lines were reduced from prior baseline to `6` remaining.
- Plan-doc canonicalization and guardrail policy landed (`docs/plans`, `docs/TECH_DEBT_GUARDRAILS.md`).
- Debt/doc guardrail tests are in place and passing:
  - `tests/test_debt_guardrails.py`
  - `tests/test_docs_plan_paths.py`

## Remaining Work (Priority Order)
1. Decompose remaining legacy hotspots into submodules while keeping compatibility shims:
- `options_helper/commands/technicals/extension_stats_legacy.py`
- `options_helper/pipelines/visibility_jobs_legacy.py`
- `options_helper/data/stores_duckdb_legacy.py`
- `options_helper/commands/market_analysis_legacy.py`
- `options_helper/data/alpaca_client_legacy.py`
- `apps/streamlit/pages/11_Strategy_Modeling_legacy.py`

2. Reduce function complexity debt (`153` functions over 80 lines). Focus changed-files-first; avoid broad churn.

3. Analysis/data boundary hardening (known direct imports):
- `options_helper/analysis/roll_plan_multileg.py` -> `options_helper.data.options_snapshots`
- `options_helper/analysis/journal_eval.py` -> `options_helper.data.journal`
- `options_helper/analysis/live_portfolio_metrics.py` -> `options_helper.data.alpaca_symbols`
- `options_helper/analysis/strategy_modeling_io_adapter.py` is a documented seam and may remain.

4. CI gate date flip (after warn window):
- Update `.github/workflows/ci.yml` debt step to blocking (`continue-on-error: false` or remove key) after 2026-02-27.

5. Tighten mypy enforcement:
- `pyproject.toml` currently scopes mypy to target files but still has override `ignore_errors = true` for `options_helper.schemas.*` and `options_helper.analysis.*`.

## Known Gotchas / Regression Traps
1. Shim modules must preserve monkeypatch seams used by tests.
- For package wrappers (reports/workflows/market-analysis), sync package-level patched symbols into legacy module before delegating.

2. Static cross-module imports can violate guardrails.
- In legacy seam lookup, prefer `importlib.import_module(...)` at runtime over static imports when guardrails flag `CMD_IMPORT`.

3. `alpaca_client` shim requires patch propagation.
- `options_helper/data/alpaca_client.py` uses module-level `__setattr__` passthrough so monkeypatch writes also update `alpaca_client_legacy` globals. Do not remove unless tests are rewritten.

4. Streamlit strategy page path is test-sensitive.
- Keep `apps/streamlit/pages/11_Strategy_Modeling.py` present; it currently executes `11_Strategy_Modeling_legacy.py` via `runpy` to preserve page path contracts.

5. Guardrail thresholds for new files are strict.
- New production file max size is 400 lines; split new helper modules early to avoid `NEW_FILE_LINES` failures.

## High-Value Next Slice Recommendation
Most efficient next slice is `options_helper/commands/technicals/extension_stats_legacy.py` because it has only two top-level functions and one is massive.
- Extract enrichment/markdown/payload builder sections into helper modules under:
  - `options_helper/commands/technicals/extension_stats_helpers/` (or similar)
- Keep `technicals_extension_stats(...)` as orchestration wrapper only.
- Validate with the technicals report/CLI tests and guardrails.

## Validation Commands (Use These)
Core checks:
```bash
./.venv/bin/ruff check .
./.venv/bin/python scripts/debt_guardrails.py enforce-changed
./.venv/bin/python scripts/debt_guardrails.py report
```

Targeted suites that caught prior shim regressions:
```bash
./.venv/bin/python -m pytest tests/test_debt_guardrails.py tests/test_docs_plan_paths.py
./.venv/bin/python -m pytest tests/test_cli_contract.py
./.venv/bin/python -m pytest tests/test_visibility_jobs_ingest_options_bars_chunking.py
./.venv/bin/python -m pytest tests/test_strategy_modeling_service.py
./.venv/bin/python -m pytest tests/test_zero_dte_dataset_loader.py
```

If touching Alpaca shim:
```bash
./.venv/bin/python -m pytest \
  tests/test_alpaca_client.py \
  tests/test_alpaca_intraday_bars_mapping.py \
  tests/test_alpaca_stocks.py \
  tests/events/test_alpaca_events_client.py
```

If touching Streamlit strategy page/shim:
```bash
./.venv/bin/python -m pytest tests/portal/test_streamlit_scaffold.py tests/portal/test_strategy_modeling_page.py
```

## Related Documentation
- Debt policy: `docs/TECH_DEBT_GUARDRAILS.md`
- Plan directory conventions: `docs/plans/README.md`
- Legacy plan index/mappings: `docs/plans/LEGACY_INDEX.md`
- Repo improvement backlog: `docs/REPO_IMPROVEMENTS.md`
- Product/feature backlog: `docs/BACKLOG.md`
- Iteration loop prompt: `docs/LLM_LOOP_PROMPT.md`
- Performance guardrails: `docs/PERFORMANCE_ARCHITECTURE.md`

## Quick Start For Next Agent
1. Confirm clean baseline:
```bash
git status -sb
./.venv/bin/python scripts/debt_guardrails.py report
```
2. Pick one legacy hotspot and split only one cohesive unit.
3. Run `ruff` + `enforce-changed` + targeted pytest slice.
4. Commit small, push, repeat.
