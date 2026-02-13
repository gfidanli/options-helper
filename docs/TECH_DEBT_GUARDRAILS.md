# Technical Debt Guardrails

This repo is informational/educational tooling only and is not financial advice.

## Purpose

These rules are mandatory for refactors and new feature work so debt does not grow faster than delivery.

## Scope

Guardrails apply to production Python code under:
- `options_helper/`
- `apps/`
- `scripts/`

Tests and fixtures are exempt from size thresholds, but should still remain readable.

## Hard limits

1. New production Python files: **<= 400 lines**.
2. New production Python functions: **<= 80 lines**.
3. Touched legacy files already > 1,000 lines: size must be non-increasing unless the same PR extracts/splits code.
4. No cross-command imports inside `options_helper/commands/*` except:
   - `options_helper.commands.common`
   - `options_helper.commands.position_metrics`
   - modules ending in `_common`
   - temporary transition modules ending in `_legacy`
5. No direct `options_helper.analysis.* -> options_helper.data.*` imports except explicit adapter seams.

## Explicit adapter seams

Direct analysis-to-data imports are only allowed in these adapter modules:
- `options_helper/analysis/strategy_modeling_io_adapter.py`

If a new seam is required, add it here first (same PR) and include a short rationale.

## Refactor expectations

1. Keep CLI behavior stable during decomposition.
2. Use thin command modules: parse args, call services/analysis, render output.
3. Move shared compute logic into `options_helper/analysis/` or `options_helper/pipelines/`.
4. Keep backward-compatible import shims for one transition cycle when moving modules (`*_legacy.py`).

## CI enforcement model

- `report` mode always runs and prints metrics.
- `enforce-changed` mode checks only changed files and will become blocking after rollout.

## Rollout schedule

- February 13-27, 2026: enforce checks run in warning mode (non-blocking).
- February 28, 2026 onward: `enforce-changed` becomes blocking in CI.

## Required checks before merge

1. `ruff check .`
2. `python scripts/debt_guardrails.py report`
3. `python scripts/debt_guardrails.py enforce-changed`
4. Relevant pytest suites for touched areas
