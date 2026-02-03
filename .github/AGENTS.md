# .github/ — CI, Automation, and Repo Hygiene

## Purpose
GitHub configuration lives here (Actions, issue templates, etc.).

## Recommended CI (add as needed)
- Lint/format: `ruff` (and `ruff format` if enabled)
- Tests: `pytest`
- Optional: type checking (`pyright` or `mypy`)

## Philosophy
- CI should be **fast** and **deterministic**. Never hit live `yfinance`.
- Prefer caching pip and reusing the same commands developers run locally.

## When changing CI
- Keep Python versions aligned with `pyproject.toml` (`>=3.10`).
- Ensure `pip install -e "[dev]"` works in a clean environment.
