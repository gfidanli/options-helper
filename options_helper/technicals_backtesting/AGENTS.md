# options_helper/technicals_backtesting/ — Backtesting Engine Conventions

## Purpose
This package is the repo's **deterministic, offline** technical indicator + backtesting subsystem.
It powers:
- indicator snapshots (ATR/RSI/Z-score/BB/extension percentiles)
- extension-tail statistics and artifact generation
- walk-forward evaluation and parameter optimization

## Golden rules
- **No network.** Inputs are candles already on disk (or fixtures in tests).
- **Deterministic outputs.** Same inputs + config must yield identical results.
- **Stable data contracts.** If you change indicator semantics, bump schema/version fields and update docs/tests.
- **Fail soft.** For insufficient history, emit `None`/warnings rather than raising, unless the caller explicitly requested strict mode.
- **No lookahead bias.** If a signal is known at close of bar `t`, entry/forward metrics must anchor from bar `t+1` open (including resampled timeframes).

## Where things live
- `indicators/`: indicator implementations and helpers (pure functions)
- `strategies/`: strategy definitions (pure; parameterized)
- `backtest/`: runner, walk-forward, optimization, constraints
- `snapshot.py` (or similar): compact per-symbol “technical snapshot” object used by reports/briefing

## Config conventions
- The canonical config is `config/technical_backtesting.yaml`.
- The contract for that file is `config/technical_backtesting.schema.json`.
- When adding config fields:
  - update the JSON schema,
  - add sensible defaults,
  - add a short note to `docs/technical_backtesting/CONFIG_SCHEMA.md`.

## Indicator semantics
- Use a **timezone-naive** `DatetimeIndex` for candles.
- Prefer **Close** for most indicators unless explicitly designed otherwise.
- Keep windowing consistent with docs:
  - RSI smoothing method, ATR definition, extension normalization, etc.
- If a definition is ambiguous, align with `docs/technical_backtesting/INDICATORS.md` and existing tests.
- For event studies/backtests, document both signal timestamp and tradable entry timestamp to avoid implicit close-to-close assumptions.

## Performance guidance
- Prefer vectorized pandas/numpy operations.
- Avoid per-row Python loops in hot paths.
- When computing rolling stats, be explicit about `min_periods`.

## Testing expectations
- Add unit tests for:
  - numeric correctness on a small synthetic series,
  - edge cases (NaNs, short series, constant series),
  - schema/stability (golden JSON where appropriate).
- Avoid flaky tests: no reliance on current date/time.

## UX expectations
- Expose user-facing artifacts through existing CLI commands (or add a new subcommand only when necessary).
- Artifacts should be both human-readable (Markdown) and LLM-friendly (JSON) where appropriate.
