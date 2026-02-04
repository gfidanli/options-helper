# tests/ — Test Strategy

## Golden rules
- Tests must be **offline** and **deterministic** (no live `yfinance` calls).
- Prefer `tmp_path` for filesystem state (caches, reports, snapshots).
- Use `typer.testing.CliRunner` for CLI coverage.

## Stubbing guidance
- Prefer monkeypatching `options_helper.cli_deps.build_provider` (stable seam) to return a stub provider.
- Avoid new tests that monkeypatch `options_helper.cli.*` imported aliases (they change during CLI refactors).
- Monkeypatch `options_helper.data.candles.CandleStore.get_daily_history` for candle-dependent commands.
- If time matters (timestamps/filenames), monkeypatch the module-local `datetime` used by the command module under test
  (or `options_helper.schemas.common.utc_now` when artifacts use it).

## What to test
- Edge cases: empty history, missing columns, NaNs, zeros from Yahoo.
- “No crash” invariants: per-symbol failures should not abort multi-symbol runs.
- Saved artifacts: reports, snapshots, metadata correctness.
