# tests/ — Test Strategy

## Golden rules
- Tests must be **offline** and **deterministic** (no live `yfinance` calls).
- Prefer `tmp_path` for filesystem state (caches, reports, snapshots).
- Use `typer.testing.CliRunner` for CLI coverage.

## Stubbing guidance
- Monkeypatch `options_helper.cli.get_provider` to return a stub provider (preferred).
- Monkeypatch `CandleStore.get_daily_history` for candle-dependent commands.
- If time matters (timestamps/filenames), monkeypatch `options_helper.cli.datetime`.

## What to test
- Edge cases: empty history, missing columns, NaNs, zeros from Yahoo.
- “No crash” invariants: per-symbol failures should not abort multi-symbol runs.
- Saved artifacts: reports, snapshots, metadata correctness.
