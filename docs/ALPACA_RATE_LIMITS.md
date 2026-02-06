# Alpaca rate limits (monitoring)

Alpaca enforces **per-endpoint rate limits**. The most reliable way to track your current usage is to read the
**rate-limit headers** Alpaca returns on each HTTP response (for example: limit, remaining, reset time). This repo
can optionally log those headers for every request made via `alpaca-py`. **Not financial advice.**

## Enable logging

Set an env var for the process running `options-helper`:

```bash
export OH_ALPACA_LOG_RATE_LIMITS=1
```

Then run any command that hits Alpaca (examples):

```bash
options-helper ingest candles
options-helper ingest options-bars --max-underlyings 1 --max-contracts 50
```

This emits `ALPACA_RATELIMIT ...` lines into the per-command log file under `data/logs/{YYYY-MM-DD}/`
(or your `--log-dir/{YYYY-MM-DD}/`).

## Inspect usage

The easiest way is the built-in debug helper:

```bash
options-helper debug rate-limits
```

It finds the most recent log in `data/logs/` (including dated subfolders) and prints the last seen `remaining`, `limit`,
and `reset_at`, plus a tail
of recent snapshots.

If you prefer doing it manually:

```bash
rg "ALPACA_RATELIMIT" data/logs -g '*.log' | tail -n 20
```

## What the fields mean

Each snapshot is derived from response headers:

- `limit`: request budget for the rate-limit window.
- `remaining`: remaining requests in the current window (after that response).
- `reset_epoch`: window reset time (UTC epoch seconds, best-effort).
- `reset_in_s`: seconds until reset (best-effort).
