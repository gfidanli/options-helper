# Logging

This tool is for informational/educational use only and is not financial advice.

## Where logs go

By default, CLI logs are written under `data/logs/` and partitioned by **America/Chicago** date:

- `data/logs/{YYYY-MM-DD}/*.log`

The log directory is gitignored (local state).
When a command runs with `--log-dir`, any legacy flat logs at the base level (for example
`data/logs/*.log`) are auto-moved into the appropriate date folder.

## CLI flags

### `--log-dir`

`--log-dir` sets the **base** directory (default `data/logs/`). Logs are written under a date subfolder.

Example:

```bash
options-helper --log-dir /tmp/options-helper-logs watchlists list
```

Writes to:

- `/tmp/options-helper-logs/{YYYY-MM-DD}/...`

### `--log-path`

`--log-path` writes logs to a specific file (append mode) and overrides per-run file generation.
This is useful when you want **one combined log file** for a job (for example: cron scripts).

Example:

```bash
RUN_DATE="$(TZ=America/Chicago date +%F)"
options-helper --log-path "data/logs/${RUN_DATE}/briefing.log" briefing portfolio.json
```

## Useful snippets

Tail today’s briefing log:

```bash
RUN_DATE="$(TZ=America/Chicago date +%F)"
tail -n 200 "data/logs/${RUN_DATE}/briefing.log"
```

Search recent rate-limit snapshots across dated logs:

```bash
rg "ALPACA_RATELIMIT" data/logs -g '*.log' | tail -n 50
```

Or use the helper:

```bash
options-helper debug rate-limits
```
