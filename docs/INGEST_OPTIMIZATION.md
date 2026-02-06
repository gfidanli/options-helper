# Ingestion Optimization Playbook (Alpaca + DuckDB)

This captures the current ingestion tuning strategy for endpoint-level performance and database throughput.
It is focused on `ingest options-bars` and should be treated as an operational playbook. **Not financial advice.**

Last updated: 2026-02-06.

## Why this exists

A single global "concurrency" number is not enough for options ingestion. Different endpoints have different limits,
response shapes, and bottlenecks:

- `GET /v2/options/contracts` (trading client) is low-limit and easy to overrun.
- `GET /v1beta1/options/bars` (option client) is high-limit and tends to hit local CPU/DB bottlenecks first.

The best results come from tuning each endpoint independently and validating with logs.

## What We Optimized

| Area | Status | What was implemented |
|---|---|---|
| Contracts endpoint (`/v2/options/contracts`) | Optimized | Separate throttle (`--contracts-max-rps`, default `2.5`) and explicit rate-limit logging support. |
| Bars endpoint (`/v1beta1/options/bars`) | Optimized | Independent concurrency + RPS controls (`--bars-concurrency`, `--bars-max-rps`) plus HTTP pool overrides (`--alpaca-http-pool-maxsize`, `--alpaca-http-pool-connections`). |
| Bars fetch benchmarking | Optimized | `--fetch-only` mode to isolate network/provider throughput from DuckDB write cost. |
| Resume/skip logic | Optimized | Uses `option_bars_meta` coverage so we avoid re-fetching contracts already attempted/covered. |
| DuckDB write path for bars/meta | Optimized | Batched flushes and single-transaction `apply_write_batch(...)` for bars + meta updates. |

## What Still Needs Optimization

| Area | Current gap | Direction |
|---|---|---|
| Adaptive rate control | Tuning is manual today. | Auto-adjust from response headers and back off immediately on first 429. |
| Contracts discovery concurrency | Discovery is intentionally conservative because limit is lower. | Keep low RPS, but evaluate bounded parallelism by underlying when provider behavior allows it. |
| Endpoint-specific retry policy | Retries are generic. | Use per-endpoint jitter/backoff tuned to each reset window and limit. |
| DuckDB write pressure | Single-writer model can still cap throughput at high fetch rates. | Add async write queue and/or larger staged write chunks. |
| Historical benchmark registry | No canonical benchmark table yet. | Persist benchmark runs and pick defaults from measured p50/p95 throughput. |

## Current Operating Profiles

Use these as starting points and validate with logs in your own account.

1. Conservative baseline (defaults):
   - `--contracts-max-rps 2.5`
   - `--bars-concurrency 8`
   - `--bars-max-rps 30`

2. Throughput profile used in recent tuning:
   - `--contracts-max-rps 2.5`
   - `--bars-concurrency 32`
   - `--bars-max-rps 2000`
   - `--alpaca-http-pool-maxsize 512`
   - `--alpaca-http-pool-connections 512`

3. Calibration rule:
   - Increase bars knobs until first bars `status=429`.
   - Set production target to ~80% of that first-limit point.
   - Re-test after major code/provider changes.

## DuckDB Changes Already Landed

The following are already in code:

- Bulk coverage reads (`coverage_bulk`) so resume checks avoid per-contract query overhead.
- Buffered success/error meta updates controlled by `--bars-write-batch-size`.
- Store-level `apply_write_batch(...)` that writes bars + success meta + error meta in one transaction.
- `--fetch-only` to confirm whether bottlenecks are network/provider or local DB writes.

Relevant code:

- `options_helper/data/ingestion/options_bars.py`
- `options_helper/data/stores_duckdb.py`
- `options_helper/commands/ingest.py`

## DuckDB Improvements We Can Still Add

1. Decoupled writer thread/process:
   - Keep fetch workers hot while a dedicated writer flushes batches to DuckDB.

2. Adaptive batch sizing:
   - Increase/decrease `bars_write_batch_size` based on observed commit latency.

3. Optional staging path for very large runs:
   - Stage bars to local parquet/csv chunks, then bulk-ingest via DuckDB `COPY`/set-based upsert.

4. DB session tuning:
   - Evaluate DuckDB settings (`threads`, memory limit, temp directory) for local hardware.

## Pattern For New Providers Or Databases

When adding a new provider or storage backend, keep the same tuning contract:

1. Provider profile per endpoint:
   - Define endpoint budgets (safe RPS, max tested RPS, reset behavior).
   - Expose separate CLI/runtime knobs for each endpoint class.

2. Storage capability surface:
   - Support `coverage_bulk` for resumability at scale.
   - Support `apply_write_batch` for transactional batch writes.
   - If unsupported, use a compatibility path with smaller batches and clear warnings.

3. Standard benchmark loop:
   - Run with `--fetch-only` first to measure provider ceiling.
   - Run full ingest next to measure storage ceiling.
   - Save benchmark outcomes and update defaults only from measured data.

4. Guardrails:
   - Keep `--resume` enabled by default.
   - Keep `--log-rate-limits` available for any provider that emits limit headers.
   - Preserve `--fail-fast` and safety caps (`--max-*`) for controlled experiments.

## Suggested Next Iteration

Implement an optional auto-tune mode:

- Warm-up at baseline.
- Increase bars concurrency/RPS stepwise.
- Stop at first 429 or sustained write-latency spike.
- Persist recommended settings per provider+endpoint+host profile.
