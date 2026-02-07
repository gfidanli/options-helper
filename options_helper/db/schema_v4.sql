-- options-helper DuckDB schema v4
-- Adds candle metadata state for max-backfill completion.

ALTER TABLE candles_meta
  ADD COLUMN IF NOT EXISTS max_backfill_complete BOOLEAN;

UPDATE candles_meta
SET max_backfill_complete = FALSE
WHERE max_backfill_complete IS NULL;

ALTER TABLE candles_meta
  ALTER COLUMN max_backfill_complete SET DEFAULT FALSE;
