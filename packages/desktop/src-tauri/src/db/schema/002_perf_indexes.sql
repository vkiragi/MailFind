-- Partial index used by the "embedded chunks" counters in queries.rs. Without
-- it, COUNT(DISTINCT message_id) FROM chunks WHERE embedding IS NOT NULL does
-- a full scan over hundreds of thousands of rows on every sync_status call.
CREATE INDEX IF NOT EXISTS idx_chunks_embedded_message
    ON chunks(message_id) WHERE embedding IS NOT NULL;
