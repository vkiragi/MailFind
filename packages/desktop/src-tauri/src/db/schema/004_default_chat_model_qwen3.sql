-- Switch the default chat model to qwen3:8b. It follows the recency/date
-- scaffolding in the Ask prompt that qwen2.5:7b ignored (correctly refusing to
-- call a 2-year-old email "recent"), at the cost of ~7.6GB RAM (16GB machines).
-- One-time upgrade: version-gated, so a user's later manual choice is preserved.
INSERT INTO app_settings(key, value) VALUES ('chat_model', 'qwen3:8b')
ON CONFLICT(key) DO UPDATE SET value = excluded.value;
