-- Switch the default chat model to qwen2.5:7b-instruct. Beat granite4.1:3b and
-- llama3.1:8b in a bake-off on recency-aware synthesis while using less RAM than
-- llama (~5.7GB loaded). One-time upgrade: runs once (version-gated), so a
-- user's later manual choice is preserved on subsequent launches.
INSERT INTO app_settings(key, value) VALUES ('chat_model', 'qwen2.5:7b-instruct')
ON CONFLICT(key) DO UPDATE SET value = excluded.value;
