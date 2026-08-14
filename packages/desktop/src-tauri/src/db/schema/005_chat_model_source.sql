-- Introduce `chat_model_source` to distinguish a user's explicit model choice
-- from an auto-derived default. Migration 004 blanket force-set `chat_model` to
-- qwen3:8b for everyone (the 16GB hard floor this feature removes); marking
-- existing installs 'auto' lets startup auto-pick re-derive a RAM-appropriate
-- model over that force-set. `set_chat_model` flips this to 'user', after which
-- the choice is never auto-overridden. INSERT OR IGNORE so a value already
-- present (e.g. from a later run) is preserved.
INSERT OR IGNORE INTO app_settings(key, value) VALUES ('chat_model_source', 'auto');
