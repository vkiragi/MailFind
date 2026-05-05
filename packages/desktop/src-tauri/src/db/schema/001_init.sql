-- MailFind local schema. Local-first replacement for the previous
-- Supabase/Postgres schema in packages/backend/migrations/mailfindv2_init.sql.

-- ---------- accounts ----------
CREATE TABLE IF NOT EXISTS accounts (
    id            TEXT PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    display_name  TEXT,
    imap_host     TEXT NOT NULL,
    imap_port     INTEGER NOT NULL DEFAULT 993,
    auth_kind     TEXT NOT NULL DEFAULT 'app_password',
    keyring_ref   TEXT NOT NULL,
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- ---------- mailboxes ----------
CREATE TABLE IF NOT EXISTS mailboxes (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    delimiter       TEXT,
    uid_validity    INTEGER,
    uid_next        INTEGER,
    last_seen_uid   INTEGER,
    last_synced_at  TEXT,
    UNIQUE(account_id, name)
);

CREATE INDEX IF NOT EXISTS idx_mailboxes_account ON mailboxes(account_id);

-- ---------- messages ----------
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    account_id      TEXT NOT NULL REFERENCES accounts(id) ON DELETE CASCADE,
    mailbox_id      TEXT REFERENCES mailboxes(id) ON DELETE SET NULL,
    imap_uid        INTEGER,
    rfc822_message_id TEXT,
    thread_id       TEXT,
    subject         TEXT,
    sender          TEXT,
    sender_email    TEXT,
    sender_domain   TEXT,
    recipients      TEXT,
    sent_at         TEXT,
    received_at     TEXT,
    snippet         TEXT,
    body_plain      TEXT,
    body_html       TEXT,
    has_attachments INTEGER NOT NULL DEFAULT 0,
    is_deleted      INTEGER NOT NULL DEFAULT 0,
    raw_size        INTEGER,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(account_id, mailbox_id, imap_uid)
);

CREATE INDEX IF NOT EXISTS idx_messages_account_date ON messages(account_id, received_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_messages_sender_email ON messages(sender_email);
CREATE INDEX IF NOT EXISTS idx_messages_rfc822 ON messages(rfc822_message_id);

-- ---------- chunks (text segments + embedding pointer) ----------
CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    message_id      TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    embedding       BLOB,        -- f32 little-endian vector, optional
    embedding_dim   INTEGER,
    embedding_model TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    UNIQUE(message_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_message ON chunks(message_id);

-- ---------- FTS5 keyword index over chunk text ----------
-- Contentless table; we keep chunk.id <-> chunks_fts.rowid in sync via triggers
-- and translate row ids back to chunk ids at query time. Subject/sender
-- boosting happens in Rust (not in FTS) because those fields live in messages.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='',
    tokenize='porter'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text)
    VALUES('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- ---------- sync state ----------
CREATE TABLE IF NOT EXISTS sync_state (
    account_id    TEXT PRIMARY KEY REFERENCES accounts(id) ON DELETE CASCADE,
    last_sync_at  TEXT,
    last_error    TEXT,
    is_running    INTEGER NOT NULL DEFAULT 0
);

-- ---------- search feedback ----------
CREATE TABLE IF NOT EXISTS search_feedback (
    id          TEXT PRIMARY KEY,
    query_text  TEXT NOT NULL,
    message_id  TEXT REFERENCES messages(id) ON DELETE SET NULL,
    action      TEXT NOT NULL,       -- 'click' | 'positive' | 'negative'
    rank        INTEGER,
    score       REAL,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_feedback_query ON search_feedback(query_text);

-- ---------- app settings (key/value) ----------
CREATE TABLE IF NOT EXISTS app_settings (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

INSERT OR IGNORE INTO app_settings(key, value) VALUES ('embedding_model', 'nomic-embed-text');
INSERT OR IGNORE INTO app_settings(key, value) VALUES ('chat_model', 'qwen2.5:3b-instruct');
INSERT OR IGNORE INTO app_settings(key, value) VALUES ('ollama_endpoint', 'http://127.0.0.1:11434');
