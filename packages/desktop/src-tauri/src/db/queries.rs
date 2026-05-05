//! Typed query helpers. Keeps SQL out of business logic and makes it easier to
//! evolve the schema without ripple-touching every module.

use chrono::{DateTime, Utc};
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::AppResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountRow {
    pub id: String,
    pub email: String,
    pub display_name: Option<String>,
    pub imap_host: String,
    pub imap_port: i64,
    pub keyring_ref: String,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct NewAccount {
    pub email: String,
    pub display_name: Option<String>,
    pub imap_host: String,
    pub imap_port: i64,
    pub keyring_ref: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageRow {
    pub id: String,
    pub account_id: String,
    pub thread_id: Option<String>,
    pub subject: String,
    pub sender: String,
    pub sender_email: Option<String>,
    pub recipients: String,
    pub date: String,
    pub snippet: String,
    pub body_preview: String,
}

#[derive(Debug, Clone)]
pub struct NewMessage {
    pub id: String,
    pub account_id: String,
    pub mailbox_id: Option<String>,
    pub imap_uid: Option<i64>,
    pub rfc822_message_id: Option<String>,
    pub thread_id: Option<String>,
    pub subject: Option<String>,
    pub sender: Option<String>,
    pub sender_email: Option<String>,
    pub sender_domain: Option<String>,
    pub recipients: Option<String>,
    pub sent_at: Option<DateTime<Utc>>,
    pub received_at: Option<DateTime<Utc>>,
    pub snippet: Option<String>,
    pub body_plain: Option<String>,
    pub body_html: Option<String>,
    pub has_attachments: bool,
    pub raw_size: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct NewChunk {
    pub message_id: String,
    pub chunk_index: i64,
    pub text: String,
    pub embedding: Option<Vec<f32>>,
    pub embedding_model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ChunkWithEmbedding {
    pub chunk_id: String,
    pub message_id: String,
    pub chunk_index: i64,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ChunkPending {
    pub chunk_id: String,
    pub message_id: String,
    pub text: String,
}

pub fn insert_account(conn: &Connection, new: &NewAccount) -> AppResult<AccountRow> {
    let id = Uuid::new_v4().to_string();
    conn.execute(
        "INSERT INTO accounts (id, email, display_name, imap_host, imap_port, keyring_ref)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            id,
            new.email,
            new.display_name,
            new.imap_host,
            new.imap_port,
            new.keyring_ref,
        ],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO sync_state (account_id) VALUES (?1)",
        params![id],
    )?;
    fetch_account(conn, &id)?
        .ok_or_else(|| crate::error::AppError::Internal("inserted account missing".into()))
}

pub fn fetch_account(conn: &Connection, id: &str) -> AppResult<Option<AccountRow>> {
    let row = conn
        .query_row(
            "SELECT id, email, display_name, imap_host, imap_port, keyring_ref, created_at
             FROM accounts WHERE id = ?1",
            params![id],
            account_from_row,
        )
        .optional()?;
    Ok(row)
}

pub fn list_accounts(conn: &Connection) -> AppResult<Vec<AccountRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, email, display_name, imap_host, imap_port, keyring_ref, created_at
         FROM accounts ORDER BY created_at ASC",
    )?;
    let rows = stmt
        .query_map([], account_from_row)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn delete_account(conn: &Connection, id: &str) -> AppResult<()> {
    conn.execute("DELETE FROM accounts WHERE id = ?1", params![id])?;
    Ok(())
}

fn account_from_row(row: &rusqlite::Row) -> rusqlite::Result<AccountRow> {
    Ok(AccountRow {
        id: row.get(0)?,
        email: row.get(1)?,
        display_name: row.get(2)?,
        imap_host: row.get(3)?,
        imap_port: row.get(4)?,
        keyring_ref: row.get(5)?,
        created_at: row.get(6)?,
    })
}

pub fn insert_message(conn: &Connection, msg: &NewMessage) -> AppResult<bool> {
    let res = conn.execute(
        "INSERT OR IGNORE INTO messages (
            id, account_id, mailbox_id, imap_uid, rfc822_message_id, thread_id,
            subject, sender, sender_email, sender_domain, recipients,
            sent_at, received_at, snippet, body_plain, body_html,
            has_attachments, raw_size
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16,
            ?17, ?18
         )",
        params![
            msg.id,
            msg.account_id,
            msg.mailbox_id,
            msg.imap_uid,
            msg.rfc822_message_id,
            msg.thread_id,
            msg.subject,
            msg.sender,
            msg.sender_email,
            msg.sender_domain,
            msg.recipients,
            msg.sent_at.map(|d| d.to_rfc3339()),
            msg.received_at.map(|d| d.to_rfc3339()),
            msg.snippet,
            msg.body_plain,
            msg.body_html,
            msg.has_attachments as i64,
            msg.raw_size,
        ],
    )?;
    Ok(res > 0)
}

pub fn insert_chunk(conn: &Connection, chunk: &NewChunk) -> AppResult<String> {
    let id = Uuid::new_v4().to_string();
    let embedding_blob = chunk.embedding.as_deref().map(encode_embedding);
    let dim = chunk.embedding.as_ref().map(|v| v.len() as i64);
    conn.execute(
        "INSERT OR IGNORE INTO chunks (
            id, message_id, chunk_index, text, embedding, embedding_dim, embedding_model
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            id,
            chunk.message_id,
            chunk.chunk_index,
            chunk.text,
            embedding_blob,
            dim,
            chunk.embedding_model,
        ],
    )?;
    Ok(id)
}

pub fn update_chunk_embedding(
    conn: &Connection,
    chunk_id: &str,
    embedding: &[f32],
    model: &str,
) -> AppResult<()> {
    conn.execute(
        "UPDATE chunks
         SET embedding = ?1, embedding_dim = ?2, embedding_model = ?3
         WHERE id = ?4",
        params![encode_embedding(embedding), embedding.len() as i64, model, chunk_id],
    )?;
    Ok(())
}

pub fn pending_embedding_chunks(
    conn: &Connection,
    limit: i64,
) -> AppResult<Vec<ChunkPending>> {
    let mut stmt = conn.prepare(
        "SELECT id, message_id, text
         FROM chunks
         WHERE embedding IS NULL
         ORDER BY created_at ASC
         LIMIT ?1",
    )?;
    let rows = stmt
        .query_map(params![limit], |row| {
            Ok(ChunkPending {
                chunk_id: row.get(0)?,
                message_id: row.get(1)?,
                text: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn all_chunks_with_embeddings(conn: &Connection) -> AppResult<Vec<ChunkWithEmbedding>> {
    let mut stmt = conn.prepare(
        "SELECT id, message_id, chunk_index, text, embedding
         FROM chunks WHERE embedding IS NOT NULL",
    )?;
    let rows = stmt
        .query_map([], |row| {
            let blob: Vec<u8> = row.get(4)?;
            Ok(ChunkWithEmbedding {
                chunk_id: row.get(0)?,
                message_id: row.get(1)?,
                chunk_index: row.get(2)?,
                text: row.get(3)?,
                embedding: decode_embedding(&blob),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn fetch_messages(
    conn: &Connection,
    ids: &[String],
) -> AppResult<Vec<MessageRow>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = ids.iter().map(|_| "?").collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT id, account_id, thread_id, subject, sender, sender_email,
                recipients, received_at, snippet, body_plain
         FROM messages WHERE id IN ({})",
        placeholders
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(params_from_iter(ids.iter()), message_from_row)?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

pub fn search_fts(
    conn: &Connection,
    query: &str,
    limit: i64,
) -> AppResult<Vec<(String, f64)>> {
    let safe = sanitize_fts_query(query);
    if safe.is_empty() {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT c.message_id, bm25(chunks_fts) AS rank
         FROM chunks_fts
         JOIN chunks c ON c.rowid = chunks_fts.rowid
         WHERE chunks_fts MATCH ?1
         ORDER BY rank ASC
         LIMIT ?2",
    )?;
    let rows = stmt
        .query_map(params![safe, limit], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn message_from_row(row: &rusqlite::Row) -> rusqlite::Result<MessageRow> {
    let body: Option<String> = row.get(9)?;
    let snippet: Option<String> = row.get(8)?;
    let snippet_value = snippet.unwrap_or_default();
    let body_preview = match &body {
        Some(b) => truncate_str(b, 360),
        None => String::new(),
    };
    Ok(MessageRow {
        id: row.get(0)?,
        account_id: row.get(1)?,
        thread_id: row.get(2)?,
        subject: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        sender: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        sender_email: row.get(5)?,
        recipients: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
        date: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
        snippet: snippet_value,
        body_preview,
    })
}

pub fn message_count(conn: &Connection) -> AppResult<i64> {
    let count: i64 =
        conn.query_row("SELECT COUNT(*) FROM messages", [], |row| row.get(0))?;
    Ok(count)
}

pub fn embedded_message_count(conn: &Connection) -> AppResult<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT message_id) FROM chunks WHERE embedding IS NOT NULL",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

pub fn account_message_count(conn: &Connection, account_id: &str) -> AppResult<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM messages WHERE account_id = ?1",
        params![account_id],
        |row| row.get(0),
    )?;
    Ok(count)
}

pub fn account_embedded_count(conn: &Connection, account_id: &str) -> AppResult<i64> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(DISTINCT m.id)
         FROM messages m JOIN chunks c ON c.message_id = m.id
         WHERE m.account_id = ?1 AND c.embedding IS NOT NULL",
        params![account_id],
        |row| row.get(0),
    )?;
    Ok(count)
}

pub fn upsert_sync_status(
    conn: &Connection,
    account_id: &str,
    is_running: bool,
    last_error: Option<&str>,
    finished: bool,
) -> AppResult<()> {
    if finished {
        conn.execute(
            "INSERT INTO sync_state (account_id, last_sync_at, last_error, is_running)
             VALUES (?1, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?2, ?3)
             ON CONFLICT(account_id) DO UPDATE SET
                last_sync_at = excluded.last_sync_at,
                last_error   = excluded.last_error,
                is_running   = excluded.is_running",
            params![account_id, last_error, is_running as i64],
        )?;
    } else {
        conn.execute(
            "INSERT INTO sync_state (account_id, last_error, is_running)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(account_id) DO UPDATE SET
                last_error = excluded.last_error,
                is_running = excluded.is_running",
            params![account_id, last_error, is_running as i64],
        )?;
    }
    Ok(())
}

pub fn fetch_sync_status(
    conn: &Connection,
    account_id: &str,
) -> AppResult<Option<(Option<String>, bool, Option<String>)>> {
    let row = conn
        .query_row(
            "SELECT last_sync_at, is_running, last_error FROM sync_state WHERE account_id = ?1",
            params![account_id],
            |r| {
                Ok((
                    r.get::<_, Option<String>>(0)?,
                    r.get::<_, i64>(1)? != 0,
                    r.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .optional()?;
    Ok(row)
}

pub fn get_setting(conn: &Connection, key: &str) -> AppResult<Option<String>> {
    let v = conn
        .query_row(
            "SELECT value FROM app_settings WHERE key = ?1",
            params![key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(v)
}

pub fn set_setting(conn: &Connection, key: &str, value: &str) -> AppResult<()> {
    conn.execute(
        "INSERT INTO app_settings(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value,
            updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
        params![key, value],
    )?;
    Ok(())
}

/// Encodes a `&[f32]` as little-endian bytes for storage in a BLOB column.
pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for f in v {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf
}

/// Inverse of `encode_embedding`. Silently truncates trailing partial floats.
pub fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let arr = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(arr));
    }
    out
}

/// FTS5's MATCH syntax has reserved characters; the simplest reliable approach
/// is to strip non-alphanumerics and OR the remaining tokens together.
fn sanitize_fts_query(query: &str) -> String {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|t| {
            t.chars()
                .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
                .collect::<String>()
        })
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"*", t))
        .collect();
    tokens.join(" OR ")
}

fn truncate_str(s: &str, n: usize) -> String {
    let mut iter = s.char_indices();
    match iter.nth(n) {
        Some((i, _)) => s[..i].to_string(),
        None => s.to_string(),
    }
}
