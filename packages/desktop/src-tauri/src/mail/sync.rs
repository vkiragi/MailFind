//! Incremental IMAP sync. v1 covers the iCloud INBOX and pulls everything
//! newer than the last seen UID. Anything more sophisticated (mailbox
//! tracking, deletions, expunge handling) plugs into this same flow without
//! the search/Q&A layers caring.

use rusqlite::{params, OptionalExtension};
use uuid::Uuid;

use crate::credentials;
use crate::db::queries::{self, AccountRow, NewChunk};
use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::mail::imap_client::{self, FetchedMessage, ImapConfig};
use crate::mail::parser::{compact_text, parse_rfc822};
use crate::search::chunking;

const DEFAULT_BACKFILL: u32 = 200;
const PRIMARY_MAILBOX: &str = "INBOX";

pub struct SyncOutcome {
    pub imported: usize,
    pub skipped: usize,
}

pub async fn run_sync(
    db: &Database,
    account: AccountRow,
    full_resync: bool,
) -> AppResult<SyncOutcome> {
    {
        let conn = db.write()?;
        queries::upsert_sync_status(&conn, &account.id, true, None, false)?;
    }

    let result = run_sync_inner(db, &account, full_resync).await;

    {
        let conn = db.write()?;
        let err_msg = result.as_ref().err().map(|e| e.to_string());
        queries::upsert_sync_status(&conn, &account.id, false, err_msg.as_deref(), true)?;
    }
    result
}

async fn run_sync_inner(
    db: &Database,
    account: &AccountRow,
    full_resync: bool,
) -> AppResult<SyncOutcome> {
    let password = credentials::fetch_password(&account.keyring_ref)?;
    let cfg = ImapConfig {
        host: account.imap_host.clone(),
        port: account.imap_port as u16,
        user: account.email.clone(),
        password,
    };

    let last_seen = if full_resync {
        None
    } else {
        load_last_seen_uid(db, &account.id, PRIMARY_MAILBOX)?
    };

    let messages = tokio::task::spawn_blocking(move || -> AppResult<Vec<FetchedMessage>> {
        let mut session = imap_client::connect(&cfg)?;
        let result = imap_client::fetch_recent(&mut session, PRIMARY_MAILBOX, last_seen, DEFAULT_BACKFILL);
        imap_client::logout(session);
        result
    })
    .await
    .map_err(|e| AppError::Internal(format!("sync task join: {e}")))??;

    if messages.is_empty() {
        return Ok(SyncOutcome {
            imported: 0,
            skipped: 0,
        });
    }

    let mailbox_id = ensure_mailbox(db, &account.id, PRIMARY_MAILBOX)?;
    let mut imported = 0;
    let mut skipped = 0;
    let mut max_uid = last_seen.unwrap_or(0);

    for fetched in messages {
        let parsed = match parse_rfc822(&fetched.raw) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(uid = fetched.uid, ?e, "parse failed; skipping");
                skipped += 1;
                continue;
            }
        };
        let body = parsed
            .body_plain
            .clone()
            .or_else(|| parsed.body_html.clone())
            .unwrap_or_default();
        let header_blurb = format!(
            "Subject: {}\nFrom: {}\nTo: {}\n",
            parsed.subject.clone().unwrap_or_default(),
            parsed.sender_display.clone().unwrap_or_default(),
            parsed.recipients.clone().unwrap_or_default(),
        );
        let combined = format!("{}\n{}", header_blurb, compact_text(&body, 8000));
        let chunks = chunking::split(&combined);
        let new_msg = parsed.into_new_message(&account.id, Some(mailbox_id.clone()), Some(fetched.uid as i64));
        let message_id = new_msg.id.clone();

        let mut handle = db.write()?;
        let tx = handle.transaction()?;
        let inserted = queries::insert_message(&tx, &new_msg)?;
        if !inserted {
            skipped += 1;
            tx.commit()?;
            continue;
        }
        for (i, text) in chunks.into_iter().enumerate() {
            queries::insert_chunk(
                &tx,
                &NewChunk {
                    message_id: message_id.clone(),
                    chunk_index: i as i64,
                    text,
                    embedding: None,
                    embedding_model: None,
                },
            )?;
        }
        tx.commit()?;
        imported += 1;
        if fetched.uid > max_uid {
            max_uid = fetched.uid;
        }
    }

    if max_uid > 0 {
        let conn = db.write()?;
        conn.execute(
            "UPDATE mailboxes SET last_seen_uid = ?1, last_synced_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE account_id = ?2 AND name = ?3",
            params![max_uid as i64, account.id, PRIMARY_MAILBOX],
        )?;
    }

    Ok(SyncOutcome { imported, skipped })
}

fn ensure_mailbox(db: &Database, account_id: &str, name: &str) -> AppResult<String> {
    let conn = db.write()?;
    let existing: Option<String> = conn
        .query_row(
            "SELECT id FROM mailboxes WHERE account_id = ?1 AND name = ?2",
            params![account_id, name],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(id) = existing {
        return Ok(id);
    }
    let id = Uuid::new_v4().to_string();
    conn.execute(
        "INSERT INTO mailboxes (id, account_id, name) VALUES (?1, ?2, ?3)",
        params![id, account_id, name],
    )?;
    Ok(id)
}

fn load_last_seen_uid(
    db: &Database,
    account_id: &str,
    name: &str,
) -> AppResult<Option<u32>> {
    let conn = db.read()?;
    let row: Option<i64> = conn
        .query_row(
            "SELECT last_seen_uid FROM mailboxes WHERE account_id = ?1 AND name = ?2",
            params![account_id, name],
            |r| r.get(0),
        )
        .optional()?;
    Ok(row.and_then(|v| u32::try_from(v).ok()))
}
