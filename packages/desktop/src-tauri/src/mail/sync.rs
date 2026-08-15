//! Incremental IMAP sync. v1 covers the iCloud INBOX and pulls everything
//! newer than the last seen UID. Anything more sophisticated (mailbox
//! tracking, deletions, expunge handling) plugs into this same flow without
//! the search/Q&A layers caring.

use std::sync::Arc;

use chrono::{Duration, Utc};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use uuid::Uuid;

use crate::credentials;
use crate::db::queries::{self, AccountRow, NewChunk};
use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::mail::imap_client::{self, ImapConfig};
use crate::mail::parser::{build_chunk_input, parse_rfc822};
use crate::search::chunking;

const PRIMARY_MAILBOX: &str = "INBOX";
// iCloud is finicky about large UID FETCHes that ask for full bodies — keep
// batches small so each FETCH command completes before the server's idle
// timeout fires. The reconnect logic in fetch_uids_batched recovers if a
// batch still fails.
const FETCH_BATCH_SIZE: usize = 20;
pub const SYNC_PROGRESS_EVENT: &str = "sync:progress";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncPhase {
    Connecting,
    Searching,
    Fetching,
    Storing,
    Done,
    Failed,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncProgress {
    pub account_id: String,
    pub phase: SyncPhase,
    /// Total messages discovered for this sync (None until SEARCH completes).
    pub total: Option<usize>,
    /// Messages processed so far (fetched, or stored — depends on phase).
    pub current: usize,
    pub message: Option<String>,
}

fn emit_progress(app: &AppHandle, payload: SyncProgress) {
    if let Err(e) = app.emit(SYNC_PROGRESS_EVENT, &payload) {
        tracing::warn!(?e, "failed to emit sync progress");
    }
}

/// How far back to pull on the *first* sync for an account. Subsequent syncs
/// are always incremental from `last_seen_uid` regardless of window.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncWindow {
    Day,
    Week,
    TwoWeeks,
    Month,
    ThreeMonths,
}

impl SyncWindow {
    fn days(self) -> i64 {
        match self {
            SyncWindow::Day => 1,
            SyncWindow::Week => 7,
            SyncWindow::TwoWeeks => 14,
            SyncWindow::Month => 30,
            SyncWindow::ThreeMonths => 90,
        }
    }

    /// IMAP SEARCH date format: `DD-Mon-YYYY` (e.g. `08-May-2026`).
    fn since_date(self) -> String {
        let cutoff = Utc::now() - Duration::days(self.days());
        cutoff.format("%d-%b-%Y").to_string()
    }
}

impl Default for SyncWindow {
    fn default() -> Self {
        SyncWindow::Month
    }
}

pub struct SyncOutcome {
    pub imported: usize,
    pub skipped: usize,
}

pub async fn run_sync(
    app: AppHandle,
    db: &Arc<Database>,
    account: AccountRow,
    full_resync: bool,
    window: SyncWindow,
) -> AppResult<SyncOutcome> {
    tracing::info!(
        account = %account.email,
        full_resync,
        ?window,
        "sync starting"
    );
    {
        let conn = db.write()?;
        queries::upsert_sync_status(&conn, &account.id, true, None, false)?;
    }

    let result = run_sync_inner(&app, db, &account, full_resync, window).await;
    match &result {
        Ok(outcome) => tracing::info!(
            account = %account.email,
            imported = outcome.imported,
            skipped = outcome.skipped,
            "sync finished"
        ),
        Err(e) => tracing::warn!(account = %account.email, error = %e, "sync failed"),
    }

    {
        let conn = db.write()?;
        let err_msg = result.as_ref().err().map(|e| e.to_string());
        queries::upsert_sync_status(&conn, &account.id, false, err_msg.as_deref(), true)?;
    }
    let phase = if result.is_ok() {
        SyncPhase::Done
    } else {
        SyncPhase::Failed
    };
    let message = result.as_ref().err().map(|e| e.to_string());
    let current = result.as_ref().map(|o| o.imported).unwrap_or(0);
    emit_progress(
        &app,
        SyncProgress {
            account_id: account.id.clone(),
            phase,
            total: None,
            current,
            message,
        },
    );
    result
}

async fn run_sync_inner(
    app: &AppHandle,
    db: &Arc<Database>,
    account: &AccountRow,
    full_resync: bool,
    window: SyncWindow,
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
    let since_date = window.since_date();
    let mailbox_id = ensure_mailbox(db, &account.id, PRIMARY_MAILBOX)?;

    emit_progress(
        app,
        SyncProgress {
            account_id: account.id.clone(),
            phase: SyncPhase::Connecting,
            total: None,
            current: 0,
            message: None,
        },
    );

    // The whole fetch→parse→store pipeline runs inside spawn_blocking, one
    // batch at a time: each fetched batch is persisted (and the UID watermark
    // advanced) before the next FETCH is issued. Memory stays bounded by the
    // batch size, and a failure mid-sync keeps everything already stored —
    // the next sync resumes from the last durable batch instead of
    // re-fetching from scratch.
    let app_for_blocking = app.clone();
    let account_id = account.id.clone();
    let db_for_blocking = Arc::clone(db);
    let outcome = tokio::task::spawn_blocking(move || -> AppResult<SyncOutcome> {
        let mut session = imap_client::connect(&cfg)?;
        emit_progress(
            &app_for_blocking,
            SyncProgress {
                account_id: account_id.clone(),
                phase: SyncPhase::Searching,
                total: None,
                current: 0,
                message: None,
            },
        );
        let uids = imap_client::search_since_date(
            &mut session,
            PRIMARY_MAILBOX,
            &since_date,
            last_seen,
        )?;
        emit_progress(
            &app_for_blocking,
            SyncProgress {
                account_id: account_id.clone(),
                phase: SyncPhase::Fetching,
                total: Some(uids.len()),
                current: 0,
                message: None,
            },
        );

        let mut imported = 0usize;
        let mut skipped = 0usize;
        let mut max_uid = last_seen.unwrap_or(0);
        let mut watermark_written = max_uid;
        let mut processed = 0usize;

        let result = imap_client::fetch_uids_batched(
            &mut session,
            &uids,
            FETCH_BATCH_SIZE,
            |batch, fetched_so_far, total| {
                emit_progress(
                    &app_for_blocking,
                    SyncProgress {
                        account_id: account_id.clone(),
                        phase: SyncPhase::Fetching,
                        total: Some(total),
                        current: fetched_so_far,
                        message: None,
                    },
                );

                processed += batch.len();
                for fetched in batch {
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
                    let combined = build_chunk_input(
                        parsed.subject.as_deref().unwrap_or_default(),
                        parsed.sender_display.as_deref().unwrap_or_default(),
                        parsed.recipients.as_deref().unwrap_or_default(),
                        &body,
                        8000,
                    );
                    let chunks = chunking::split(&combined);
                    let new_msg = parsed.into_new_message(
                        &account_id,
                        Some(mailbox_id.clone()),
                        Some(fetched.uid as i64),
                    );
                    let message_id = new_msg.id.clone();

                    // Cross-source dedup: this message may already exist from a local
                    // Apple Mail import (which carries no mailbox_id/imap_uid, so the
                    // UNIQUE constraint can't catch it). Skip it, but still advance the
                    // UID watermark so we don't re-fetch it next sync.
                    if let Some(rfc822_id) = new_msg.rfc822_message_id.as_deref() {
                        let conn = db_for_blocking.read()?;
                        if queries::message_exists_by_rfc822(&conn, &account_id, rfc822_id)? {
                            skipped += 1;
                            if fetched.uid > max_uid {
                                max_uid = fetched.uid;
                            }
                            continue;
                        }
                    }

                    let mut handle = db_for_blocking.write()?;
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

                // This batch is durable — advance the watermark so a failure
                // in a later batch doesn't force a re-fetch of this one.
                if max_uid > watermark_written {
                    let conn = db_for_blocking.write()?;
                    conn.execute(
                        "UPDATE mailboxes SET last_seen_uid = ?1, last_synced_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                         WHERE account_id = ?2 AND name = ?3",
                        params![max_uid as i64, account_id, PRIMARY_MAILBOX],
                    )?;
                    watermark_written = max_uid;
                }

                emit_progress(
                    &app_for_blocking,
                    SyncProgress {
                        account_id: account_id.clone(),
                        phase: SyncPhase::Storing,
                        total: Some(total),
                        current: processed,
                        message: None,
                    },
                );
                Ok(())
            },
        );
        imap_client::logout(session);
        result?;
        Ok(SyncOutcome { imported, skipped })
    })
    .await
    .map_err(|e| AppError::Internal(format!("sync task join: {e}")))??;

    Ok(outcome)
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
