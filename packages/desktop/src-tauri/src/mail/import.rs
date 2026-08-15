//! Local mail import: Apple Mail (`.emlx`) and standard `mbox` files.
//!
//! This is the "instant value on first launch" path from the PRD — it reads
//! mail that already lives on disk so the user can search/ask immediately
//! without waiting on a slow, rate-limited IMAP sync.
//!
//! Apple Mail does **not** store a flat `mbox`. It keeps each message as a
//! separate `.emlx` file inside `.mbox` *bundle directories* under
//! `~/Library/Mail/V*/`. An `.emlx` file is: a decimal byte-count line, then
//! exactly that many bytes of RFC822 message, then an XML plist of flags.
//! We also support genuine `mbox` files (exported mailboxes, Thunderbird,
//! etc.) so the `.mbox` entry in the import file-picker is no longer a lie.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;
use tauri::{AppHandle, Emitter};

use crate::db::queries::{self, NewChunk};
use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::mail::parser::{build_chunk_input, parse_rfc822};
use crate::search::chunking;

/// Tauri event channel for Apple Mail import progress.
pub const IMPORT_PROGRESS_EVENT: &str = "import:progress";

/// How often (in messages) to emit a progress event during a large import.
const PROGRESS_EVERY: usize = 25;

#[derive(Debug, Clone, Default, Serialize)]
pub struct ImportReport {
    pub imported: usize,
    pub skipped: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImportProgress {
    pub account_id: String,
    pub total: usize,
    pub current: usize,
    pub imported: usize,
    pub skipped: usize,
    pub done: bool,
}

// ---------------------------------------------------------------------------
// Apple Mail discovery
// ---------------------------------------------------------------------------

/// Returns `~/Library/Mail` if it exists. The versioned subdirectory (`V2`…
/// `V10`, depending on the macOS release) lives underneath and is handled by
/// the recursive `.emlx` walk, so we don't hard-code a version here.
pub fn apple_mail_dir() -> Option<PathBuf> {
    let mail = dirs::home_dir()?.join("Library").join("Mail");
    mail.is_dir().then_some(mail)
}

/// Recursively collects every `.emlx` / `.partial.emlx` file under `root`.
/// Symlinked directories are not followed, which avoids cycles.
pub fn collect_emlx(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if file_type.is_dir() {
            collect_emlx(&path, out);
        } else if file_type.is_file() && has_extension(&path, "emlx") {
            out.push(path);
        }
    }
}

fn has_extension(path: &Path, ext: &str) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case(ext))
        .unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Format parsing
// ---------------------------------------------------------------------------

/// Extracts the RFC822 message from an Apple Mail `.emlx` file.
///
/// The file starts with a decimal byte count on its own line, followed by the
/// message, followed by an XML plist. If the count line is missing or invalid
/// we fall back to trimming the trailing plist.
pub fn parse_emlx(data: &[u8]) -> AppResult<Vec<u8>> {
    let newline = data
        .iter()
        .position(|&b| b == b'\n')
        .ok_or_else(|| AppError::MailParse("emlx: missing length header".into()))?;

    let header = std::str::from_utf8(&data[..newline])
        .map(str::trim)
        .unwrap_or("");
    if let Ok(len) = header.parse::<usize>() {
        let start = newline + 1;
        let end = start.saturating_add(len).min(data.len());
        if end > start {
            return Ok(data[start..end].to_vec());
        }
    }

    // Fallback: the message ends where the flags plist begins.
    if let Some(pos) = find_subsequence(data, b"\n<?xml") {
        return Ok(data[..pos].to_vec());
    }
    Ok(data.to_vec())
}

/// Splits a standard `mbox` file into individual RFC822 messages.
///
/// Messages are separated by lines beginning with `From ` (the mbox "From_"
/// line). Body lines that were escaped to `>From `, `>>From `, … (mboxrd) are
/// unescaped by dropping one leading `>`. If no separator line is ever seen we
/// treat the whole input as a single message so a misdetected file still
/// imports.
pub fn split_mbox(data: &[u8]) -> Vec<Vec<u8>> {
    let mut messages: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut seen_separator = false;

    for line in lines_with_endings(data) {
        if line.starts_with(b"From ") {
            // The From_ line is a separator, not message content.
            if seen_separator && !current.is_empty() {
                messages.push(std::mem::take(&mut current));
            } else {
                // Discard anything before the first separator.
                current.clear();
            }
            seen_separator = true;
            continue;
        }
        current.extend_from_slice(&unescape_from(line));
    }
    if !current.iter().all(|b| b.is_ascii_whitespace()) {
        messages.push(current);
    }
    messages
}

fn unescape_from(line: &[u8]) -> Vec<u8> {
    let gt = line.iter().take_while(|&&b| b == b'>').count();
    if gt > 0 && line[gt..].starts_with(b"From ") {
        line[1..].to_vec()
    } else {
        line.to_vec()
    }
}

fn lines_with_endings(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, &b) in data.iter().enumerate() {
        if b == b'\n' {
            out.push(&data[start..=i]);
            start = i + 1;
        }
    }
    if start < data.len() {
        out.push(&data[start..]);
    }
    out
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Parses a single raw RFC822 message and stores it (plus its text chunks).
///
/// Returns `Ok(true)` if a new message was stored, `Ok(false)` if it was a
/// duplicate. Deduplication is by `(account_id, rfc822_message_id)`: imported
/// messages carry no `mailbox_id`/`imap_uid`, so the table's UNIQUE constraint
/// (which treats NULLs as distinct) would not catch re-imports on its own.
pub fn store_rfc822(db: &Database, account_id: &str, raw: &[u8]) -> AppResult<bool> {
    let parsed = parse_rfc822(raw)?;

    if let Some(message_id) = parsed.rfc822_message_id.as_deref() {
        let conn = db.read()?;
        if queries::message_exists_by_rfc822(&conn, account_id, message_id)? {
            return Ok(false);
        }
    }

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

    let new_msg = parsed.into_new_message(account_id, None, None);
    let message_id = new_msg.id.clone();

    let mut handle = db.write()?;
    let tx = handle.transaction()?;
    let inserted = queries::insert_message(&tx, &new_msg)?;
    if !inserted {
        tx.commit()?;
        return Ok(false);
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
    Ok(true)
}

// ---------------------------------------------------------------------------
// Apple Mail import driver
// ---------------------------------------------------------------------------

/// Imports every `.emlx` message found under `~/Library/Mail`, emitting
/// [`IMPORT_PROGRESS_EVENT`] as it goes. Blocking — call from `spawn_blocking`.
pub fn run_apple_mail_import(
    app: &AppHandle,
    db: &Database,
    account_id: &str,
) -> AppResult<ImportReport> {
    let dir = apple_mail_dir().ok_or_else(|| {
        AppError::NotFound(
            "Apple Mail directory (~/Library/Mail) not found on this Mac".into(),
        )
    })?;

    let mut files = Vec::new();
    collect_emlx(&dir, &mut files);
    let total = files.len();
    tracing::info!(account_id, total, "apple mail import starting");

    let mut report = ImportReport::default();
    emit_progress(app, account_id, total, 0, &report, false);

    for (i, path) in files.iter().enumerate() {
        match fs::read(path) {
            Ok(bytes) => match parse_emlx(&bytes).and_then(|raw| store_rfc822(db, account_id, &raw)) {
                Ok(true) => report.imported += 1,
                Ok(false) => report.skipped += 1,
                Err(e) => {
                    tracing::warn!(?e, path = %path.display(), "emlx import error");
                    report.errors += 1;
                }
            },
            Err(e) => {
                tracing::warn!(?e, path = %path.display(), "emlx read error");
                report.errors += 1;
            }
        }
        let done = i + 1;
        if done % PROGRESS_EVERY == 0 || done == total {
            emit_progress(app, account_id, total, done, &report, false);
        }
    }

    emit_progress(app, account_id, total, total, &report, true);
    tracing::info!(
        account_id,
        imported = report.imported,
        skipped = report.skipped,
        errors = report.errors,
        "apple mail import finished"
    );
    Ok(report)
}

fn emit_progress(
    app: &AppHandle,
    account_id: &str,
    total: usize,
    current: usize,
    report: &ImportReport,
    done: bool,
) {
    let payload = ImportProgress {
        account_id: account_id.to_string(),
        total,
        current,
        imported: report.imported,
        skipped: report.skipped,
        done,
    };
    if let Err(e) = app.emit(IMPORT_PROGRESS_EVENT, &payload) {
        tracing::warn!(?e, "failed to emit import progress");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MSG_A: &[u8] = b"From: Alice <alice@example.com>\r\n\
Subject: First\r\n\
Message-ID: <a@example.com>\r\n\
\r\n\
Body of the first message.\r\n";

    #[test]
    fn parse_emlx_uses_byte_count_header() {
        let mut emlx = format!("{}\n", MSG_A.len()).into_bytes();
        emlx.extend_from_slice(MSG_A);
        emlx.extend_from_slice(b"<?xml version=\"1.0\"?><plist></plist>");

        let parsed = parse_emlx(&emlx).unwrap();
        assert_eq!(parsed, MSG_A);
    }

    #[test]
    fn parse_emlx_falls_back_to_plist_marker() {
        // No valid count line — the message still ends at the plist.
        let mut emlx = b"not-a-number\n".to_vec();
        emlx.extend_from_slice(MSG_A);
        emlx.extend_from_slice(b"\n<?xml version=\"1.0\"?><plist></plist>");

        let parsed = parse_emlx(&emlx).unwrap();
        assert!(parsed.windows(5).any(|w| w == b"Alice"));
        assert!(!parsed.windows(5).any(|w| w == b"<?xml"));
    }

    #[test]
    fn split_mbox_separates_messages() {
        let mbox = b"From alice@example.com Mon Jan  1 00:00:00 2024\n\
Subject: One\n\
\n\
Body one.\n\
From bob@example.com Mon Jan  1 00:01:00 2024\n\
Subject: Two\n\
\n\
Body two.\n";
        let messages = split_mbox(mbox);
        assert_eq!(messages.len(), 2);
        assert!(messages[0].windows(3).any(|w| w == b"One"));
        assert!(messages[1].windows(3).any(|w| w == b"Two"));
    }

    #[test]
    fn split_mbox_unescapes_quoted_from_lines() {
        let mbox = b"From alice@example.com Mon Jan  1 00:00:00 2024\n\
Subject: Escaped\n\
\n\
>From the desk of Alice\n";
        let messages = split_mbox(mbox);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].windows(5).any(|w| w == b"From "));
        assert!(!messages[0].windows(6).any(|w| w == b">From "));
    }

    #[test]
    fn split_mbox_handles_non_mbox_as_single_message() {
        let messages = split_mbox(MSG_A);
        assert_eq!(messages.len(), 1);
    }
}
