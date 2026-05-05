//! Email fixture ingestion. Lets the app load `.eml` files (or directories of
//! them) before IMAP sync is wired in. This is what makes search/Q&A
//! demonstrable without an iCloud account configured.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::db::queries::{self, NewChunk};
use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::mail::parser::{compact_text, parse_rfc822};
use crate::search::chunking;

#[derive(Debug, Default, Serialize)]
pub struct IngestReport {
    pub imported: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

/// Ingests a file or recursively a directory. Each parsed message is stored
/// (subject/sender/body) and split into chunks ready for embedding.
pub fn ingest_path(db: &Database, account_id: &str, path: &Path) -> AppResult<IngestReport> {
    let mut report = IngestReport::default();
    let files = collect_eml_files(path)?;
    if files.is_empty() {
        return Err(AppError::InvalidInput(format!(
            "no .eml files found at {}",
            path.display()
        )));
    }
    for f in files {
        match ingest_one(db, account_id, &f) {
            Ok(true) => report.imported += 1,
            Ok(false) => report.skipped += 1,
            Err(e) => report.errors.push(format!("{}: {}", f.display(), e)),
        }
    }
    Ok(report)
}

fn collect_eml_files(path: &Path) -> AppResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    if path.is_file() {
        out.push(path.to_path_buf());
        return Ok(out);
    }
    if path.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir() {
                out.extend(collect_eml_files(&p)?);
            } else if matches_eml(&p) {
                out.push(p);
            }
        }
        return Ok(out);
    }
    Err(AppError::NotFound(format!("path not found: {}", path.display())))
}

fn matches_eml(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            e == "eml" || e == "msg" || e == "txt"
        })
        .unwrap_or(false)
}

fn ingest_one(db: &Database, account_id: &str, file: &Path) -> AppResult<bool> {
    let bytes = fs::read(file)?;
    let parsed = parse_rfc822(&bytes)?;

    // Build chunked text up front so we can wrap insertion in a single transaction.
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

    let new_msg = parsed.into_new_message(account_id, None, None);
    let message_id = new_msg.id.clone();

    let mut handle = db.write()?;
    let tx = handle.transaction()?;
    let inserted = queries::insert_message(&tx, &new_msg)?;
    if !inserted {
        // Already imported (matched on UNIQUE constraints).
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
