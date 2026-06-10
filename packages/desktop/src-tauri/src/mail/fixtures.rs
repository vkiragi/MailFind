//! Email file ingestion. Loads `.eml` single messages, `.emlx` Apple Mail
//! files, and `mbox` files (or directories of any of these) into the local
//! store. This is what makes search/Q&A demonstrable without an IMAP account,
//! and backs the "Import file" button in the UI.
//!
//! The actual format parsing and message storage live in [`crate::mail::import`];
//! this module just walks paths and tallies the results.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::mail::import;

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
    let files = collect_mail_files(path)?;
    if files.is_empty() {
        return Err(AppError::InvalidInput(format!(
            "no mail files found at {}",
            path.display()
        )));
    }
    for f in files {
        match ingest_one(db, account_id, &f) {
            Ok((imported, skipped)) => {
                report.imported += imported;
                report.skipped += skipped;
            }
            Err(e) => report.errors.push(format!("{}: {}", f.display(), e)),
        }
    }
    Ok(report)
}

fn collect_mail_files(path: &Path) -> AppResult<Vec<PathBuf>> {
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
                out.extend(collect_mail_files(&p)?);
            } else if is_mail_file(&p) {
                out.push(p);
            }
        }
        return Ok(out);
    }
    Err(AppError::NotFound(format!("path not found: {}", path.display())))
}

fn is_mail_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            matches!(e.as_str(), "eml" | "emlx" | "mbox" | "msg" | "txt")
        })
        .unwrap_or(false)
}

/// Ingests one file, which may contain a single message (`.eml`/`.emlx`) or
/// many (`mbox`). Returns `(imported, skipped)`. A message that fails to parse
/// is counted as skipped rather than aborting the whole file.
fn ingest_one(db: &Database, account_id: &str, file: &Path) -> AppResult<(usize, usize)> {
    let bytes = fs::read(file)?;
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    let raws: Vec<Vec<u8>> = match ext.as_str() {
        "mbox" => import::split_mbox(&bytes),
        "emlx" => vec![import::parse_emlx(&bytes)?],
        // Unknown extension: an mbox starts with a `From ` line, otherwise
        // treat the file as one RFC822 message.
        _ => {
            if bytes.starts_with(b"From ") {
                import::split_mbox(&bytes)
            } else {
                vec![bytes]
            }
        }
    };

    let mut imported = 0;
    let mut skipped = 0;
    for raw in raws {
        if raw.iter().all(|b| b.is_ascii_whitespace()) {
            continue;
        }
        match import::store_rfc822(db, account_id, &raw) {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                tracing::warn!(?e, file = %file.display(), "skipped unparseable message");
                skipped += 1;
            }
        }
    }
    Ok((imported, skipped))
}
