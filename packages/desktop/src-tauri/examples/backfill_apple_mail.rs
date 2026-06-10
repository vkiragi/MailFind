//! One-off: backfill a single account from `~/Library/Mail/*.emlx` without
//! going through the Tauri UI. Run with:
//!
//!   cargo run --example backfill_apple_mail -- <account_id>
//!
//! Make sure the desktop app is NOT running so SQLite write locks don't fight.

use std::env;
use std::fs;

use mailfind_lib::db::Database;
use mailfind_lib::mail::import::{
    apple_mail_dir, collect_emlx, parse_emlx, store_rfc822,
};
use mailfind_lib::state;

fn main() {
    let account_id = env::args().nth(1).expect(
        "usage: cargo run --example backfill_apple_mail -- <account_id>",
    );

    let data_dir = state::default_data_dir().expect("data dir");
    let db_path = data_dir.join("mailfind.sqlite");
    println!("opening {}", db_path.display());
    let db = Database::open(&db_path).expect("open db");

    let dir = apple_mail_dir().expect("~/Library/Mail not found");
    println!("scanning {}", dir.display());
    let mut files = Vec::new();
    collect_emlx(&dir, &mut files);
    let total = files.len();
    println!("found {} .emlx files", total);

    let mut imported = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;

    for (i, path) in files.iter().enumerate() {
        let result = fs::read(path).map_err(|e| e.to_string()).and_then(|bytes| {
            parse_emlx(&bytes)
                .map_err(|e| e.to_string())
                .and_then(|raw| {
                    store_rfc822(&db, &account_id, &raw).map_err(|e| e.to_string())
                })
        });
        match result {
            Ok(true) => imported += 1,
            Ok(false) => skipped += 1,
            Err(e) => {
                errors += 1;
                if errors < 10 {
                    eprintln!("err on {}: {}", path.display(), e);
                }
            }
        }
        if (i + 1) % 250 == 0 || i + 1 == total {
            println!(
                "{}/{}  imported={} skipped={} errors={}",
                i + 1,
                total,
                imported,
                skipped,
                errors
            );
        }
    }

    println!(
        "DONE  imported={} skipped={} errors={} (of {})",
        imported, skipped, errors, total
    );
}
