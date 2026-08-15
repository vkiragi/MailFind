//! Re-chunk every message in the live DB using the current (CSS-stripping)
//! pipeline. Drops old chunks per-message and inserts fresh ones; FTS index
//! is rebuilt by the existing AFTER INSERT/DELETE triggers on `chunks`.
//!
//!   cargo run --release --example rechunk
//!
//! IMPORTANT: stop the desktop app first so we don't fight for the write lock.

use mailfind_lib::db::queries::{self, NewChunk};
use mailfind_lib::db::Database;
use mailfind_lib::mail::parser::build_chunk_input;
use mailfind_lib::search::chunking;
use mailfind_lib::state;

fn main() {
    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open db");

    // Read all message metadata up front so the write phase doesn't share a
    // connection with an open read cursor.
    let messages: Vec<(String, String, String, String, String)> = {
        let conn = db.read().expect("read");
        let mut stmt = conn
            .prepare(
                "SELECT id, \
                        COALESCE(subject, ''), \
                        COALESCE(sender, ''), \
                        COALESCE(recipients, ''), \
                        COALESCE(body_plain, COALESCE(body_html, '')) \
                 FROM messages",
            )
            .expect("prepare");
        stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
            ))
        })
        .expect("query")
        .filter_map(Result::ok)
        .collect()
    };

    let total = messages.len();
    println!("re-chunking {total} messages...");

    let mut done = 0usize;
    let mut skipped_empty = 0usize;
    for (i, (msg_id, subject, sender, recipients, body)) in messages.into_iter().enumerate() {
        let combined = build_chunk_input(&subject, &sender, &recipients, &body, 8000);
        let chunks = chunking::split(&combined);

        if chunks.is_empty() {
            skipped_empty += 1;
            continue;
        }

        let mut handle = db.write().expect("write");
        let tx = handle.transaction().expect("tx");
        tx.execute("DELETE FROM chunks WHERE message_id = ?1", [&msg_id])
            .expect("delete old chunks");
        for (idx, text) in chunks.into_iter().enumerate() {
            queries::insert_chunk(
                &tx,
                &NewChunk {
                    message_id: msg_id.clone(),
                    chunk_index: idx as i64,
                    text,
                    embedding: None,
                    embedding_model: None,
                },
            )
            .expect("insert chunk");
        }
        tx.commit().expect("commit");
        done += 1;

        if (i + 1) % 1000 == 0 {
            println!("  {}/{}", i + 1, total);
        }
    }

    println!("done: {done} re-chunked, {skipped_empty} skipped (empty body)");
}
