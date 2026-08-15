//! One-time backfill: re-derives `messages.snippet` and `chunks.text` using
//! the current cleaning pipeline (decode_entities + strip_invisible +
//! strip_css), so mail ingested before that pipeline existed gets the same
//! clean text new mail gets.
//!
//! Two strategies depending on whether cleaning shifted chunk boundaries:
//! - Same chunk count as before (the overwhelming majority: decoding a
//!   handful of entities rarely shifts `chunking::split`'s paragraph/sentence
//!   boundaries): rewrite `chunks.text` **in place**, preserving the existing
//!   embedding. The `chunks_au` trigger keeps `chunks_fts` in sync.
//! - Different chunk count: delete and reinsert that message's chunks (new
//!   ones have `embedding = NULL`), same as `examples/rechunk.rs`. This is
//!   deliberately scoped to ONLY the messages that need it -- running the
//!   full `rechunk` example instead would delete+reinsert all ~280k chunks
//!   corpus-wide and force a full re-embed, exactly what the in-place
//!   strategy above exists to avoid. The background embedder picks up the
//!   small remainder's NULL embeddings automatically next launch.
//!
//! IMPORTANT: back up the DB and stop the desktop app first (write lock).
//!
//!   cargo run --release --example backfill_text

use mailfind_lib::db::queries::{self, NewChunk};
use mailfind_lib::db::Database;
use mailfind_lib::mail::parser::{build_chunk_input, clean_body, compact_text};
use mailfind_lib::search::chunking;
use mailfind_lib::state;

const BATCH_SIZE: usize = 500;

fn main() {
    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open db");

    // Deliberately no WHERE prefilter. An earlier version of this backfill
    // only selected messages containing a literal '&' or '{' (entity-text or
    // CSS shape), which caught HTML-entity padding fine but silently missed
    // messages where the padding is an already-decoded literal invisible
    // Unicode character (e.g. raw U+034F with no surrounding "&...;" text) --
    // there's no reliable SQL LIKE pattern that enumerates every such
    // codepoint. Every message's body/snippet is cheap to recompute and
    // compare in Rust, so just scan everything; only rows that actually
    // change get written.
    let candidates: Vec<(String, String, String, String, String, String)> = {
        let conn = db.read().expect("read");
        let mut stmt = conn
            .prepare(
                "SELECT id, COALESCE(subject,''), COALESCE(sender,''), COALESCE(recipients,''),
                        COALESCE(body_plain, COALESCE(body_html, '')), COALESCE(snippet, '')
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
                r.get::<_, String>(5)?,
            ))
        })
        .expect("query")
        .filter_map(Result::ok)
        .collect()
    };

    let total = candidates.len();
    println!("scanning all {total} messages...");

    let mut snippets_updated = 0usize;
    let mut messages_with_chunk_updates_in_place = 0usize;
    let mut messages_rechunked = 0usize;
    let mut new_chunks_needing_embedding = 0usize;
    let mut fully_unchanged = 0usize;
    let mut processed = 0usize;

    for batch in candidates.chunks(BATCH_SIZE) {
        let mut handle = db.write().expect("write");
        let tx = handle.transaction().expect("tx");

        for (id, subject, sender, recipients, body, old_snippet) in batch {
            let mut touched = false;

            // Snippet.
            let new_snippet = compact_text(&clean_body(body), 200);
            if &new_snippet != old_snippet {
                queries::update_message_snippet(&tx, id, &new_snippet).expect("update snippet");
                snippets_updated += 1;
                touched = true;
            }

            let combined = build_chunk_input(subject, sender, recipients, body, 8000);
            let new_chunks = chunking::split(&combined);
            let old_chunks = queries::fetch_chunk_texts(&tx, id).expect("fetch chunks");

            if new_chunks.len() == old_chunks.len() {
                // Same chunk count -- rewrite in place, embeddings preserved.
                let mut any_chunk_changed = false;
                for ((chunk_id, old_text), new_text) in old_chunks.iter().zip(new_chunks.iter()) {
                    if old_text != new_text {
                        queries::update_chunk_text(&tx, chunk_id, new_text)
                            .expect("update chunk text");
                        any_chunk_changed = true;
                    }
                }
                if any_chunk_changed {
                    messages_with_chunk_updates_in_place += 1;
                    touched = true;
                }
            } else if !new_chunks.is_empty() {
                // Chunk boundaries moved -- delete and reinsert just this
                // message's chunks. New chunks start with no embedding; the
                // background embedder fills them in on next launch.
                tx.execute("DELETE FROM chunks WHERE message_id = ?1", [id])
                    .expect("delete old chunks");
                for (idx, text) in new_chunks.into_iter().enumerate() {
                    queries::insert_chunk(
                        &tx,
                        &NewChunk {
                            message_id: id.clone(),
                            chunk_index: idx as i64,
                            text,
                            embedding: None,
                            embedding_model: None,
                        },
                    )
                    .expect("insert chunk");
                    new_chunks_needing_embedding += 1;
                }
                messages_rechunked += 1;
                touched = true;
            }
            // new_chunks.is_empty() with a nonzero old count would mean the
            // cleaned body somehow became empty text -- doesn't happen given
            // clean_body only decodes/strips, never reduces valid text to
            // nothing, but if it ever did, leaving old chunks untouched here
            // is the safe default.

            if !touched {
                fully_unchanged += 1;
            }

            processed += 1;
            if processed % 1000 == 0 {
                println!("  {processed}/{total}");
            }
        }

        tx.commit().expect("commit");
    }

    println!(
        "\ndone: {snippets_updated} snippets updated, \
         {messages_with_chunk_updates_in_place} messages' chunk text updated in place \
         (embeddings preserved), {messages_rechunked} messages rechunked \
         ({new_chunks_needing_embedding} new chunks now pending embedding), \
         {fully_unchanged} already clean"
    );
}
