//! Embeds every chunk with `embedding IS NULL`, looping until none remain.
//! Standalone equivalent of the background embedder's catch-up pass, useful
//! right after a backfill/rechunk run that reset some chunks' embeddings
//! (e.g. `examples/backfill_text.rs`, `examples/rechunk.rs`) without needing
//! to launch the full desktop app.
//!
//!   cargo run --release --example drain_pending_embeddings

use mailfind_lib::db::Database;
use mailfind_lib::models::{OllamaClient, OllamaConfig};
use mailfind_lib::search::ensure_embeddings;
use mailfind_lib::state;

#[tokio::main]
async fn main() {
    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open db");
    let ollama = OllamaClient::new(OllamaConfig::new(
        "http://127.0.0.1:11434".to_string(),
        "nomic-embed-text".to_string(),
        "qwen3:8b".to_string(),
    ))
    .expect("ollama client");

    let total_pending: i64 = {
        let conn = db.read().expect("read");
        conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL",
            [],
            |r| r.get(0),
        )
        .expect("count pending")
    };
    println!("{total_pending} chunks pending embedding");

    let mut done = 0usize;
    loop {
        match ensure_embeddings(&db, &ollama, 50).await {
            Ok(0) => break,
            Ok(n) => {
                done += n;
                if done % 500 < 50 {
                    println!("  {done}/{total_pending}");
                }
            }
            Err(e) => {
                eprintln!("embedding batch failed: {e}");
                break;
            }
        }
    }

    let remaining: i64 = {
        let conn = db.read().expect("read");
        conn.query_row(
            "SELECT COUNT(*) FROM chunks WHERE embedding IS NULL",
            [],
            |r| r.get(0),
        )
        .expect("count pending")
    };
    println!("done: {done} embedded, {remaining} still pending (Ollama unreachable/error?)");
}
