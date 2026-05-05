//! Background-style embedding pass. Pulls chunks without an embedding from the
//! database and asks Ollama for vectors, one chunk at a time. Designed to be
//! safe to call repeatedly: it picks up where it left off and stops cleanly on
//! Ollama errors so a missing model doesn't block the rest of the app.

use crate::db::queries;
use crate::db::Database;
use crate::error::AppResult;
use crate::models::OllamaClient;

/// Embeds up to `limit` pending chunks. Returns the number actually embedded.
pub async fn ensure_embeddings(
    db: &Database,
    ollama: &OllamaClient,
    limit: i64,
) -> AppResult<usize> {
    let pending = {
        let conn = db.read()?;
        queries::pending_embedding_chunks(&conn, limit)?
    };
    if pending.is_empty() {
        return Ok(0);
    }

    let model = ollama.config.embedding_model.clone();
    let mut done = 0usize;
    for chunk in pending {
        match ollama.embed(&chunk.text).await {
            Ok(vec) => {
                let handle = db.write()?;
                queries::update_chunk_embedding(&handle, &chunk.chunk_id, &vec, &model)?;
                done += 1;
            }
            Err(e) => {
                tracing::warn!(?e, "embed failed; halting batch");
                break;
            }
        }
    }
    Ok(done)
}
