use std::time::Duration;

use crate::search::indexer::ensure_embeddings;
use crate::state::AppState;

/// Spawns a persistent background task that embeds un-embedded chunks one
/// batch at a time. Runs for the lifetime of the Tauri runtime — no explicit
/// shutdown needed, the task is cancelled when the runtime drops on exit.
///
/// Batch size is kept small (50) so Ollama stays responsive for interactive
/// search/ask calls. `bump_counts_version` is called after each successful
/// batch so the "Embedded: N" counter in the Accounts tab updates live.
pub fn spawn_background_embedder(state: AppState) {
    tauri::async_runtime::spawn(async move {
        let mut total_done = 0usize;
        loop {
            match ensure_embeddings(&state.db, &state.ollama, 50).await {
                Ok(n) if n > 0 => {
                    total_done += n;
                    state.bump_counts_version();
                    if total_done % 1000 < 50 {
                        tracing::info!(total_done, "background embedder progress");
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Ok(_) => {
                    // Nothing pending — new mail may arrive after a sync, so keep checking.
                    tokio::time::sleep(Duration::from_secs(60)).await;
                }
                Err(e) => {
                    tracing::warn!(?e, "background embedder error; retrying in 30s");
                    tokio::time::sleep(Duration::from_secs(30)).await;
                }
            }
        }
    });
}
