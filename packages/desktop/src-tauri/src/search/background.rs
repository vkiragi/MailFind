use std::time::Duration;

use crate::search::indexer::ensure_embeddings;
use crate::state::AppState;

/// Spawns a persistent background task that embeds un-embedded chunks one
/// batch at a time. Runs for the lifetime of the Tauri runtime — no explicit
/// shutdown needed, the task is cancelled when the runtime drops on exit.
///
/// Batch size is kept small (50) so Ollama stays responsive for interactive
/// search/ask calls. `bump_counts_version` invalidates the counts cache that
/// `sync_status` relies on, so it is throttled to at most once every 5s during
/// active embedding — bumping after every 50-chunk batch (~2x/sec) thrashes the
/// cache and forces `sync_status` to recompute its COUNT on every Accounts-tab
/// poll. A deferred bump is flushed when the backlog drains so the
/// "Embedded: N" counter settles on the final value.
pub fn spawn_background_embedder(state: AppState) {
    tauri::async_runtime::spawn(async move {
        let mut total_done = 0usize;
        let mut last_bump = std::time::Instant::now();
        let mut pending_bump = false;
        loop {
            match ensure_embeddings(&state.db, &state.ollama, 50).await {
                Ok(n) if n > 0 => {
                    total_done += n;
                    pending_bump = true;
                    if last_bump.elapsed() >= Duration::from_secs(5) {
                        state.bump_counts_version();
                        last_bump = std::time::Instant::now();
                        pending_bump = false;
                    }
                    if total_done % 1000 < 50 {
                        tracing::info!(total_done, "background embedder progress");
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                Ok(_) => {
                    // Backlog drained — flush any deferred bump so the counter
                    // reflects the final state. New mail may arrive after a
                    // sync, so keep checking.
                    if pending_bump {
                        state.bump_counts_version();
                        pending_bump = false;
                    }
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
