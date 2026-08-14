//! Shared application state. Holds the SQLite connection pool and the Ollama
//! client handle so Tauri commands can read/write without re-initializing
//! resources on every call.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;

use crate::db::queries;
use crate::db::Database;
use crate::error::AppResult;
use crate::models::{OllamaClient, OllamaConfig};

/// SQLite key prefix for persisted cooldowns: `cooldown:<account_id>` →
/// unix-millis timestamp at which the cooldown expires.
pub const COOLDOWN_KEY_PREFIX: &str = "cooldown:";

/// Returns current wall-clock time as unix milliseconds.
pub fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Per-account guard against hammering remote IMAP servers. Tracks "an
/// account is currently syncing" (so the UI can't fire two in parallel) and
/// "this account is in a server-imposed cooldown until X" (so we don't open a
/// new connection that will just trip the rate limit again).
///
/// Cooldown timestamps are stored as wall-clock unix-millis (not Instant)
/// because they are persisted to SQLite and must survive process restarts.
#[derive(Default)]
pub struct SyncGuard {
    pub in_progress: HashMap<String, ()>,
    pub cooldown_until_ms: HashMap<String, i64>,
}

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub ollama: Arc<OllamaClient>,
    pub data_dir: PathBuf,
    pub sync_guard: Arc<Mutex<SyncGuard>>,
    /// Cache for the "Embedded: N" stat. The underlying query
    /// (COUNT(DISTINCT) over chunks JOIN messages) is ~2s cold on a large DB
    /// and the UI polls every 4s — so we cache forever and invalidate via
    /// `bump_counts_version` whenever something writes that changes the
    /// count. Key `""` is the global tally; other keys are `account_id`.
    pub embedded_count_cache: Arc<Mutex<HashMap<String, (u64, i64)>>>,
    /// Monotonic counter bumped by any write that affects message/embedded
    /// counts. Cached values stamped with a stale version are recomputed.
    pub counts_version: Arc<AtomicU64>,
    /// In-memory snapshot of every chunk embedding for the vector search pass.
    /// Loading all ~850MB of BLOBs from SQLite per query was the dominant search
    /// cost (~13s); this holds them resident so the cosine scan is milliseconds.
    /// Stamped with `counts_version` — any embedding write bumps that, forcing a
    /// lazy reload on the next search.
    pub embedding_cache: Arc<Mutex<EmbeddingCache>>,
    /// Cancellation flags for in-progress model pulls, keyed by model tag. The
    /// `pull_model` command registers one and the pull loop checks it; the
    /// `cancel_pull` command sets it to abort the download.
    pub pull_cancels: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

/// Cached embeddings plus the `counts_version` they were built at.
#[derive(Default)]
pub struct EmbeddingCache {
    version: Option<u64>,
    data: Arc<Vec<(String, Vec<f32>)>>,
}

impl AppState {
    /// Mark cached message/embedded counts as stale. Cheap (one atomic add) —
    /// safe to call after any write path that could have changed those
    /// counts.
    pub fn bump_counts_version(&self) {
        self.counts_version.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns the embedding snapshot for vector search, loading it from SQLite
    /// on a cold or stale (out-of-date `counts_version`) cache. The load runs
    /// outside the lock since it can take seconds on a large corpus; a rare race
    /// where two searches both load is harmless (last write wins).
    pub fn embeddings_snapshot(&self) -> AppResult<Arc<Vec<(String, Vec<f32>)>>> {
        let current = self.counts_version.load(Ordering::Relaxed);
        {
            let cache = self.embedding_cache.lock();
            if cache.version == Some(current) {
                return Ok(cache.data.clone());
            }
        }
        let data = Arc::new({
            let conn = self.db.read()?;
            queries::all_message_embeddings(&conn)?
        });
        let mut cache = self.embedding_cache.lock();
        cache.version = Some(current);
        cache.data = data.clone();
        Ok(data)
    }
}

impl AppState {
    pub fn initialize(data_dir: PathBuf) -> AppResult<Self> {
        let db_path = data_dir.join("mailfind.sqlite");
        let db = Database::open(&db_path)?;

        let cfg = read_ollama_config(&db)?;
        let ollama = OllamaClient::new(cfg)?;

        let mut sync_guard = SyncGuard::default();
        load_persisted_cooldowns(&db, &mut sync_guard)?;

        Ok(Self {
            db: Arc::new(db),
            ollama: Arc::new(ollama),
            data_dir,
            sync_guard: Arc::new(Mutex::new(sync_guard)),
            embedded_count_cache: Arc::new(Mutex::new(HashMap::new())),
            counts_version: Arc::new(AtomicU64::new(0)),
            embedding_cache: Arc::new(Mutex::new(EmbeddingCache::default())),
            pull_cancels: Arc::new(Mutex::new(HashMap::new())),
        })
    }
}

/// Read all `cooldown:*` rows from `app_settings`, drop any that have already
/// expired (also delete them from the DB), and seed the in-memory map with
/// what's left. Called once at startup.
fn load_persisted_cooldowns(db: &Database, guard: &mut SyncGuard) -> AppResult<()> {
    let conn = db.read()?;
    let mut stmt = conn.prepare(
        "SELECT key, value FROM app_settings WHERE key LIKE 'cooldown:%'",
    )?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)))?
        .filter_map(|r| r.ok())
        .collect();
    drop(stmt);
    drop(conn);

    let now = now_unix_ms();
    let mut to_delete: Vec<String> = Vec::new();
    for (key, value) in rows {
        let Some(account_id) = key.strip_prefix(COOLDOWN_KEY_PREFIX) else {
            continue;
        };
        let Ok(until_ms) = value.parse::<i64>() else {
            to_delete.push(key);
            continue;
        };
        if until_ms <= now {
            to_delete.push(key);
        } else {
            guard
                .cooldown_until_ms
                .insert(account_id.to_string(), until_ms);
        }
    }

    if !to_delete.is_empty() {
        let conn = db.write()?;
        for key in to_delete {
            let _ = conn.execute("DELETE FROM app_settings WHERE key = ?1", [key]);
        }
    }
    Ok(())
}

fn read_ollama_config(db: &Database) -> AppResult<OllamaConfig> {
    let conn = db.read()?;
    let endpoint = queries::get_setting(&conn, "ollama_endpoint")?
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
    let embedding = queries::get_setting(&conn, "embedding_model")?
        .unwrap_or_else(|| "nomic-embed-text".to_string());
    let chat = queries::get_setting(&conn, "chat_model")?
        .unwrap_or_else(|| "qwen3:8b".to_string());
    Ok(OllamaConfig::new(endpoint, embedding, chat))
}

/// Resolves the on-disk data directory, e.g. `~/Library/Application Support/com.mailfind.desktop`
/// on macOS. Used both at startup and in tests.
pub fn default_data_dir() -> AppResult<PathBuf> {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    let path = base.join("com.mailfind.desktop");
    Ok(path)
}
