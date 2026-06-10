//! Tauri command surface. Everything the React UI calls via `invoke()` lives
//! here. Keep this layer thin; business logic stays in `db`, `mail`, `models`,
//! `search`, `qa` so it remains testable without Tauri.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};
use uuid::Uuid;

use crate::credentials;
use crate::db::queries::{self, AccountRow, NewAccount};
use crate::error::{AppError, AppResult};
use crate::mail::sync::SyncWindow;
use crate::mail::{fixtures, sync};
use crate::qa::AnswerOutcome;
use crate::search::{self, SearchOutcome};
use crate::state::AppState;

/// Returns the cached embedded-chunk count if it was computed at the current
/// `counts_version`, otherwise calls `compute` and caches the result with the
/// current version. Writers invalidate by calling `state.bump_counts_version()`
/// — there's no time-based TTL, so a tab switch with no intervening writes is
/// instant.
fn cached_embedded_count(
    state: &AppState,
    key: &str,
    compute: impl FnOnce() -> AppResult<i64>,
) -> AppResult<i64> {
    let current = state
        .counts_version
        .load(std::sync::atomic::Ordering::Relaxed);
    {
        let cache = state.embedded_count_cache.lock();
        if let Some((stamped, value)) = cache.get(key) {
            if *stamped == current {
                return Ok(*value);
            }
        }
    }
    let value = compute()?;
    state
        .embedded_count_cache
        .lock()
        .insert(key.to_string(), (current, value));
    Ok(value)
}

#[derive(Debug, Deserialize)]
pub struct AddAccountRequest {
    pub email: String,
    pub password: String,
    pub display_name: Option<String>,
    pub imap_host: Option<String>,
    pub imap_port: Option<i64>,
}

#[derive(Debug, Serialize, Clone)]
pub struct AccountSummary {
    pub id: String,
    pub email: String,
    pub display_name: Option<String>,
    pub imap_host: String,
    pub created_at: String,
}

impl From<AccountRow> for AccountSummary {
    fn from(row: AccountRow) -> Self {
        Self {
            id: row.id,
            email: row.email,
            display_name: row.display_name,
            imap_host: row.imap_host,
            created_at: row.created_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct SyncStatusOut {
    pub account_id: Option<String>,
    pub is_running: bool,
    pub last_sync: Option<String>,
    pub total_messages: i64,
    pub embedded_messages: i64,
    pub last_error: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ModelStatusOut {
    pub ollama_reachable: bool,
    pub embedding_model: String,
    pub embedding_available: bool,
    pub chat_model: String,
    pub chat_available: bool,
    pub endpoint: String,
}

#[derive(Debug, Deserialize)]
pub struct IngestFixtureRequest {
    pub account_id: String,
    pub path: String,
}

#[derive(Debug, Serialize)]
pub struct IngestFixtureResponse {
    pub imported: usize,
    pub skipped: usize,
    pub errors: Vec<String>,
}

#[tauri::command]
pub fn greet(name: &str) -> String {
    format!("Hello, {name}, from MailFind!")
}

#[tauri::command]
pub fn list_accounts(state: State<AppState>) -> AppResult<Vec<AccountSummary>> {
    let conn = state.db.read()?;
    Ok(queries::list_accounts(&conn)?
        .into_iter()
        .map(Into::into)
        .collect())
}

#[tauri::command]
pub fn add_account(
    req: AddAccountRequest,
    state: State<AppState>,
) -> AppResult<AccountSummary> {
    let email = req.email.trim().to_string();
    let password = req.password.trim().to_string();
    if email.is_empty() || password.is_empty() {
        return Err(AppError::InvalidInput(
            "email and password are required".into(),
        ));
    }
    let host = req
        .imap_host
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| default_host_for(&email));
    let port = req.imap_port.filter(|p| *p > 0).unwrap_or(993);
    let display = req.display_name.filter(|s| !s.trim().is_empty());

    let keyring_ref = format!("imap:{}:{}", host, Uuid::new_v4());
    credentials::store_password(&keyring_ref, &password)?;

    let row = {
        let conn = state.db.write()?;
        match queries::insert_account(
            &conn,
            &NewAccount {
                email,
                display_name: display,
                imap_host: host,
                imap_port: port,
                keyring_ref: keyring_ref.clone(),
            },
        ) {
            Ok(r) => r,
            Err(e) => {
                // Keep the keychain clean if DB insert fails.
                let _ = credentials::delete_password(&keyring_ref);
                return Err(e);
            }
        }
    };
    Ok(row.into())
}

#[tauri::command]
pub fn remove_account(
    account_id: String,
    state: State<AppState>,
) -> AppResult<()> {
    let row = {
        let conn = state.db.read()?;
        queries::fetch_account(&conn, &account_id)?
    };
    if let Some(row) = row {
        let _ = credentials::delete_password(&row.keyring_ref);
        let conn = state.db.write()?;
        queries::delete_account(&conn, &account_id)?;
        state.bump_counts_version();
    }
    Ok(())
}

#[tauri::command]
pub fn sync_status(
    account_id: Option<String>,
    state: State<AppState>,
) -> AppResult<SyncStatusOut> {
    let conn = state.db.read()?;
    let (account_id, last_sync, is_running, last_error, total, embedded) = match account_id {
        Some(id) => {
            let s = queries::fetch_sync_status(&conn, &id)?;
            let (last_sync, is_running, last_error) = s
                .map(|(ls, ir, le)| (ls, ir, le))
                .unwrap_or((None, false, None));
            let total = queries::account_message_count(&conn, &id)?;
            let embedded = cached_embedded_count(&state, &id, || {
                queries::account_embedded_count(&conn, &id)
            })?;
            (Some(id), last_sync, is_running, last_error, total, embedded)
        }
        None => {
            let total = queries::message_count(&conn)?;
            let embedded = cached_embedded_count(&state, "", || {
                queries::embedded_message_count(&conn)
            })?;
            (None, None, false, None, total, embedded)
        }
    };
    Ok(SyncStatusOut {
        account_id,
        is_running,
        last_sync,
        total_messages: total,
        embedded_messages: embedded,
        last_error,
    })
}

#[tauri::command]
pub async fn sync_now(
    account_id: String,
    full_resync: bool,
    window: Option<SyncWindow>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> AppResult<SyncStatusOut> {
    // Guard 1: refuse if a sync is already running for this account, or if
    // the server has us in cooldown. Holding the mutex briefly while we check
    // and reserve the in-progress slot is fine — the actual sync runs after
    // the lock drops.
    let mut clear_persisted_cooldown = false;
    {
        let mut guard = state.sync_guard.lock();
        let now_ms = crate::state::now_unix_ms();
        if let Some(&until_ms) = guard.cooldown_until_ms.get(&account_id) {
            if now_ms < until_ms {
                let remaining_secs = ((until_ms - now_ms) / 1000).max(0);
                let mins = (remaining_secs + 59) / 60;
                return Err(AppError::RateLimited(format!(
                    "iCloud is throttling this account. Try again in ~{mins} minute{} (we paused syncs to let it recover).",
                    if mins == 1 { "" } else { "s" }
                )));
            }
            guard.cooldown_until_ms.remove(&account_id);
            clear_persisted_cooldown = true;
        }
        if guard.in_progress.contains_key(&account_id) {
            return Err(AppError::InvalidInput(
                "a sync is already running for this account".into(),
            ));
        }
        guard.in_progress.insert(account_id.clone(), ());
    }
    if clear_persisted_cooldown {
        let key = format!("{}{}", crate::state::COOLDOWN_KEY_PREFIX, account_id);
        if let Ok(conn) = state.db.write() {
            let _ = conn.execute("DELETE FROM app_settings WHERE key = ?1", [&key]);
        }
    }

    // Drop guard reservation when we leave this function, regardless of how.
    struct InProgressGuard {
        sync_guard: Arc<parking_lot::Mutex<crate::state::SyncGuard>>,
        account_id: String,
    }
    impl Drop for InProgressGuard {
        fn drop(&mut self) {
            self.sync_guard.lock().in_progress.remove(&self.account_id);
        }
    }
    let _in_progress = InProgressGuard {
        sync_guard: Arc::clone(&state.sync_guard),
        account_id: account_id.clone(),
    };

    let account = {
        let conn = state.db.read()?;
        queries::fetch_account(&conn, &account_id)?
            .ok_or_else(|| AppError::NotFound(format!("account {account_id} not found")))?
    };

    let db = Arc::clone(&state.db);
    let ollama = Arc::clone(&state.ollama);
    let acct_clone = account.clone();
    let acct_id = account.id.clone();
    let window = window.unwrap_or_default();
    let sync_result = sync::run_sync(app, &db, acct_clone, full_resync, window).await;

    // If the server rate-limited us, lock this account out for 10 minutes so
    // the user (or any retry) can't hammer it back into a deeper hole. The
    // cooldown is also persisted to SQLite so it survives app restarts —
    // otherwise a quick restart would let the user bypass the guard.
    if let Err(AppError::RateLimited(_)) = &sync_result {
        let until_ms = crate::state::now_unix_ms() + 10 * 60 * 1000;
        state
            .sync_guard
            .lock()
            .cooldown_until_ms
            .insert(account_id.clone(), until_ms);
        let key = format!("{}{}", crate::state::COOLDOWN_KEY_PREFIX, account_id);
        if let Ok(conn) = state.db.write() {
            let _ = queries::set_setting(&conn, &key, &until_ms.to_string());
        }
        tracing::warn!(account = %account_id, "applied 10-minute cooldown after rate limit");
    }
    let sync_outcome = sync_result?;
    tracing::info!(
        account = %acct_id,
        imported = sync_outcome.imported,
        skipped = sync_outcome.skipped,
        "sync complete"
    );
    if sync_outcome.imported > 0 {
        state.bump_counts_version();
    }

    // Best-effort embedding pass after sync.
    match search::ensure_embeddings(&db, &ollama, 256).await {
        Ok(n) if n > 0 => state.bump_counts_version(),
        Ok(_) => {}
        Err(e) => tracing::warn!(?e, "post-sync embedding pass failed"),
    }

    sync_status(Some(account_id), state)
}

#[tauri::command]
pub async fn model_status(state: State<'_, AppState>) -> AppResult<ModelStatusOut> {
    let health = state.ollama.health().await;
    Ok(ModelStatusOut {
        ollama_reachable: health.reachable,
        embedding_model: state.ollama.config.embedding_model.clone(),
        embedding_available: health.embedding_available,
        chat_model: state.ollama.config.chat_model.clone(),
        chat_available: health.chat_available,
        endpoint: state.ollama.config.endpoint.clone(),
    })
}

#[tauri::command]
pub async fn search_messages(
    query: String,
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> AppResult<SearchOutcome> {
    // Embed any backlog first so the dense pass has something to work with.
    if let Ok(n) = search::ensure_embeddings(&state.db, &state.ollama, 64).await {
        if n > 0 {
            state.bump_counts_version();
        }
    }
    search::search(&state.db, Some(&state.ollama), &query, limit.unwrap_or(20)).await
}

#[tauri::command]
pub async fn ask_question(
    question: String,
    limit: Option<usize>,
    state: State<'_, AppState>,
) -> AppResult<AnswerOutcome> {
    if let Ok(n) = search::ensure_embeddings(&state.db, &state.ollama, 64).await {
        if n > 0 {
            state.bump_counts_version();
        }
    }
    crate::qa::ask(&state.db, &state.ollama, &question, limit.unwrap_or(8)).await
}

#[tauri::command]
pub fn ingest_fixture(
    req: IngestFixtureRequest,
    state: State<AppState>,
) -> AppResult<IngestFixtureResponse> {
    let path = PathBuf::from(req.path);
    let report = fixtures::ingest_path(&state.db, &req.account_id, &path)?;
    if report.imported > 0 {
        state.bump_counts_version();
    }
    Ok(IngestFixtureResponse {
        imported: report.imported,
        skipped: report.skipped,
        errors: report.errors,
    })
}

#[derive(Debug, Serialize)]
pub struct AppleMailScanOut {
    /// Absolute path to `~/Library/Mail`, or `None` if Apple Mail isn't set up.
    pub mail_dir: Option<String>,
    /// Number of `.emlx` messages discovered on disk.
    pub message_count: usize,
}

#[derive(Debug, Serialize)]
pub struct ImportResultOut {
    pub imported: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// Counts the `.emlx` messages Apple Mail has stored locally so the UI can
/// show "Found N messages" before the user commits to an import. A count of 0
/// on a Mac that uses Apple Mail usually means the app lacks Full Disk Access.
#[tauri::command]
pub async fn scan_apple_mail() -> AppResult<AppleMailScanOut> {
    tokio::task::spawn_blocking(|| {
        let dir = crate::mail::import::apple_mail_dir();
        let message_count = match &dir {
            Some(d) => {
                let mut files = Vec::new();
                crate::mail::import::collect_emlx(d, &mut files);
                files.len()
            }
            None => 0,
        };
        AppleMailScanOut {
            mail_dir: dir.map(|d| d.display().to_string()),
            message_count,
        }
    })
    .await
    .map_err(|e| AppError::Internal(format!("apple mail scan join: {e}")))
}

/// Imports every locally-stored Apple Mail message into `account_id`. Emits
/// `import:progress` events while running; runs an embedding pass afterwards.
#[tauri::command]
pub async fn import_apple_mail(
    account_id: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> AppResult<ImportResultOut> {
    {
        let conn = state.db.read()?;
        queries::fetch_account(&conn, &account_id)?
            .ok_or_else(|| AppError::NotFound(format!("account {account_id} not found")))?;
    }

    let db = Arc::clone(&state.db);
    let acct = account_id.clone();
    let report = tokio::task::spawn_blocking(move || {
        crate::mail::import::run_apple_mail_import(&app, &db, &acct)
    })
    .await
    .map_err(|e| AppError::Internal(format!("apple mail import join: {e}")))??;
    if report.imported > 0 {
        state.bump_counts_version();
    }

    // Best-effort embedding pass so the freshly imported mail is searchable by
    // vector too. Keyword (FTS) search already works immediately.
    match search::ensure_embeddings(&state.db, &state.ollama, 256).await {
        Ok(n) if n > 0 => state.bump_counts_version(),
        Ok(_) => {}
        Err(e) => tracing::warn!(?e, "post-import embedding pass failed"),
    }

    tracing::info!(
        account = %account_id,
        imported = report.imported,
        skipped = report.skipped,
        errors = report.errors,
        "apple mail import complete"
    );
    Ok(ImportResultOut {
        imported: report.imported,
        skipped: report.skipped,
        errors: report.errors,
    })
}

#[tauri::command]
pub fn total_messages(state: State<AppState>) -> AppResult<i64> {
    let conn = state.db.read()?;
    Ok(queries::message_count(&conn)?)
}

/// Returns the unix-millis timestamp at which the account's sync cooldown
/// ends, or 0 if there's no active cooldown. The frontend uses this on mount
/// so it can show the correct countdown after an app restart.
#[tauri::command]
pub fn sync_cooldown_until(
    account_id: String,
    state: State<AppState>,
) -> AppResult<i64> {
    let guard = state.sync_guard.lock();
    let now = crate::state::now_unix_ms();
    let until = guard
        .cooldown_until_ms
        .get(&account_id)
        .copied()
        .unwrap_or(0);
    Ok(if until > now { until } else { 0 })
}

fn default_host_for(email: &str) -> String {
    let lower = email.to_ascii_lowercase();
    if lower.ends_with("@icloud.com")
        || lower.ends_with("@me.com")
        || lower.ends_with("@mac.com")
    {
        "imap.mail.me.com".to_string()
    } else if lower.ends_with("@gmail.com") {
        "imap.gmail.com".to_string()
    } else {
        "imap.mail.me.com".to_string()
    }
}
