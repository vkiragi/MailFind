//! Tauri command surface. Everything the React UI calls via `invoke()` lives
//! here. Keep this layer thin; business logic stays in `db`, `mail`, `models`,
//! `search`, `qa` so it remains testable without Tauri.

use std::path::PathBuf;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use uuid::Uuid;

use crate::credentials;
use crate::db::queries::{self, AccountRow, NewAccount};
use crate::error::{AppError, AppResult};
use crate::mail::sync::SyncWindow;
use crate::mail::{fixtures, sync};
use crate::models::select::{self, AutoPick};
use crate::qa::AnswerOutcome;
use crate::search::{self, SearchOutcome};
use crate::state::AppState;

/// Event carrying each streamed answer fragment from `ask_question` to the UI.
pub const ASK_TOKEN_EVENT: &str = "ask:token";

/// `app_settings` key holding the active chat model.
pub const CHAT_MODEL_SETTING: &str = "chat_model";
/// `app_settings` key: `"auto"` (derive from RAM/installed models) or `"user"`
/// (explicit picker choice — never auto-overridden).
pub const CHAT_MODEL_SOURCE_SETTING: &str = "chat_model_source";

/// Startup: unless the user explicitly chose a chat model, derive the best one
/// this machine can run from total RAM + installed Ollama models and apply it.
/// This is what neutralizes migration 004's blanket qwen3:8b force-set on
/// machines that can't afford it. Runs off the UI path (spawned in `lib::run`)
/// because it makes an async Ollama call that would hang startup if the daemon
/// is down. A sub-second window where the seed model is active before this
/// resolves is harmless — Ask won't fire that fast.
pub async fn auto_pick_chat_model(state: &AppState) {
    // An explicit user choice is authoritative — never override it.
    let source = match state.db.read() {
        Ok(conn) => queries::get_setting(&conn, CHAT_MODEL_SOURCE_SETTING)
            .ok()
            .flatten(),
        Err(_) => None,
    };
    if source.as_deref() == Some("user") {
        return;
    }

    let installed = match state.ollama.list_models().await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(?e, "auto-pick: Ollama unreachable; leaving chat model as-is");
            return;
        }
    };
    let budget = select::model_budget_gb(select::total_ram_gb());
    match select::auto_pick(budget, &installed) {
        AutoPick::Model(model) => {
            if model != state.ollama.chat_model() {
                state.ollama.set_chat_model(model.clone()).await;
                if let Ok(conn) = state.db.write() {
                    let _ = queries::set_setting(&conn, CHAT_MODEL_SETTING, &model);
                }
                tracing::info!(%model, budget_gb = budget, "auto-picked chat model");
            }
        }
        AutoPick::NeedsPull(model) => {
            tracing::info!(
                %model,
                budget_gb = budget,
                "auto-pick: best-fitting model not installed; Ask needs `ollama pull`"
            );
        }
        AutoPick::SearchOnly => {
            tracing::info!(budget_gb = budget, "auto-pick: RAM below Ask floor; search-only");
        }
    }
}

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

/// Precomputes the embedded-message count for every account at startup so the
/// first Accounts-tab load doesn't pay the cold ~1.6s COUNT (cold OS page cache
/// + a version bump from the launch auto-sync would otherwise land on the
/// user's click). Reuses `cached_embedded_count` so the warmed value is keyed
/// and version-stamped exactly like the UI path. Runs on its own thread — a
/// wasted recompute (if a write bumps the version mid-warm) is harmless.
pub fn warm_embedded_counts(state: &AppState) {
    let ids = match state.db.read() {
        Ok(conn) => queries::list_accounts(&conn)
            .map(|rows| rows.into_iter().map(|r| r.id).collect::<Vec<_>>())
            .unwrap_or_default(),
        Err(_) => return,
    };
    for id in ids {
        let _ = cached_embedded_count(state, &id, || {
            let conn = state.db.read()?;
            queries::account_embedded_count(&conn, &id)
        });
    }
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

/// One selectable chat model, with the labels the picker shows.
#[derive(Debug, Serialize)]
pub struct ModelOption {
    pub model: String,
    /// Approx RAM (GiB) to load it.
    pub needs_gb: f64,
    /// Present in Ollama right now.
    pub installed: bool,
    /// Fits this machine's RAM budget.
    pub fits: bool,
    /// Eligible for automatic selection (false = opt-in only, e.g. the tiny floor).
    pub auto: bool,
    /// Non-fatal caveat to show next to the model.
    pub warn: Option<String>,
    /// This is the currently active chat model.
    pub is_current: bool,
}

/// Everything the settings model-picker needs in one call.
#[derive(Debug, Serialize)]
pub struct ModelListOut {
    pub ollama_reachable: bool,
    pub total_ram_gb: f64,
    pub budget_gb: f64,
    /// Active chat model.
    pub current_model: String,
    /// `"auto"` or `"user"`.
    pub source: String,
    /// What auto-pick would choose now: `"model"` | `"needs_pull"` | `"search_only"`.
    pub auto_pick_state: String,
    /// The model for `"model"`/`"needs_pull"` states (None for `"search_only"`).
    pub auto_pick_model: Option<String>,
    /// The curated, RAM-labeled recommendation ladder.
    pub options: Vec<ModelOption>,
    /// Installed chat models not in the ladder (unlabeled advanced choices).
    pub other_installed: Vec<String>,
}

/// Lists the recommended model ladder plus any other installed chat models,
/// annotated with RAM fit / installed / current, so the settings picker can
/// render everything from one call. Also reports what auto-pick would choose —
/// which phase-5 Ask gating uses to show a "needs a model" state.
#[tauri::command]
pub async fn list_models(state: State<'_, AppState>) -> AppResult<ModelListOut> {
    let (ollama_reachable, installed) = match state.ollama.list_models().await {
        Ok(m) => (true, m),
        Err(_) => (false, vec![]),
    };

    let total_ram_gb = select::total_ram_gb();
    let budget_gb = select::model_budget_gb(total_ram_gb);
    let current_model = state.ollama.chat_model();

    let source = {
        let conn = state.db.read()?;
        queries::get_setting(&conn, CHAT_MODEL_SOURCE_SETTING)?
            .unwrap_or_else(|| "auto".to_string())
    };

    let (auto_pick_state, auto_pick_model) = match select::auto_pick(budget_gb, &installed) {
        AutoPick::Model(m) => ("model", Some(m)),
        AutoPick::NeedsPull(m) => ("needs_pull", Some(m)),
        AutoPick::SearchOnly => ("search_only", None),
    };

    let options: Vec<ModelOption> = select::RECOMMENDED
        .iter()
        .map(|r| ModelOption {
            model: r.model.to_string(),
            needs_gb: r.needs_gb,
            installed: select::is_model_installed(&installed, r.model),
            fits: r.needs_gb <= budget_gb,
            auto: r.auto,
            warn: r.warn.map(|w| w.to_string()),
            is_current: r.model == current_model,
        })
        .collect();

    // Installed models the ladder doesn't cover, minus the embedding model —
    // power users may still pick these even though we can't RAM-label them.
    let embed_base = state
        .ollama
        .config
        .embedding_model
        .split(':')
        .next()
        .unwrap_or("")
        .to_string();
    let other_installed: Vec<String> = installed
        .iter()
        .filter(|tag| {
            let base = tag.split(':').next().unwrap_or("");
            base != embed_base
                && !select::RECOMMENDED
                    .iter()
                    .any(|r| select::is_model_installed(std::slice::from_ref(*tag), r.model))
        })
        .cloned()
        .collect();

    Ok(ModelListOut {
        ollama_reachable,
        total_ram_gb,
        budget_gb,
        current_model,
        source,
        auto_pick_state: auto_pick_state.to_string(),
        auto_pick_model,
        options,
        other_installed,
    })
}

/// Sets the active chat model: swaps the live client (unloading the previous
/// model), persists the choice, and marks the source `"user"` so startup
/// auto-pick never overrides it again. Takes effect on the next Ask — no restart.
#[tauri::command]
pub async fn set_chat_model(model: String, state: State<'_, AppState>) -> AppResult<()> {
    let model = model.trim().to_string();
    if model.is_empty() {
        return Err(AppError::InvalidInput("model is required".into()));
    }
    state.ollama.set_chat_model(model.clone()).await;
    let conn = state.db.write()?;
    queries::set_setting(&conn, CHAT_MODEL_SETTING, &model)?;
    queries::set_setting(&conn, CHAT_MODEL_SOURCE_SETTING, "user")?;
    Ok(())
}

/// Event carrying model-pull progress from `pull_model` to the UI.
pub const MODEL_PULL_EVENT: &str = "model:pull";

/// One `model:pull` event: streamed progress, then a terminal event with
/// `done: true` (success) or `error: Some(_)` (failed or cancelled).
#[derive(Debug, Clone, Serialize)]
pub struct ModelPullOut {
    pub model: String,
    pub status: String,
    pub completed: u64,
    pub total: u64,
    pub done: bool,
    pub error: Option<String>,
}

/// Downloads a model into Ollama, streaming progress via `model:pull` events so
/// the UI can show a progress bar. Registers a cancel flag `cancel_pull` can
/// flip. This is the in-app alternative to running `ollama pull` in a terminal.
#[tauri::command]
pub async fn pull_model(
    model: String,
    app: AppHandle,
    state: State<'_, AppState>,
) -> AppResult<()> {
    use std::sync::atomic::AtomicBool;
    let model = model.trim().to_string();
    if model.is_empty() {
        return Err(AppError::InvalidInput("model is required".into()));
    }

    let flag = Arc::new(AtomicBool::new(false));
    state.pull_cancels.lock().insert(model.clone(), flag.clone());

    let progress_app = app.clone();
    let progress_model = model.clone();
    let outcome = state
        .ollama
        .pull_model(
            &model,
            |p| {
                let _ = progress_app.emit(
                    MODEL_PULL_EVENT,
                    ModelPullOut {
                        model: progress_model.clone(),
                        status: p.status,
                        completed: p.completed,
                        total: p.total,
                        done: false,
                        error: None,
                    },
                );
            },
            &flag,
        )
        .await;

    state.pull_cancels.lock().remove(&model);

    match outcome {
        Ok(true) => {
            let _ = app.emit(
                MODEL_PULL_EVENT,
                ModelPullOut {
                    model,
                    status: "success".into(),
                    completed: 0,
                    total: 0,
                    done: true,
                    error: None,
                },
            );
            Ok(())
        }
        Ok(false) => {
            let _ = app.emit(
                MODEL_PULL_EVENT,
                ModelPullOut {
                    model,
                    status: "cancelled".into(),
                    completed: 0,
                    total: 0,
                    done: false,
                    error: Some("Cancelled".into()),
                },
            );
            Ok(())
        }
        Err(e) => {
            let _ = app.emit(
                MODEL_PULL_EVENT,
                ModelPullOut {
                    model,
                    status: "error".into(),
                    completed: 0,
                    total: 0,
                    done: false,
                    error: Some(e.to_string()),
                },
            );
            Err(e)
        }
    }
}

/// Signals an in-progress `pull_model` to stop.
#[tauri::command]
pub fn cancel_pull(model: String, state: State<AppState>) -> AppResult<()> {
    use std::sync::atomic::Ordering;
    if let Some(flag) = state.pull_cancels.lock().get(&model) {
        flag.store(true, Ordering::Relaxed);
    }
    Ok(())
}

#[tauri::command]
pub async fn model_status(state: State<'_, AppState>) -> AppResult<ModelStatusOut> {
    let health = state.ollama.health().await;
    Ok(ModelStatusOut {
        ollama_reachable: health.reachable,
        embedding_model: state.ollama.config.embedding_model.clone(),
        embedding_available: health.embedding_available,
        chat_model: state.ollama.chat_model(),
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
    let embeddings = state.embeddings_snapshot()?;
    search::search(
        &state.db,
        Some(&state.ollama),
        &query,
        limit.unwrap_or(20),
        Some(embeddings),
    )
    .await
}

#[tauri::command]
pub async fn ask_question(
    question: String,
    limit: Option<usize>,
    app: AppHandle,
    state: State<'_, AppState>,
) -> AppResult<AnswerOutcome> {
    // Stream answer fragments to the frontend as they arrive; the full
    // AnswerOutcome (with citations) is still returned when the command resolves.
    let embeddings = state.embeddings_snapshot()?;
    crate::qa::ask(
        &state.db,
        &state.ollama,
        &question,
        limit.unwrap_or(8),
        Some(embeddings),
        |delta| {
            let _ = app.emit(ASK_TOKEN_EVENT, delta);
        },
    )
    .await
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
