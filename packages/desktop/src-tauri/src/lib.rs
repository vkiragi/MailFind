//! MailFind library entry point. `main.rs` is a tiny shim that calls `run()`.

pub mod commands;
pub mod credentials;
pub mod db;
pub mod error;
pub mod mail;
pub mod models;
pub mod qa;
pub mod search;
pub mod state;

use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();

    let data_dir = state::default_data_dir().expect("data dir");
    let app_state = state::AppState::initialize(data_dir).expect("init app state");

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .setup({
            let state = app_state.clone();
            move |_app| {
                search::spawn_background_embedder(state);
                Ok(())
            }
        })
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            commands::greet,
            commands::list_accounts,
            commands::add_account,
            commands::remove_account,
            commands::sync_status,
            commands::sync_now,
            commands::model_status,
            commands::search_messages,
            commands::ask_question,
            commands::ingest_fixture,
            commands::scan_apple_mail,
            commands::import_apple_mail,
            commands::total_messages,
            commands::sync_cooldown_until,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
