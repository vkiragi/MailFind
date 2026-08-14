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
        .menu(|handle| {
            use tauri::menu::{Menu, MenuItem, Submenu};
            // Start from the standard macOS menu (App/Edit/Window/Help so
            // Quit, copy/paste, etc. keep working) and add a View menu with the
            // usual zoom commands. The items just emit to the frontend, where
            // the actual CSS zoom lives.
            let menu = Menu::default(handle)?;
            let zoom_in =
                MenuItem::with_id(handle, "zoom_in", "Zoom In", true, Some("CmdOrCtrl+Plus"))?;
            let zoom_out =
                MenuItem::with_id(handle, "zoom_out", "Zoom Out", true, Some("CmdOrCtrl+-"))?;
            let zoom_reset =
                MenuItem::with_id(handle, "zoom_reset", "Actual Size", true, Some("CmdOrCtrl+0"))?;
            let view =
                Submenu::with_items(handle, "View", true, &[&zoom_in, &zoom_out, &zoom_reset])?;
            menu.append(&view)?;
            Ok(menu)
        })
        .on_menu_event(|app, event| {
            use tauri::Emitter;
            let id = event.id().as_ref();
            if matches!(id, "zoom_in" | "zoom_out" | "zoom_reset") {
                let _ = app.emit("menu://zoom", id);
            }
        })
        .setup({
            let state = app_state.clone();
            move |_app| {
                // Warm the embedded-count cache and the ~850MB embedding
                // snapshot off the UI path, so the first Accounts-tab load and
                // the first search don't pay their cold loads interactively.
                std::thread::spawn({
                    let state = state.clone();
                    move || {
                        commands::warm_embedded_counts(&state);
                        let _ = state.embeddings_snapshot();
                    }
                });
                // Derive the best chat model for this machine's RAM (async: hits
                // Ollama). Off the UI/setup path so a down daemon can't hang it.
                tauri::async_runtime::spawn({
                    let state = state.clone();
                    async move {
                        commands::auto_pick_chat_model(&state).await;
                    }
                });
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
            commands::list_models,
            commands::set_chat_model,
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
