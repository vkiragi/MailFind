//! Shared application state. Holds the SQLite connection pool and the Ollama
//! client handle so Tauri commands can read/write without re-initializing
//! resources on every call.

use std::path::PathBuf;
use std::sync::Arc;

use crate::db::queries;
use crate::db::Database;
use crate::error::AppResult;
use crate::models::{OllamaClient, OllamaConfig};

#[derive(Clone)]
pub struct AppState {
    pub db: Arc<Database>,
    pub ollama: Arc<OllamaClient>,
    pub data_dir: PathBuf,
}

impl AppState {
    pub fn initialize(data_dir: PathBuf) -> AppResult<Self> {
        let db_path = data_dir.join("mailfind.sqlite");
        let db = Database::open(&db_path)?;

        let cfg = read_ollama_config(&db)?;
        let ollama = OllamaClient::new(cfg)?;

        Ok(Self {
            db: Arc::new(db),
            ollama: Arc::new(ollama),
            data_dir,
        })
    }
}

fn read_ollama_config(db: &Database) -> AppResult<OllamaConfig> {
    let conn = db.read()?;
    let endpoint = queries::get_setting(&conn, "ollama_endpoint")?
        .unwrap_or_else(|| "http://127.0.0.1:11434".to_string());
    let embedding = queries::get_setting(&conn, "embedding_model")?
        .unwrap_or_else(|| "nomic-embed-text".to_string());
    let chat = queries::get_setting(&conn, "chat_model")?
        .unwrap_or_else(|| "granite4.1:3b".to_string());
    Ok(OllamaConfig::new(endpoint, embedding, chat))
}

/// Resolves the on-disk data directory, e.g. `~/Library/Application Support/com.mailfind.desktop`
/// on macOS. Used both at startup and in tests.
pub fn default_data_dir() -> AppResult<PathBuf> {
    let base = dirs::data_dir().unwrap_or_else(|| PathBuf::from("."));
    let path = base.join("com.mailfind.desktop");
    Ok(path)
}
