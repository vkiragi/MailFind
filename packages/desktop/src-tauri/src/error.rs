use serde::{Serialize, Serializer};
use thiserror::Error;

/// Top level result type used by Tauri commands.
pub type AppResult<T> = std::result::Result<T, AppError>;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("connection pool error: {0}")]
    Pool(#[from] r2d2::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("http error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("imap error: {0}")]
    Imap(String),

    #[error("rate limited by mail server: {0}")]
    RateLimited(String),

    #[error("mail parsing error: {0}")]
    MailParse(String),

    #[error("credential storage error: {0}")]
    Keyring(String),

    #[error("ollama error: {0}")]
    Ollama(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl AppError {
    pub fn imap<E: std::fmt::Display>(err: E) -> Self {
        AppError::Imap(err.to_string())
    }
}

impl From<keyring::Error> for AppError {
    fn from(value: keyring::Error) -> Self {
        AppError::Keyring(value.to_string())
    }
}

impl From<imap::Error> for AppError {
    fn from(value: imap::Error) -> Self {
        AppError::Imap(value.to_string())
    }
}

impl From<anyhow::Error> for AppError {
    fn from(value: anyhow::Error) -> Self {
        AppError::Internal(value.to_string())
    }
}

/// Tauri requires errors to be serializable; the frontend gets a user-readable string.
impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
