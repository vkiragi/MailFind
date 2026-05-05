//! Async Ollama client. Targets a locally-running Ollama daemon
//! (default `http://127.0.0.1:11434`).
//!
//! - Embeddings: POST `/api/embeddings` with `{ model, prompt }`
//! - Chat:       POST `/api/chat`       with `{ model, messages, stream:false }`
//! - List:       GET  `/api/tags`

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub endpoint: String,
    pub embedding_model: String,
    pub chat_model: String,
}

impl OllamaConfig {
    pub fn new(endpoint: String, embedding_model: String, chat_model: String) -> Self {
        Self {
            endpoint,
            embedding_model,
            chat_model,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OllamaClient {
    pub config: OllamaConfig,
    http: reqwest::Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelHealth {
    pub reachable: bool,
    pub embedding_available: bool,
    pub chat_available: bool,
    pub installed_models: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResp {
    embedding: Vec<f32>,
}

#[derive(Deserialize)]
struct ChatResp {
    message: ChatMessageOwned,
}

#[derive(Deserialize)]
struct ChatMessageOwned {
    #[allow(dead_code)]
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct TagsResp {
    models: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    name: String,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> AppResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        Ok(Self { config, http })
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.config.endpoint.trim_end_matches('/'), path)
    }

    /// Lightweight health check. Tries `/api/tags` and inspects whether the
    /// configured embedding/chat models are present (a missing model is not a
    /// fatal error — we still expose it so the UI can prompt the user to run
    /// `ollama pull <model>`).
    pub async fn health(&self) -> ModelHealth {
        match self.list_models().await {
            Ok(models) => {
                let embedding_available = model_present(&models, &self.config.embedding_model);
                let chat_available = model_present(&models, &self.config.chat_model);
                ModelHealth {
                    reachable: true,
                    embedding_available,
                    chat_available,
                    installed_models: models,
                }
            }
            Err(_) => ModelHealth {
                reachable: false,
                embedding_available: false,
                chat_available: false,
                installed_models: vec![],
            },
        }
    }

    pub async fn list_models(&self) -> AppResult<Vec<String>> {
        let resp = self
            .http
            .get(self.url("/api/tags"))
            .send()
            .await
            .map_err(|e| AppError::Ollama(format!("tags request failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(AppError::Ollama(format!(
                "tags returned status {}",
                resp.status()
            )));
        }
        let body: TagsResp = resp
            .json()
            .await
            .map_err(|e| AppError::Ollama(format!("tags parse failed: {e}")))?;
        Ok(body.models.into_iter().map(|m| m.name).collect())
    }

    /// Single embedding call. The Ollama API only supports one prompt per
    /// request, so batching is implemented as concurrent calls in the caller.
    pub async fn embed(&self, text: &str) -> AppResult<Vec<f32>> {
        let resp = self
            .http
            .post(self.url("/api/embeddings"))
            .json(&json!({
                "model": self.config.embedding_model,
                "prompt": text,
            }))
            .send()
            .await
            .map_err(|e| AppError::Ollama(format!("embed request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Ollama(format!(
                "embeddings returned {status}: {body}"
            )));
        }
        let parsed: EmbeddingResp = resp
            .json()
            .await
            .map_err(|e| AppError::Ollama(format!("embeddings parse failed: {e}")))?;
        Ok(parsed.embedding)
    }

    pub async fn chat(&self, messages: Vec<ChatMessage>) -> AppResult<String> {
        let resp = self
            .http
            .post(self.url("/api/chat"))
            .json(&json!({
                "model": self.config.chat_model,
                "messages": messages,
                "stream": false,
            }))
            .send()
            .await
            .map_err(|e| AppError::Ollama(format!("chat request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Ollama(format!("chat returned {status}: {body}")));
        }
        let parsed: ChatResp = resp
            .json()
            .await
            .map_err(|e| AppError::Ollama(format!("chat parse failed: {e}")))?;
        Ok(parsed.message.content)
    }
}

fn model_present(models: &[String], wanted: &str) -> bool {
    // Ollama returns full tags like `qwen2.5:3b-instruct`; we accept either
    // an exact match or a prefix match on the base name.
    let base = wanted.split(':').next().unwrap_or(wanted);
    models
        .iter()
        .any(|m| m == wanted || m == base || m.starts_with(&format!("{base}:")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_presence_matches_tags() {
        let models = vec![
            "qwen2.5:3b-instruct".to_string(),
            "nomic-embed-text:latest".to_string(),
        ];
        assert!(model_present(&models, "qwen2.5:3b-instruct"));
        assert!(model_present(&models, "nomic-embed-text"));
        assert!(!model_present(&models, "llama3:70b"));
    }
}
