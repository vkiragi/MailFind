//! Async Ollama client. Targets a locally-running Ollama daemon
//! (default `http://127.0.0.1:11434`).
//!
//! - Embeddings: POST `/api/embeddings` with `{ model, prompt }`
//! - Chat:       POST `/api/chat`       with `{ model, messages, stream:false }`
//! - List:       GET  `/api/tags`

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::{AppError, AppResult};

#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub endpoint: String,
    pub embedding_model: String,
    /// Seed for the client's live chat model — read only once, in
    /// `OllamaClient::new`. After construction the active chat model lives in
    /// `OllamaClient::chat_model` (interior-mutable) so it can be swapped at
    /// runtime; never read `config.chat_model` past construction.
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
    /// The active chat model. Interior-mutable so `set_chat_model` (from the
    /// settings picker or startup auto-pick) takes effect on the next `chat()`
    /// call without rebuilding the client or restarting the app. Shared across
    /// every `Arc<OllamaClient>` clone.
    chat_model: Arc<Mutex<String>>,
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
struct ChatStreamChunk {
    message: Option<ChatMessageOwned>,
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

/// One progress update while pulling a model.
pub struct PullProgress {
    /// Ollama's status text, e.g. "pulling manifest", "downloading …".
    pub status: String,
    /// Bytes downloaded so far for the current layer (0 if not applicable).
    pub completed: u64,
    /// Total bytes for the current layer (0 if unknown).
    pub total: u64,
}

#[derive(Deserialize)]
struct PullStreamLine {
    status: Option<String>,
    total: Option<u64>,
    completed: Option<u64>,
    error: Option<String>,
}

impl OllamaClient {
    pub fn new(config: OllamaConfig) -> AppResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?;
        let chat_model = Arc::new(Mutex::new(config.chat_model.clone()));
        Ok(Self {
            config,
            chat_model,
            http,
        })
    }

    /// The active chat model.
    pub fn chat_model(&self) -> String {
        self.chat_model.lock().clone()
    }

    /// Swap the active chat model at runtime. No-op if unchanged. Best-effort
    /// unloads the previously-active model from Ollama (`keep_alive: 0`) so it
    /// doesn't sit warm for the usual 30m eating RAM — the whole point of the
    /// picker is to fit constrained machines. Persisting the choice to SQLite is
    /// the caller's responsibility.
    pub async fn set_chat_model(&self, model: String) {
        let previous = {
            let mut guard = self.chat_model.lock();
            if *guard == model {
                return;
            }
            std::mem::replace(&mut *guard, model)
        };
        self.unload_model(&previous).await;
    }

    /// Asks Ollama to evict a model from memory immediately (`keep_alive: 0`).
    /// Best-effort: failures (model wasn't loaded, daemon down) are ignored.
    async fn unload_model(&self, model: &str) {
        let _ = self
            .http
            .post(self.url("/api/generate"))
            .json(&json!({ "model": model, "keep_alive": 0 }))
            .send()
            .await;
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
                let chat_available = model_present(&models, &self.chat_model());
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

    /// Streams a chat completion. `on_delta` is called with each new content
    /// fragment as it arrives; the full accumulated answer is returned at the
    /// end. `keep_alive` pins the model in RAM for 30m so spaced-out questions
    /// avoid a cold reload — Ollama frees the memory after that idle window.
    pub async fn chat<F: FnMut(&str)>(
        &self,
        messages: Vec<ChatMessage>,
        mut on_delta: F,
    ) -> AppResult<String> {
        let mut resp = self
            .http
            .post(self.url("/api/chat"))
            .json(&json!({
                "model": self.chat_model(),
                "messages": messages,
                "stream": true,
                "keep_alive": "30m",
                // Low temperature: this is factual RAG over the user's mail, not
                // creative writing. Ollama defaults to 0.8, which caused the
                // model to occasionally mis-attribute citations.
                "options": { "temperature": 0.2 },
            }))
            .send()
            .await
            .map_err(|e| AppError::Ollama(format!("chat request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Ollama(format!("chat returned {status}: {body}")));
        }

        // Ollama streams one JSON object per line. Accumulate bytes and parse
        // each complete line, forwarding new content deltas as they arrive.
        let mut full = String::new();
        let mut buf: Vec<u8> = Vec::new();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| AppError::Ollama(format!("chat stream failed: {e}")))?
        {
            buf.extend_from_slice(&chunk);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = &line[..line.len() - 1];
                if line.is_empty() {
                    continue;
                }
                let Ok(msg) = serde_json::from_slice::<ChatStreamChunk>(line) else {
                    continue;
                };
                if let Some(m) = msg.message {
                    if !m.content.is_empty() {
                        full.push_str(&m.content);
                        on_delta(&m.content);
                    }
                }
            }
        }
        Ok(full)
    }

    /// Pulls a model via `/api/pull`, streaming progress to `on_progress`.
    /// Returns `Ok(true)` on success, `Ok(false)` if `cancel` was set mid-pull.
    /// Uses a very long per-request timeout since model downloads (GBs) far
    /// exceed the client's default request timeout.
    pub async fn pull_model<F: FnMut(PullProgress)>(
        &self,
        model: &str,
        mut on_progress: F,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> AppResult<bool> {
        use std::sync::atomic::Ordering;

        let mut resp = self
            .http
            .post(self.url("/api/pull"))
            .timeout(Duration::from_secs(24 * 60 * 60))
            .json(&json!({ "model": model, "stream": true }))
            .send()
            .await
            .map_err(|e| AppError::Ollama(format!("pull request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::Ollama(format!("pull returned {status}: {body}")));
        }

        let mut buf: Vec<u8> = Vec::new();
        let mut total = 0u64;
        let mut completed = 0u64;
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| AppError::Ollama(format!("pull stream failed: {e}")))?
        {
            if cancel.load(Ordering::Relaxed) {
                return Ok(false);
            }
            buf.extend_from_slice(&chunk);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=nl).collect();
                let line = &line[..line.len() - 1];
                if line.is_empty() {
                    continue;
                }
                let Ok(msg) = serde_json::from_slice::<PullStreamLine>(line) else {
                    continue;
                };
                if let Some(err) = msg.error {
                    return Err(AppError::Ollama(format!("pull failed: {err}")));
                }
                if let Some(t) = msg.total {
                    total = t;
                }
                if let Some(c) = msg.completed {
                    completed = c;
                }
                on_progress(PullProgress {
                    status: msg.status.unwrap_or_default(),
                    completed,
                    total,
                });
            }
        }
        Ok(true)
    }
}

pub(crate) fn model_present(models: &[String], wanted: &str) -> bool {
    // Ollama returns full tags like `granite4.1:3b`; we accept either
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
