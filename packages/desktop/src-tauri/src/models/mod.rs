//! Local model integration. v1 uses Ollama for both `nomic-embed-text`
//! embeddings and Qwen-class chat completions.

pub mod ollama;
pub mod select;

pub use ollama::{ChatMessage, ModelHealth, OllamaClient, OllamaConfig};
