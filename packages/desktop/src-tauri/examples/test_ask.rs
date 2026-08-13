//! Run the Ask (RAG) pipeline against the live DB with a chosen chat model.
//! Same retrieval + prompt as the app, so it isolates model behavior — use it
//! to compare answer quality/ordering across models without touching settings.
//!
//!   cargo run --example test_ask -- granite4.1:3b "do I have any new online assessments"
//!   cargo run --example test_ask -- llama3.1:8b  "do I have any new online assessments"

use std::env;

use mailfind_lib::db::Database;
use mailfind_lib::models::{OllamaClient, OllamaConfig};
use mailfind_lib::qa;
use mailfind_lib::state;

#[tokio::main]
async fn main() {
    let mut args = env::args().skip(1);
    let model = args.next().unwrap_or_else(|| "granite4.1:3b".to_string());
    let question = args.collect::<Vec<_>>().join(" ");
    let question = if question.is_empty() {
        "do I have any new online assessments or interviews".to_string()
    } else {
        question
    };

    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open db");

    let ollama = OllamaClient::new(OllamaConfig::new(
        "http://127.0.0.1:11434".to_string(),
        "nomic-embed-text".to_string(),
        model.clone(),
    ))
    .expect("ollama client");

    println!("\n=== model: {model} ===\n=== question: {question:?} ===\n");
    let out = qa::ask(&db, &ollama, &question, 8).await.expect("ask");

    println!("--- answer ({} ms) ---\n{}\n", out.took_ms, out.answer);
    println!("--- citations (in retrieval order) ---");
    for (i, c) in out.citations.iter().enumerate() {
        println!("[{}] {} | {} | {}", i + 1, c.date, c.sender, c.subject);
    }
}
