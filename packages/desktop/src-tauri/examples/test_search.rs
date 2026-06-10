//! Run hybrid search against the live DB without going through the Tauri UI.
//! Used to verify recency + bulk-mail demote changes.
//!
//!   cargo run --example test_search -- "interview"
//!   cargo run --example test_search -- "tldr"

use std::env;

use mailfind_lib::db::Database;
use mailfind_lib::models::{OllamaClient, OllamaConfig};
use mailfind_lib::search::hybrid;
use mailfind_lib::state;

#[tokio::main]
async fn main() {
    let query: String = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .join(" ");
    let query = if query.is_empty() {
        "interview".to_string()
    } else {
        query
    };

    let data_dir = state::default_data_dir().expect("data dir");
    let db_path = data_dir.join("mailfind.sqlite");
    let db = Database::open(&db_path).expect("open db");

    let ollama_cfg = OllamaConfig::new(
        "http://127.0.0.1:11434".to_string(),
        "nomic-embed-text".to_string(),
        "granite4.1:3b".to_string(),
    );
    let ollama = OllamaClient::new(ollama_cfg).expect("ollama client");

    println!("\n=== query: {:?} ===", query);
    let out = hybrid::search(&db, Some(&ollama), &query, 10)
        .await
        .expect("search");
    println!(
        "took={}ms used_keyword={} used_vector={} total={}",
        out.took_ms, out.used_keyword, out.used_vector, out.total
    );
    for (i, hit) in out.results.iter().enumerate() {
        let date = if hit.date.len() >= 10 {
            &hit.date[..10]
        } else {
            &hit.date
        };
        let subj = if hit.subject.chars().count() > 70 {
            hit.subject.chars().take(67).collect::<String>() + "..."
        } else {
            hit.subject.clone()
        };
        let sender_email = hit.sender_email.clone().unwrap_or_default();
        println!(
            "{:>2}. score={:.4} sim={:>5} kw={:>5} | {} | {} | {}",
            i + 1,
            hit.combined_score,
            hit.similarity
                .map(|s| format!("{:.2}", s))
                .unwrap_or_else(|| "-".into()),
            hit.keyword_score
                .map(|s| format!("{:.2}", s))
                .unwrap_or_else(|| "-".into()),
            date,
            sender_email,
            subj,
        );
    }
}
