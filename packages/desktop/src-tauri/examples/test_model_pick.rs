//! Prints the live chat-model auto-pick decision: total RAM, the model budget,
//! the installed Ollama tags, and what `select::auto_pick` chooses from them.
//! Use it to sanity-check the RAM path and tag-format matching without the UI.
//!
//!   cargo run --example test_model_pick

use mailfind_lib::models::select::{self, AutoPick};
use mailfind_lib::models::{OllamaClient, OllamaConfig};

#[tokio::main]
async fn main() {
    let ollama = OllamaClient::new(OllamaConfig::new(
        "http://127.0.0.1:11434".to_string(),
        "nomic-embed-text".to_string(),
        "qwen3:8b".to_string(),
    ))
    .expect("ollama client");

    let total = select::total_ram_gb();
    let budget = select::model_budget_gb(total);
    println!("total RAM:    {total:.1} GB");
    println!("model budget: {budget:.1} GB");

    let installed = match ollama.list_models().await {
        Ok(m) => m,
        Err(e) => {
            println!("Ollama unreachable: {e}");
            return;
        }
    };
    println!("installed:    {installed:?}\n");

    println!("recommended ladder (best-first):");
    for rec in select::RECOMMENDED {
        let fits = rec.needs_gb <= budget;
        println!(
            "  {:<18} needs {:>4.1} GB  auto={}  {}{}",
            rec.model,
            rec.needs_gb,
            rec.auto,
            if fits { "fits" } else { "too big" },
            rec.warn.map(|w| format!("  ⚠ {w}")).unwrap_or_default(),
        );
    }

    match select::auto_pick(budget, &installed) {
        AutoPick::Model(m) => println!("\n=> auto-pick: use {m}"),
        AutoPick::NeedsPull(m) => println!("\n=> auto-pick: needs setup — run `ollama pull {m}`"),
        AutoPick::SearchOnly => println!("\n=> auto-pick: search-only (RAM below Ask floor)"),
    }
}
