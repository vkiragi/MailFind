//! Search-quality evaluation harness. Runs a fixed set of labeled queries
//! against the live DB and reports precision plus specific failure modes, so
//! ranking changes can be compared with a number instead of a vibe.
//!
//!   cargo run --release --example eval_search          # summary
//!   cargo run --release --example eval_search -- -v    # per-result detail
//!
//! Relevance labels are RULE-BASED APPROXIMATIONS, not hand-judged ground
//! truth: a hit counts as relevant if its sender or subject matches that
//! query's `relevant` patterns, and as junk if it matches `junk`. They encode
//! "obviously right" and "obviously wrong" for this corpus — good enough to
//! catch regressions, not a substitute for reading results.

use mailfind_lib::db::Database;
use mailfind_lib::models::{OllamaClient, OllamaConfig};
use mailfind_lib::search;
use mailfind_lib::state;

struct Case {
    query: &'static str,
    /// What this query is probing, for the report.
    shape: &'static str,
    /// Lowercase substrings; a hit matching any (in sender or subject) is relevant.
    relevant: &'static [&'static str],
    /// Lowercase substrings that mark an obviously-wrong hit.
    junk: &'static [&'static str],
}

const CASES: &[Case] = &[
    Case {
        query: "music production tips",
        shape: "conceptual (few exact keywords)",
        relevant: &[
            "adsrsounds", "busyworksbeats", "splice", "hitproducerstash",
            "mix", "mastering", "music", "beat", "producer", "sound design",
        ],
        junk: &["subimods", "exhaust", "awe tuning", "23andme"],
    },
    Case {
        query: "flight confirmations",
        shape: "transactional + lexical",
        relevant: &[
            "flyfrontier", "aa.com", "emirates", "southwest", "jal", "united",
            "flight", "trip", "booking", "itinerary", "boarding",
        ],
        junk: &["subimods", "newsletter", "unsubscribe to stop"],
    },
    Case {
        query: "online assessment invitation",
        shape: "transactional, job-related",
        relevant: &[
            "coderbyte", "workday", "shl.com", "amazon", "assessment",
            "invite", "invitation", "hackerrank", "codesignal",
        ],
        junk: &["subimods", "exhaust"],
    },
    Case {
        query: "receipts and invoices",
        shape: "conceptual, financial",
        relevant: &[
            "paypal", "stripe", "receipt", "invoice", "payment", "order",
            "billing", "purchase", "charged",
        ],
        junk: &["exhaust", "assessment"],
    },
    Case {
        query: "subscription renewals",
        shape: "conceptual, financial",
        relevant: &[
            "subscription", "renew", "billing", "plan", "membership",
            "paypal", "stripe", "invoice", "payment",
        ],
        junk: &["exhaust", "flight"],
    },
];

/// How many top hits each case is scored over.
const AT_K: usize = 5;

#[tokio::main]
async fn main() {
    let verbose = std::env::args().any(|a| a == "-v" || a == "--verbose");

    let data_dir = state::default_data_dir().expect("data dir");
    let db = Database::open(&data_dir.join("mailfind.sqlite")).expect("open db");
    let ollama = OllamaClient::new(OllamaConfig::new(
        "http://127.0.0.1:11434".to_string(),
        "nomic-embed-text".to_string(),
        "qwen3:8b".to_string(),
    ))
    .expect("ollama client");

    let mut total_rel = 0usize;
    let mut total_junk = 0usize;
    let mut total_scored = 0usize;
    let mut junk_at_1 = 0usize;
    let mut no_evidence = 0usize;

    println!("\n=== search quality @{AT_K} ===\n");

    for case in CASES {
        let outcome = search::search(&db, Some(&ollama), case.query, AT_K, None)
            .await
            .expect("search");

        let mut rel = 0usize;
        let mut junk = 0usize;
        let mut lines: Vec<String> = Vec::new();

        for (i, hit) in outcome.results.iter().take(AT_K).enumerate() {
            let hay = format!(
                "{} {}",
                hit.subject.to_lowercase(),
                hit.sender_email.clone().unwrap_or_default().to_lowercase()
            );
            let is_rel = case.relevant.iter().any(|p| hay.contains(p));
            let is_junk = case.junk.iter().any(|p| hay.contains(p));
            if is_junk {
                junk += 1;
                if i == 0 {
                    junk_at_1 += 1;
                }
            } else if is_rel {
                rel += 1;
            }
            // A hit with neither semantic support nor a strong keyword score is
            // riding on weak evidence — the shape of the length-bias bug.
            if hit.similarity.is_none() && hit.keyword_score.unwrap_or(0.0) < 0.85 {
                no_evidence += 1;
            }
            let mark = if is_junk {
                "JUNK"
            } else if is_rel {
                "ok  "
            } else {
                "?   "
            };
            lines.push(format!(
                "    {mark} {:.4} sim={} kw={} | {}",
                hit.combined_score,
                hit.similarity
                    .map(|s| format!("{s:.2}"))
                    .unwrap_or_else(|| "  - ".into()),
                hit.keyword_score
                    .map(|s| format!("{s:.2}"))
                    .unwrap_or_else(|| "  - ".into()),
                hit.subject.chars().take(64).collect::<String>(),
            ));
        }

        let scored = outcome.results.len().min(AT_K);
        total_rel += rel;
        total_junk += junk;
        total_scored += scored;

        let p = if scored > 0 {
            rel as f32 / scored as f32
        } else {
            0.0
        };
        println!(
            "  P@{AT_K}={:.0}%  junk={junk}  | {:<32} [{}]",
            p * 100.0,
            case.query,
            case.shape
        );
        if verbose {
            for l in lines {
                println!("{l}");
            }
            println!();
        }
    }

    let p_overall = if total_scored > 0 {
        total_rel as f32 / total_scored as f32
    } else {
        0.0
    };
    println!("\n  ---------------------------------------------");
    println!("  precision@{AT_K}      {:.1}%  ({total_rel}/{total_scored})", p_overall * 100.0);
    println!("  junk hits         {total_junk}");
    println!("  junk at rank 1    {junk_at_1}");
    println!("  weak-evidence     {no_evidence}  (no vector support and kw < 0.85)");
    println!();
}
