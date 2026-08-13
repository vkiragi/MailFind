use std::time::Instant;

use chrono::DateTime;
use serde::Serialize;

use crate::db::Database;
use crate::error::{AppError, AppResult};
use crate::models::{ChatMessage, OllamaClient};
use crate::search::{self, MessageHit};

#[derive(Debug, Clone, Serialize)]
pub struct AnswerCitation {
    pub message_id: String,
    pub rfc822_message_id: Option<String>,
    pub subject: String,
    pub sender: String,
    pub date: String,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerOutcome {
    pub question: String,
    pub answer: String,
    pub model: String,
    pub citations: Vec<AnswerCitation>,
    pub took_ms: u64,
}

const SYSTEM_PROMPT: &str = "You are MailFind, an assistant that answers \
questions strictly using the user's local email. Always cite the most relevant \
emails by their numeric reference like [1], [2]. If the emails do not contain \
the answer, say so plainly instead of guessing. Keep responses concise. \
Each email is labeled with its Date. When the question is about what is new, \
recent, upcoming, or latest, lead with the most recent email and present items \
newest-first, stating each item's date — do not assume the [1], [2] order \
reflects recency.";

pub async fn ask(
    db: &Database,
    ollama: &OllamaClient,
    question: &str,
    top_k: usize,
) -> AppResult<AnswerOutcome> {
    let started = Instant::now();
    let q = question.trim();
    if q.is_empty() {
        return Err(AppError::InvalidInput("empty question".into()));
    }

    let mut outcome = search::search(db, Some(ollama), q, top_k).await?;
    // For recency-flavored questions ("new", "latest", "upcoming"…), reorder the
    // retrieved emails newest-first before numbering them, so [1] is the most
    // recent. This makes the answer's ordering correct even for a small model
    // that just walks [1]→[n], and keeps prompt numbers aligned with the
    // citations shown in the UI (both derive from `outcome.results`).
    if is_recency_question(q) {
        outcome
            .results
            .sort_by(|a, b| date_key(&b.date).cmp(&date_key(&a.date)));
    }
    if outcome.results.is_empty() {
        return Ok(AnswerOutcome {
            question: q.to_string(),
            answer: "I could not find any relevant emails locally.".to_string(),
            model: ollama.config.chat_model.clone(),
            citations: vec![],
            took_ms: started.elapsed().as_millis() as u64,
        });
    }

    let prompt = build_user_prompt(q, &outcome.results);
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: SYSTEM_PROMPT.to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: prompt,
        },
    ];

    let answer = ollama.chat(messages).await?;
    let citations = outcome
        .results
        .iter()
        .take(top_k)
        .map(|h| AnswerCitation {
            message_id: h.message_id.clone(),
            rfc822_message_id: h.rfc822_message_id.clone(),
            subject: h.subject.clone(),
            sender: h.sender.clone(),
            date: h.date.clone(),
            snippet: h.snippet.clone(),
        })
        .collect();

    Ok(AnswerOutcome {
        question: q.to_string(),
        answer,
        model: ollama.config.chat_model.clone(),
        citations,
        took_ms: started.elapsed().as_millis() as u64,
    })
}

fn build_user_prompt(question: &str, hits: &[MessageHit]) -> String {
    let mut buf = String::new();
    buf.push_str("Question: ");
    buf.push_str(question);
    buf.push_str("\n\nRelevant emails:\n");
    for (i, hit) in hits.iter().enumerate() {
        let n = i + 1;
        buf.push_str(&format!(
            "\n[{n}] From: {sender}\n    Subject: {subject}\n    Date: {date}\n    Excerpt: {snippet}\n",
            n = n,
            sender = clip(&hit.sender, 120),
            subject = clip(&hit.subject, 200),
            date = hit.date,
            snippet = clip(&hit.body_preview.is_empty().then(|| hit.snippet.clone()).unwrap_or_else(|| hit.body_preview.clone()), 800),
        ));
    }
    buf.push_str(
        "\nAnswer the question using only these emails. Cite emails as [1], [2], etc. \
         If the emails are insufficient, say so.",
    );
    buf
}

/// Whether the question is asking about recency ("new", "latest", "upcoming"),
/// in which case we present retrieved emails newest-first. Deliberately narrow:
/// relevance-only questions keep their retrieval-score ordering.
fn is_recency_question(q: &str) -> bool {
    const TERMS: &[&str] = &[
        "new", "recent", "latest", "newest", "upcoming", "lately", "so far",
        "this week", "these days", "current", "any updates",
    ];
    let lower = q.to_lowercase();
    TERMS.iter().any(|t| lower.contains(t))
}

/// Sortable key from a hit's date string. Unparseable/empty dates sort oldest so
/// dated mail always leads a newest-first ordering.
fn date_key(date: &str) -> i64 {
    DateTime::parse_from_rfc3339(date)
        .map(|d| d.timestamp())
        .unwrap_or(i64::MIN)
}

fn clip(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max).collect();
        out.push('…');
        out
    }
}
