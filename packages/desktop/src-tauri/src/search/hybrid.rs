//! Hybrid search: keyword (FTS5) + dense vector retrieval, fused by reciprocal
//! rank fusion at the chunk level and aggregated up to messages.
//!
//! Vector search is deliberately implemented in Rust by scanning all chunks
//! that have an embedding and computing cosine similarity. For mailbox-sized
//! corpora (tens of thousands of chunks) this is fast enough and avoids
//! shipping a sqlite-vec extension. If the corpus grows we can swap this for
//! `sqlite-vec` or `lance` without changing the public API.

use std::collections::HashMap;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::db::queries::{self, ChunkWithEmbedding, MessageRow};
use crate::db::Database;
use crate::error::AppResult;
use crate::models::OllamaClient;

#[derive(Debug, Clone, Serialize)]
pub struct MessageHit {
    pub message_id: String,
    pub account_id: String,
    pub thread_id: Option<String>,
    pub rfc822_message_id: Option<String>,
    pub subject: String,
    pub sender: String,
    pub sender_email: Option<String>,
    pub recipients: String,
    pub date: String,
    pub snippet: String,
    pub body_preview: String,
    pub similarity: Option<f32>,
    pub keyword_score: Option<f32>,
    pub combined_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchOutcome {
    pub query: String,
    pub results: Vec<MessageHit>,
    pub total: usize,
    pub used_vector: bool,
    pub used_keyword: bool,
    pub took_ms: u64,
}

const RRF_K: f32 = 60.0;
// Lexical hits beat semantic hits on ties. Most user queries are at least
// partly keyword-shaped ("tldr", a name, an order number) — a noisy nearest-
// neighbor at rank 0 shouldn't outrank a perfect FTS match. Tuned so a top
// keyword hit dominates a single top vector hit but doesn't drown out a
// message that hits BOTH modalities.
const KW_RRF_WEIGHT: f32 = 2.0;
const VEC_RRF_WEIGHT: f32 = 1.0;
// Vector candidates below this cosine similarity are dropped before fusion.
// Otherwise sparse-corpus noise (e.g. before background embedding catches up)
// can rank above real matches just by being "the least-bad of what's
// embedded".
const VEC_SIM_FLOOR: f32 = 0.5;
// Recency half-life in days for the exponential decay applied to the fused
// score. A message sent τ days ago is worth half a freshly sent one. Email
// is intrinsically time-sensitive — "important" almost always implies recent.
const RECENCY_TAU_DAYS: f32 = 180.0;
// Score multiplier applied to bulk/newsletter mail. <1 demotes; 0 would hide
// them entirely. 0.4 lets bulk still surface on overwhelming keyword/vector
// match (e.g. a query *about* a newsletter) without dominating real mail.
const BULK_PENALTY: f32 = 0.4;

pub async fn search(
    db: &Database,
    ollama: Option<&OllamaClient>,
    query: &str,
    limit: usize,
) -> AppResult<SearchOutcome> {
    let started = Instant::now();
    let query_trim = query.trim();
    if query_trim.is_empty() {
        return Ok(SearchOutcome {
            query: query_trim.to_string(),
            results: vec![],
            total: 0,
            used_vector: false,
            used_keyword: false,
            took_ms: 0,
        });
    }

    // Wider than just `limit * 5` so a recent, non-bulk message buried at rank
    // ~30 still has a chance to climb back into the top-K after demotion.
    let candidate_pool = (limit * 10).max(80);

    // Keyword pass.
    let keyword_chunks: Vec<(String, f64)> = {
        let conn = db.read()?;
        queries::search_fts(&conn, query_trim, candidate_pool as i64)?
    };
    let used_keyword = !keyword_chunks.is_empty();

    // Vector pass (best-effort: a missing model or unreachable Ollama just
    // skips the dense step).
    let mut vector_chunks: Vec<(String, f32)> = Vec::new();
    let mut used_vector = false;
    if let Some(ollama) = ollama {
        match ollama.embed(query_trim).await {
            Ok(qvec) => {
                let chunks = {
                    let conn = db.read()?;
                    queries::all_chunks_with_embeddings(&conn)?
                };
                if !chunks.is_empty() {
                    used_vector = true;
                    vector_chunks = vector_top_k(&qvec, &chunks, candidate_pool);
                }
            }
            Err(e) => {
                tracing::warn!(?e, "query embedding failed; falling back to keyword only");
            }
        }
    }

    // Fuse at the message level using weighted reciprocal rank fusion.
    let mut fused: HashMap<String, FusedScore> = HashMap::new();
    for (rank, (msg_id, _bm25)) in keyword_chunks.iter().enumerate() {
        let entry = fused.entry(msg_id.clone()).or_default();
        entry.score += KW_RRF_WEIGHT / (RRF_K + rank as f32 + 1.0);
        // Track the best (lowest) rank across all matching chunks for this
        // message, not the last one we happened to iterate.
        entry.keyword_rank = Some(
            entry
                .keyword_rank
                .map(|r| r.min(rank))
                .unwrap_or(rank),
        );
    }
    for (rank, (msg_id, sim)) in vector_chunks.iter().enumerate() {
        if *sim < VEC_SIM_FLOOR {
            continue;
        }
        let entry = fused.entry(msg_id.clone()).or_default();
        entry.score += VEC_RRF_WEIGHT / (RRF_K + rank as f32 + 1.0);
        entry.similarity = Some(entry.similarity.map(|s| s.max(*sim)).unwrap_or(*sim));
        entry.vector_rank = Some(rank);
    }

    if fused.is_empty() {
        return Ok(SearchOutcome {
            query: query_trim.to_string(),
            results: vec![],
            total: 0,
            used_vector,
            used_keyword,
            took_ms: started.elapsed().as_millis() as u64,
        });
    }

    // Fetch the message rows for the candidate pool (not yet truncated to
    // `limit`) so we can apply recency / bulk multipliers — these may reorder
    // results enough that a candidate outside the original top-K ends up in
    // the final top-K.
    let ids: Vec<String> = fused.keys().cloned().collect();
    let messages = {
        let conn = db.read()?;
        queries::fetch_messages(&conn, &ids)?
    };
    let by_id: HashMap<String, MessageRow> =
        messages.into_iter().map(|m| (m.id.clone(), m)).collect();

    let now = Utc::now();
    let mut ranked: Vec<(String, FusedScore)> = fused
        .into_iter()
        .map(|(id, mut s)| {
            if let Some(m) = by_id.get(&id) {
                s.score *= recency_factor(&m.date, now);
                if m.is_bulk {
                    s.score *= BULK_PENALTY;
                }
            }
            (id, s)
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit);

    let kw_total = keyword_chunks.len().max(1) as f32;
    let results: Vec<MessageHit> = ranked
        .into_iter()
        .filter_map(|(id, fused)| {
            by_id.get(&id).map(|m| MessageHit {
                message_id: m.id.clone(),
                account_id: m.account_id.clone(),
                thread_id: m.thread_id.clone(),
                rfc822_message_id: m.rfc822_message_id.clone(),
                subject: m.subject.clone(),
                sender: m.sender.clone(),
                sender_email: m.sender_email.clone(),
                recipients: m.recipients.clone(),
                date: m.date.clone(),
                snippet: m.snippet.clone(),
                body_preview: m.body_preview.clone(),
                similarity: fused.similarity,
                keyword_score: fused
                    .keyword_rank
                    .map(|r| 1.0 - (r as f32 / kw_total)),
                combined_score: fused.score,
            })
        })
        .collect();

    Ok(SearchOutcome {
        query: query_trim.to_string(),
        total: results.len(),
        results,
        used_vector,
        used_keyword,
        took_ms: started.elapsed().as_millis() as u64,
    })
}

#[derive(Debug, Default, Clone)]
struct FusedScore {
    score: f32,
    similarity: Option<f32>,
    keyword_rank: Option<usize>,
    vector_rank: Option<usize>,
}

/// Exponential decay by message age. A blank or unparseable date returns 1.0
/// so undated mail is never silently demoted to zero.
fn recency_factor(sent_at: &str, now: DateTime<Utc>) -> f32 {
    if sent_at.is_empty() {
        return 1.0;
    }
    let sent = match DateTime::parse_from_rfc3339(sent_at) {
        Ok(d) => d.with_timezone(&Utc),
        Err(_) => return 1.0,
    };
    let age_days = (now - sent).num_days().max(0) as f32;
    (-age_days / RECENCY_TAU_DAYS).exp()
}

/// Cosine-similarity scan. For each chunk we keep the best chunk per message,
/// which is what we care about (a single matching paragraph is enough to
/// surface a long email).
fn vector_top_k(
    query: &[f32],
    chunks: &[ChunkWithEmbedding],
    limit: usize,
) -> Vec<(String, f32)> {
    let mut best_per_message: HashMap<String, (String, f32)> = HashMap::new();
    let qnorm = norm(query);
    if qnorm == 0.0 {
        return vec![];
    }
    for chunk in chunks {
        if chunk.embedding.len() != query.len() {
            continue;
        }
        let sim = cosine_pre(query, qnorm, &chunk.embedding);
        match best_per_message.get_mut(&chunk.message_id) {
            Some(existing) if existing.1 >= sim => {}
            _ => {
                best_per_message
                    .insert(chunk.message_id.clone(), (chunk.chunk_id.clone(), sim));
            }
        }
    }
    let mut v: Vec<(String, f32)> = best_per_message
        .into_iter()
        .map(|(k, (_chunk, sim))| (k, sim))
        .collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(limit);
    v
}

fn cosine_pre(q: &[f32], qnorm: f32, v: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut vn = 0.0f32;
    for i in 0..v.len() {
        dot += q[i] * v[i];
        vn += v[i] * v[i];
    }
    if vn == 0.0 {
        return 0.0;
    }
    dot / (qnorm * vn.sqrt())
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_top_k_returns_best_per_message() {
        let q = vec![1.0, 0.0];
        let chunks = vec![
            ChunkWithEmbedding {
                chunk_id: "c1".into(),
                message_id: "m1".into(),
                chunk_index: 0,
                text: "".into(),
                embedding: vec![1.0, 0.0],
            },
            ChunkWithEmbedding {
                chunk_id: "c2".into(),
                message_id: "m1".into(),
                chunk_index: 1,
                text: "".into(),
                embedding: vec![0.5, 0.5],
            },
            ChunkWithEmbedding {
                chunk_id: "c3".into(),
                message_id: "m2".into(),
                chunk_index: 0,
                text: "".into(),
                embedding: vec![0.0, 1.0],
            },
        ];
        let results = vector_top_k(&q, &chunks, 5);
        assert_eq!(results[0].0, "m1");
        // Best chunk for m1 is c1 (perfectly aligned with q), so similarity ~ 1.
        assert!(results[0].1 > 0.99);
        assert_eq!(results[1].0, "m2");
        assert!(results[1].1.abs() < 1e-5);
    }
}
