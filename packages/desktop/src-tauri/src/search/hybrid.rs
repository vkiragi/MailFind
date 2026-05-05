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

    let candidate_pool = (limit * 5).max(40);

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

    // Fuse at the message level using reciprocal rank fusion.
    let mut fused: HashMap<String, FusedScore> = HashMap::new();
    for (rank, (msg_id, _bm25)) in keyword_chunks.iter().enumerate() {
        let entry = fused.entry(msg_id.clone()).or_default();
        entry.score += 1.0 / (RRF_K + rank as f32 + 1.0);
        entry.keyword_rank = Some(rank);
    }
    for (rank, (msg_id, sim)) in vector_chunks.iter().enumerate() {
        let entry = fused.entry(msg_id.clone()).or_default();
        entry.score += 1.0 / (RRF_K + rank as f32 + 1.0);
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

    let mut ranked: Vec<(String, FusedScore)> = fused.into_iter().collect();
    ranked.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked.truncate(limit);

    let ids: Vec<String> = ranked.iter().map(|(id, _)| id.clone()).collect();
    let messages = {
        let conn = db.read()?;
        queries::fetch_messages(&conn, &ids)?
    };
    let by_id: HashMap<String, MessageRow> =
        messages.into_iter().map(|m| (m.id.clone(), m)).collect();

    let kw_total = keyword_chunks.len().max(1) as f32;
    let results: Vec<MessageHit> = ranked
        .into_iter()
        .filter_map(|(id, fused)| {
            by_id.get(&id).map(|m| MessageHit {
                message_id: m.id.clone(),
                account_id: m.account_id.clone(),
                thread_id: m.thread_id.clone(),
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
