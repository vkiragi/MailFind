//! Hybrid search: keyword (FTS5) + dense vector retrieval, fused by reciprocal
//! rank fusion at the chunk level and aggregated up to messages.
//!
//! Vector search is deliberately implemented in Rust by scanning all chunks
//! that have an embedding and computing cosine similarity. For mailbox-sized
//! corpora (tens of thousands of chunks) this is fast enough and avoids
//! shipping a sqlite-vec extension. If the corpus grows we can swap this for
//! `sqlite-vec` or `lance` without changing the public API.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::{DateTime, Utc};
use serde::Serialize;

use crate::db::queries::{self, MessageRow};
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
    /// Whether this hit was flagged bulk/newsletter mail (see `is_bulk` on
    /// `MessageRow`). Surfaced so the UI's confidence indicator can explain a
    /// surprising rank — e.g. "this scored lower because it's flagged bulk".
    pub is_bulk: bool,
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
const KW_RRF_WEIGHT: f32 = 1.0;
const VEC_RRF_WEIGHT: f32 = 1.0;
// Vector candidates below this cosine similarity are dropped before fusion.
// Otherwise sparse-corpus noise (e.g. before background embedding catches up)
// can rank above real matches just by being "the least-bad of what's
// embedded".
const VEC_SIM_FLOOR: f32 = 0.5;
// Recency time constant in days for the exponential decay applied to the
// fused score. Email is somewhat time-sensitive — recent mail should win
// among comparable matches.
const RECENCY_TAU_DAYS: f32 = 180.0;
// Lower bound on the recency factor: fresh mail is worth at most
// 1/RECENCY_FLOOR (2×) an arbitrarily old message. RRF scores are flat (rank
// 0 is only ~1.5× rank 30), so an unbounded decay makes age the dominant sort
// key — an 18-month-old exact match (×0.05) loses to every recent partial
// match, and archival mail becomes unfindable. The factor blends toward the
// floor instead of clipping so ordering among old mail is still age-aware.
const RECENCY_FLOOR: f32 = 0.5;
// Score multiplier applied to bulk/newsletter mail. <1 demotes; 0 would hide
// them entirely. 0.4 lets bulk still surface on overwhelming keyword/vector
// match (e.g. a query *about* a newsletter) without dominating real mail.
const BULK_PENALTY: f32 = 0.4;
// A bulk-flagged message with a keyword match this strong (normalized rank,
// 1.0 = best in the candidate pool) is exempt from BULK_PENALTY. Without
// this, searching a brand/company name by name can fail: that brand's own
// promotional mail is a near-perfect keyword hit but IS "bulk" by definition,
// and the bulk penalty compounding with recency decay (e.g. an older message
// already at ~0.6x from age) can flip a much weaker, fresher, non-bulk match
// above it (measured: "CHANEL" ranked a Half Baked newsletter — kw=0.11, not
// bulk-flagged, 3 months old — above a near-perfect-match CHANEL promo email
// — kw=0.94, bulk-flagged, 10 months old — combined penalty ~4x). Being
// "bulk" means "don't surface this when the user didn't ask for it", not
// "demote it even when they explicitly searched for exactly this".
const BULK_PENALTY_EXEMPT_KEYWORD_SCORE: f32 = 0.85;

pub async fn search(
    db: &Database,
    ollama: Option<&OllamaClient>,
    query: &str,
    limit: usize,
    // Preloaded embedding snapshot (from `AppState::embeddings_snapshot`). When
    // `None` — e.g. the example binaries — embeddings are loaded from SQLite,
    // preserving the old per-call behavior.
    embeddings: Option<Arc<Vec<(String, Vec<f32>)>>>,
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
                let chunks = match embeddings {
                    Some(cached) => cached,
                    None => Arc::new({
                        let conn = db.read()?;
                        queries::all_message_embeddings(&conn)?
                    }),
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
    for (msg_id, rank) in best_rank_per_message(&keyword_chunks) {
        let entry = fused.entry(msg_id).or_default();
        entry.score += KW_RRF_WEIGHT / (RRF_K + rank as f32 + 1.0);
        entry.keyword_rank = Some(rank);
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

    // Computed here (not just at result-building time) so the bulk-penalty
    // exemption below uses the exact same normalized keyword strength that's
    // ultimately shown to the caller as `keyword_score`.
    let kw_total = keyword_chunks.len().max(1) as f32;
    let now = Utc::now();
    let mut ranked: Vec<(String, FusedScore)> = fused
        .into_iter()
        .map(|(id, mut s)| {
            if let Some(m) = by_id.get(&id) {
                s.score *= recency_factor(&m.date, now);
                if m.is_bulk {
                    let kw_strength = s.keyword_rank.map(|r| 1.0 - (r as f32 / kw_total));
                    let exempt = kw_strength
                        .map(|k| k >= BULK_PENALTY_EXEMPT_KEYWORD_SCORE)
                        .unwrap_or(false);
                    if !exempt {
                        s.score *= BULK_PENALTY;
                    }
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
                is_bulk: m.is_bulk,
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
    RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * (-age_days / RECENCY_TAU_DAYS).exp()
}

/// Cosine-similarity scan. For each chunk we keep the best chunk per message,
/// which is what we care about (a single matching paragraph is enough to
/// surface a long email).
/// Collapses the keyword pass's per-*chunk* hits to one entry per message,
/// keeping each message's best (lowest) rank.
///
/// The FTS pass ranks chunks, so a long message can appear many times. Adding
/// an RRF term per chunk would score length instead of relevance — a 10-chunk
/// marketing blast accumulates ten contributions and outranks a short,
/// on-topic message with one. `vector_top_k` already returns best-per-message,
/// so this keeps both rankers symmetric, which is what RRF assumes.
fn best_rank_per_message(chunks: &[(String, f64)]) -> HashMap<String, usize> {
    let mut best: HashMap<String, usize> = HashMap::new();
    for (rank, (msg_id, _bm25)) in chunks.iter().enumerate() {
        best.entry(msg_id.clone())
            .and_modify(|r| *r = (*r).min(rank))
            .or_insert(rank);
    }
    best
}

fn vector_top_k(
    query: &[f32],
    chunks: &[(String, Vec<f32>)],
    limit: usize,
) -> Vec<(String, f32)> {
    let qnorm = norm(query);
    if qnorm == 0.0 {
        return vec![];
    }
    // Normalize the query once; cached embeddings are already unit vectors, so
    // cosine similarity is just their dot product.
    let q: Vec<f32> = query.iter().map(|x| x / qnorm).collect();
    let mut best_per_message: HashMap<String, f32> = HashMap::new();
    for (message_id, embedding) in chunks {
        if embedding.len() != q.len() {
            continue;
        }
        let mut sim = 0.0f32;
        for i in 0..q.len() {
            sim += q[i] * embedding[i];
        }
        // Clone the id only when this message is seen for the first time, not on
        // every one of its chunks.
        match best_per_message.get_mut(message_id) {
            Some(best) => {
                if sim > *best {
                    *best = sim;
                }
            }
            None => {
                best_per_message.insert(message_id.clone(), sim);
            }
        }
    }
    let mut v: Vec<(String, f32)> = best_per_message.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    v.truncate(limit);
    v
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recency_factor_blends_toward_floor_for_old_mail() {
        let now = Utc::now();
        let fresh = (now - chrono::Duration::days(1)).to_rfc3339();
        let old = (now - chrono::Duration::days(730)).to_rfc3339();
        let f_fresh = recency_factor(&fresh, now);
        let f_old = recency_factor(&old, now);
        assert!(f_fresh > 0.95, "fresh mail should be near 1.0: {f_fresh}");
        assert!(f_old >= RECENCY_FLOOR, "old mail never below floor: {f_old}");
        assert!(f_old < f_fresh, "recency still prefers fresh mail");
        // Undated mail is never demoted.
        assert_eq!(recency_factor("", now), 1.0);
    }

    #[test]
    fn keyword_rank_collapses_chunks_to_best_per_message() {
        // `m_long` matches on three chunks (ranks 0, 2, 3); `m_short` on one.
        let chunks = vec![
            ("m_long".to_string(), 9.0),
            ("m_short".to_string(), 8.0),
            ("m_long".to_string(), 7.0),
            ("m_long".to_string(), 6.0),
        ];
        let best = best_rank_per_message(&chunks);
        assert_eq!(best.len(), 2, "one entry per message, not per chunk");
        assert_eq!(best["m_long"], 0, "keeps the best (lowest) rank");
        assert_eq!(best["m_short"], 1);

        // Regression: a message must not earn extra RRF weight just for being
        // long. Summing a term per chunk gave `m_long` ~3x its due, which let
        // verbose marketing mail outrank short, on-topic messages.
        let rrf = |r: usize| KW_RRF_WEIGHT / (RRF_K + r as f32 + 1.0);
        let deduped = rrf(best["m_long"]);
        let per_chunk = rrf(0) + rrf(2) + rrf(3);
        assert!(
            deduped < per_chunk,
            "per-chunk accumulation inflates long messages: {deduped} vs {per_chunk}"
        );
        assert!(
            (deduped - rrf(0)).abs() < f32::EPSILON,
            "a message contributes exactly one term, at its best rank"
        );
    }

    #[test]
    fn vector_top_k_returns_best_per_message() {
        let q = vec![1.0, 0.0];
        let chunks = vec![
            ("m1".to_string(), vec![1.0, 0.0]),
            ("m1".to_string(), vec![0.5, 0.5]),
            ("m2".to_string(), vec![0.0, 1.0]),
        ];
        let results = vector_top_k(&q, &chunks, 5);
        assert_eq!(results[0].0, "m1");
        // Best chunk for m1 is c1 (perfectly aligned with q), so similarity ~ 1.
        assert!(results[0].1 > 0.99);
        assert_eq!(results[1].0, "m2");
        assert!(results[1].1.abs() < 1e-5);
    }
}
