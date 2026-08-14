//! Chat-model recommendation and auto-pick — the "graceful degradation" core.
//!
//! MailFind's semantic search needs only ~1.5GB and runs anywhere; the RAM cost
//! is entirely the Ask/chat LLM. This module maps a machine's total RAM to the
//! best chat model it can comfortably run, and picks the best one that BOTH fits
//! the budget AND is already installed in Ollama. No in-app pulling: if a model
//! fits but isn't installed, we surface the `ollama pull` command instead.
//!
//! Pure and DB-free on purpose (unit-testable without a live Ollama/SQLite).
//! The startup orchestration that wires this to the client + settings lives in
//! `commands::auto_pick_chat_model`.

/// RAM (GiB) reserved for MailFind itself: the embedder model, the resident
/// embedding cache (~850MB), and app/runtime overhead.
const RESERVED_APP_GB: f64 = 2.0;
/// RAM (GiB) reserved for the OS and general headroom, so picking a model never
/// pushes the machine into swap.
const RESERVED_OS_GB: f64 = 5.0;

/// A chat model MailFind knows how to recommend, with the RAM it needs loaded.
pub struct Recommendation {
    /// Ollama model tag.
    pub model: &'static str,
    /// Approximate RAM (GiB) to hold this model loaded.
    pub needs_gb: f64,
    /// Eligible for automatic selection. Models flagged `false` (e.g. the tiny
    /// 3B floor) are only ever chosen when the user explicitly opts in via the
    /// picker — never auto-picked — because their answer quality is marginal.
    pub auto: bool,
    /// Non-fatal caveat shown next to the model in the picker.
    pub warn: Option<&'static str>,
}

/// Known chat models, most-capable first. `auto_pick` walks this order and takes
/// the first `auto` model that fits the RAM budget and is installed.
///
/// Every `auto` model here passes the recency scaffold (`test_ask <m> "do I have
/// any new online assessments"` — must refuse to call a 2-year-old email
/// "recent"). `needs_gb` is loaded size plus KV/context headroom, tuned so each
/// model lands on its intended tier: 35b only on 48GB+, 20b from 32GB, 8b from
/// 16GB. 8GB stays search-only (granite is opt-in, never auto).
pub const RECOMMENDED: &[Recommendation] = &[
    Recommendation {
        // 48GB+ tier. 35B-MoE, ~21.9GB loaded; best recency reasoning of the
        // validated set. Higher `needs_gb` keeps it off 32GB (budget 25) so a
        // 32GB box gets the lighter 20b instead.
        model: "qwen3.6:35b-mlx",
        needs_gb: 28.0,
        auto: true,
        warn: None,
    },
    Recommendation {
        // 32GB tier. ~13.8GB loaded; fast and nuanced (distinguishes a reminder
        // from a new invite). Also the 48GB fallback if the 35b isn't installed.
        model: "gpt-oss:20b",
        needs_gb: 16.0,
        auto: true,
        warn: None,
    },
    Recommendation {
        // 16GB tier (settled). ~5.2GB loaded.
        model: "qwen3:8b",
        needs_gb: 7.6,
        auto: true,
        warn: None,
    },
    Recommendation {
        // 8GB opt-in floor only — never auto-picked.
        model: "granite4.1:3b",
        needs_gb: 3.0,
        auto: false,
        warn: Some(
            "Small model — weaker at date/recency reasoning, so Ask answers may be less reliable.",
        ),
    },
];

/// RAM (GiB) available for a chat model after MailFind's own reservations.
/// Floored at 0.
pub fn model_budget_gb(total_ram_gb: f64) -> f64 {
    (total_ram_gb - RESERVED_APP_GB - RESERVED_OS_GB).max(0.0)
}

/// Outcome of matching the RAM budget against installed models.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AutoPick {
    /// Use this installed, budget-fitting model.
    Model(String),
    /// A model fits the budget but isn't installed — show `ollama pull <model>`.
    NeedsPull(String),
    /// The budget is too small for any Ask model — search only, Ask off.
    SearchOnly,
}

/// Whether `wanted` (a size-tagged tag like `qwen3:8b`) is present in the
/// installed tags. Stricter than `ollama::model_present`: that one prefix-matches
/// on the base name, which would wrongly treat `qwen3:8b` as satisfying
/// `qwen3:14b` (both start `qwen3:`) and collapse the whole size ladder. Here we
/// require an exact tag, tolerating only a quant/variant suffix (`qwen3:8b-q4…`).
pub fn is_model_installed(installed: &[String], wanted: &str) -> bool {
    installed
        .iter()
        .any(|inst| inst == wanted || inst.starts_with(&format!("{wanted}-")))
}

/// Pick the best chat model for `budget_gb` given the `installed` Ollama tags.
///
/// - Best installed-and-fitting `auto` model wins.
/// - Else, if some `auto` model fits the budget but none are installed, return
///   the best-fitting one as `NeedsPull` so the UI can prompt a pull.
/// - Else the machine is below the Ask floor: `SearchOnly`.
pub fn auto_pick(budget_gb: f64, installed: &[String]) -> AutoPick {
    let mut best_fit: Option<&Recommendation> = None;
    for rec in RECOMMENDED.iter().filter(|r| r.auto) {
        if rec.needs_gb > budget_gb {
            continue;
        }
        // First (most-capable) fitting model that's installed wins outright.
        if is_model_installed(installed, rec.model) {
            return AutoPick::Model(rec.model.to_string());
        }
        // Otherwise remember the best-fitting model to suggest pulling.
        if best_fit.is_none() {
            best_fit = Some(rec);
        }
    }
    match best_fit {
        Some(rec) => AutoPick::NeedsPull(rec.model.to_string()),
        None => AutoPick::SearchOnly,
    }
}

/// Total physical RAM in GiB, via sysinfo. Returns 0.0 if it can't be read.
pub fn total_ram_gb() -> f64 {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_subtracts_reservations() {
        assert_eq!(model_budget_gb(16.0), 9.0);
        assert_eq!(model_budget_gb(8.0), 1.0);
        assert_eq!(model_budget_gb(4.0), 0.0); // floored, never negative
    }

    #[test]
    fn eight_gb_is_search_only() {
        // budget 1.0 — below every auto model's needs.
        let installed = vec!["qwen3:8b".to_string(), "granite4.1:3b".to_string()];
        assert_eq!(auto_pick(model_budget_gb(8.0), &installed), AutoPick::SearchOnly);
    }

    #[test]
    fn sixteen_gb_picks_qwen3_8b_when_installed() {
        let installed = vec!["qwen3:8b".to_string()];
        assert_eq!(
            auto_pick(model_budget_gb(16.0), &installed),
            AutoPick::Model("qwen3:8b".to_string())
        );
    }

    #[test]
    fn sixteen_gb_without_it_installed_asks_to_pull() {
        // 35b/20b don't fit 9GB budget; 8b fits but isn't installed.
        let installed = vec!["nomic-embed-text".to_string()];
        assert_eq!(
            auto_pick(model_budget_gb(16.0), &installed),
            AutoPick::NeedsPull("qwen3:8b".to_string())
        );
    }

    #[test]
    fn thirty_two_gb_prefers_20b_falls_back_to_8b() {
        let budget = model_budget_gb(32.0); // 25GB — 20b fits, 35b (28) does not
        // Only 8b installed → falls back to it.
        assert_eq!(
            auto_pick(budget, &["qwen3:8b".to_string()]),
            AutoPick::Model("qwen3:8b".to_string())
        );
        // 20b installed → preferred over 8b; 35b can't fit this budget.
        assert_eq!(
            auto_pick(budget, &["gpt-oss:20b".to_string(), "qwen3:8b".to_string()]),
            AutoPick::Model("gpt-oss:20b".to_string())
        );
    }

    #[test]
    fn forty_eight_gb_prefers_35b() {
        let budget = model_budget_gb(48.0); // 41GB — 35b fits
        assert_eq!(
            auto_pick(
                budget,
                &["qwen3.6:35b-mlx".to_string(), "gpt-oss:20b".to_string()]
            ),
            AutoPick::Model("qwen3.6:35b-mlx".to_string())
        );
        // 35b not installed → falls back to the 20b.
        assert_eq!(
            auto_pick(budget, &["gpt-oss:20b".to_string()]),
            AutoPick::Model("gpt-oss:20b".to_string())
        );
    }

    #[test]
    fn size_ladder_is_not_collapsed_by_prefix() {
        // Regression: qwen3:8b must NOT count as qwen3:14b. On a 32GB box with
        // only 8b installed, we pick 8b (fits + installed), not claim 14b.
        assert!(!is_model_installed(&["qwen3:8b".to_string()], "qwen3:14b"));
        // A quant/variant suffix of the same size still matches.
        assert!(is_model_installed(&["qwen3:8b-q4_K_M".to_string()], "qwen3:8b"));
    }

    #[test]
    fn granite_is_never_auto_picked() {
        // Even with only granite installed on an 8GB box, we stay search-only
        // rather than auto-selecting the warned tiny model.
        let installed = vec!["granite4.1:3b".to_string()];
        assert_eq!(auto_pick(model_budget_gb(8.0), &installed), AutoPick::SearchOnly);
    }
}
