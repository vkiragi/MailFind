//! Local hybrid search: keyword (FTS5) + vector (cosine) over chunk
//! embeddings, fused by reciprocal rank fusion. Mirrors the strategy in the
//! prior backend (`packages/backend/main.py::_combine_search_candidates`) but
//! everything stays on-device.

pub mod background;
pub mod chunking;
pub mod hybrid;
pub mod indexer;

pub use background::spawn_background_embedder;
pub use hybrid::{search, MessageHit, SearchOutcome};
pub use indexer::ensure_embeddings;
