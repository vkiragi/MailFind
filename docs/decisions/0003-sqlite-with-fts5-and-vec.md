# ADR 0003 — SQLite with FTS5 + sqlite-vec for storage and search

**Status:** Accepted

**Context:**

We need persistent storage for messages, threads, and embeddings, plus fast keyword and vector search. Options considered:

- SQLite + FTS5 + sqlite-vec
- DuckDB + custom vector indexing
- Embedded vector DB (LanceDB, qdrant-rs, chroma)
- Multi-store: SQLite for metadata + separate vector DB

**Decision:**

Single SQLite file with FTS5 for keyword search and `sqlite-vec` extension for vector search.

**Reasoning:**

- **Single file, single backup, no daemon.** Trivial to manage, trivial to debug, trivial for users to back up themselves.
- **Mature.** SQLite is the most-deployed database in the world. FTS5 has been stable for nearly a decade.
- **sqlite-vec is the right scale for this product.** A user with 500K emails has maybe 2M chunks → 2M vectors. sqlite-vec handles this comfortably on commodity hardware.
- **No process boundary.** Everything in one transaction across messages, threads, embeddings. No sync issues between stores.
- **Cross-platform out of the box.** SQLite ships everywhere; extensions compile cleanly for Mac/Windows/Linux.

**Trade-offs accepted:**

- sqlite-vec is newer than FTS5 — small risk of API changes. Pin version, watch upstream.
- Vector search is brute-force (no HNSW index). At our scale (~millions of vectors per user), brute-force is fine. Add an index if scale demands.
- Multi-store would let us pick best-of-breed for each — but adds complexity that isn't justified at single-user scale.

**Migration plan:**

If sqlite-vec scaling becomes a problem (e.g., users with 10M+ chunks), switch to a separate vector store (LanceDB or similar) without changing the metadata DB. The schema is structured so this swap doesn't touch other tables.
