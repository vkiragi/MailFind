# MailFind — Architecture

This document describes the system at two levels: the world MailFind lives in (System Context) and the major running pieces (Container view). Code-level structure is left to the codebase itself.

## Level 1 — System Context

MailFind is a single-user desktop application. Everything happens on the user's machine. The only outbound network traffic is to email servers (IMAP) for fetching mail.

```mermaid
C4Context
    title MailFind System Context

    Person(user, "User", "Has multiple email accounts, wants to search and ask questions across all of them")

    System(mailfind, "MailFind", "Local desktop app: syncs, indexes, searches, answers")

    System_Ext(icloud, "iCloud Mail", "IMAP server")
    System_Ext(fastmail, "Fastmail", "IMAP server")
    System_Ext(proton, "Proton Mail Bridge", "Local IMAP proxy to Proton")
    System_Ext(yahoo, "Yahoo Mail", "IMAP server")
    System_Ext(generic, "Generic IMAP", "Self-hosted, Zoho, university, etc.")
    System_Ext(applemail, "Apple Mail", "Source of .mbox files for initial import")

    Rel(user, mailfind, "Searches, asks questions")
    Rel(mailfind, icloud, "IMAP sync (read-only)")
    Rel(mailfind, fastmail, "IMAP sync (read-only)")
    Rel(mailfind, proton, "IMAP sync (read-only)")
    Rel(mailfind, yahoo, "IMAP sync (read-only)")
    Rel(mailfind, generic, "IMAP sync (read-only)")
    Rel(mailfind, applemail, "Reads .mbox files (one-time import)")
```

**Key properties:**

- No backend service. No MailFind servers exist.
- No outbound traffic except IMAP fetches and (opt-in) crash reports.
- All credentials in OS keychain.
- All data in local SQLite + local model files.

## Level 2 — Container view

The app is one Tauri binary, but internally has clearly separated processes/threads.

```mermaid
C4Container
    title MailFind Container View

    Person(user, "User")

    System_Boundary(mailfind, "MailFind (single Tauri binary)") {
        Container(ui, "UI", "React + Tauri webview", "Search bar, results, settings, onboarding")
        Container(core, "Core", "Rust", "Orchestrates sync, search, RAG. Tauri commands.")
        Container(sync, "Sync Engine", "Rust async (tokio)", "Per-account IMAP connections, IDLE, incremental fetch")
        Container(store, "Storage", "SQLite + FTS5 + sqlite-vec", "Messages, threads, embeddings, sync state")
        Container(search, "Search Engine", "Rust", "Hybrid keyword + vector retrieval, ranking")
        Container(rag, "RAG", "Rust", "Question handling, context assembly, citation parsing")
        Container(llm, "LLM Runtime", "llama.cpp via sidecar", "Local inference for embeddings + chat")
        Container(keychain, "Keychain Adapter", "Rust", "Per-OS credential storage")
    }

    System_Ext(imap, "IMAP Servers", "iCloud, Fastmail, etc.")
    System_Ext(os_keychain, "OS Keychain", "macOS Keychain / Win Cred Mgr / Secret Service")
    System_Ext(applemail, "Apple Mail .mbox", "Filesystem")

    Rel(user, ui, "Hotkey, click, type")
    Rel(ui, core, "invoke()")
    Rel(core, sync, "Trigger sync, get status")
    Rel(core, search, "Query")
    Rel(core, rag, "Ask question")
    Rel(sync, imap, "IMAP fetch + IDLE")
    Rel(sync, store, "Write messages")
    Rel(sync, applemail, "Read .mbox on first launch")
    Rel(search, store, "FTS5 + vector queries")
    Rel(rag, search, "Retrieve top-K chunks")
    Rel(rag, llm, "Generate answer")
    Rel(sync, llm, "Generate embeddings for new mail")
    Rel(keychain, os_keychain, "Get/set credentials")
    Rel(sync, keychain, "Read account passwords")
```

## Component responsibilities

### UI (React + Tauri webview)

- Renders the global hotkey overlay (search bar + results + answer view)
- Renders settings, onboarding, account management
- Renders menu bar icon and tray
- Calls Rust via Tauri `invoke()` — never touches data or network directly
- Streams answer tokens from RAG via Tauri events

### Core (Rust)

- Tauri command handlers that bridge UI → backend
- Owns app lifecycle: startup, shutdown, hotkey registration
- Coordinates the sync engine, search engine, RAG
- Holds the application config (accounts, preferences, license)

### Sync Engine (Rust async)

- One IMAP connection per account, kept alive with IDLE for live updates
- Tiered sync strategy:
  1. **Tier 1 (minutes):** metadata only (subject/from/to/date) for entire archive
  2. **Tier 2 (hour):** full body + FTS5 index for last 90 days
  3. **Tier 3 (background, days):** full body for everything older
  4. **Tier 4 (background, days):** embeddings, last 90 days first
- Resumable across restarts (UID + UIDVALIDITY tracked per folder)
- Throttle awareness per provider
- Detects junk/marketing mail via `List-Unsubscribe` header + sender frequency, deprioritizes embedding

### Storage (SQLite + extensions)

- Single SQLite database file in app data directory
- Tables: `accounts`, `folders`, `messages`, `threads`, `chunks`, `embeddings`, `sync_state`, `license`
- FTS5 virtual table for keyword search over message body + subject
- `sqlite-vec` extension for vector storage and KNN search over chunk embeddings
- WAL mode for concurrent read during sync writes

### Search Engine (Rust)

- Hybrid retrieval: FTS5 (BM25) + vector cosine, fused with reciprocal rank fusion
- Pre-filters by account, sender, date range when query implies them
- Recency boost (newer = higher rank, configurable decay)
- Returns ranked message chunks with stable IDs for citation

### RAG (Rust)

- Decides: keyword query vs question (presence of `?`, `what/when/who/why/how`, length heuristic)
- For questions: retrieve top-K chunks → assemble prompt with citations format → stream LLM tokens → parse `[cite:msg_id]` tags into clickable links
- For aggregation queries ("summarize all from X"): different code path — filter all matches, summarize in batches, combine
- Always returns sources alongside answer

### LLM Runtime

- Bundled `llama.cpp` binary as Tauri sidecar
- Loads `nomic-embed-text` for embeddings
- Loads chat model (default: `granite4.1:3b` or equivalent ~2-3B parameter quantized model)
- HTTP-style API on local socket (no network exposure)
- Hardware detection picks Metal (Apple Silicon), CUDA (NVIDIA), or CPU
- Models downloaded on first launch (not bundled in installer to keep download small)

### Keychain Adapter

- macOS: Security framework via `security-framework` crate
- Windows (v2): Credential Manager via `windows-credentials` crate
- Linux (v2): Secret Service via `secret-service` crate
- Stores per-account IMAP passwords / OAuth tokens

## Data flow: a search query

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant Core
    participant Search
    participant Store
    participant LLM

    User->>UI: Cmd+Shift+E, types "what did Sarah say about the budget"
    UI->>Core: invoke("query", "what did Sarah...")
    Core->>Core: Detect: question (has "what")
    Core->>LLM: Embed query
    LLM-->>Core: query vector
    Core->>Search: hybrid_search(query, vector)
    Search->>Store: FTS5 search "Sarah budget"
    Search->>Store: vector KNN over chunks
    Store-->>Search: ranked candidates
    Search-->>Core: top-20 chunks with metadata
    Core->>LLM: chat(prompt + context + citation rules)
    LLM-->>Core: stream tokens
    Core-->>UI: emit "answer-token" events
    UI-->>User: Stream answer with [cite:msg_id] → clickable links
```

## Data flow: initial sync

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant Sync
    participant IMAP
    participant Store
    participant LLM

    User->>UI: Adds iCloud account, app password
    UI->>Sync: start_sync(account)
    Sync->>IMAP: LOGIN + LIST folders
    IMAP-->>Sync: folder list
    
    par Tier 1 — Metadata
        Sync->>IMAP: FETCH headers for all UIDs
        IMAP-->>Sync: headers (batched)
        Sync->>Store: INSERT message stubs
    end
    
    par Tier 2 — Recent bodies
        Sync->>IMAP: FETCH bodies for last 90 days
        IMAP-->>Sync: bodies
        Sync->>Store: UPDATE bodies + FTS5
    end
    
    par Tier 3 — Older bodies (background)
        loop until done
            Sync->>IMAP: FETCH older bodies (paged)
            Sync->>Store: UPDATE bodies + FTS5
        end
    end
    
    par Tier 4 — Embeddings (background)
        loop per chunk
            Sync->>LLM: Embed chunk
            LLM-->>Sync: vector
            Sync->>Store: INSERT into vec table
        end
    end

    Sync->>IMAP: IDLE (long-poll for new mail)
```

## Storage schema (sketch)

```mermaid
erDiagram
    accounts ||--o{ folders : has
    folders ||--o{ messages : contains
    messages ||--o{ chunks : split_into
    chunks ||--|| embeddings : has
    messages }o--o{ threads : grouped_by

    accounts {
        int id
        string provider
        string email
        string display_name
        json sync_state
    }
    folders {
        int id
        int account_id
        string name
        int uidvalidity
        int last_uid
    }
    messages {
        int id
        int folder_id
        string message_id
        string thread_id
        datetime sent_at
        string from_addr
        string to_addrs
        string subject
        text body
        text body_stripped
        bool is_junk
    }
    threads {
        string id
        string normalized_subject
        datetime last_message_at
    }
    chunks {
        int id
        int message_id
        int chunk_index
        text content
    }
    embeddings {
        int chunk_id
        blob vector
    }
```

(Note: `embeddings` is actually a `sqlite-vec` virtual table in practice; ER notation is approximate.)

## Why these choices

| Choice | Reason |
|---|---|
| **Tauri** | 10-20MB binaries vs Electron's 100MB+. Native webview. Cross-platform with one codebase. Already in use. |
| **Rust core** | Performance for sync + indexing. Memory safety for code that handles user mail. Strong async ecosystem (tokio). |
| **SQLite + FTS5 + sqlite-vec** | Single-file storage, ACID, no daemon. FTS5 is mature. sqlite-vec gives us vector search without a separate vector DB. |
| **Bundled llama.cpp sidecar** | Removes Ollama as a user dependency. Apple Silicon Metal acceleration is solid. Cross-platform builds available. |
| **Local-only architecture** | Core differentiator. No backend = no infra costs = no recurring revenue requirement = one-time pricing works. |
| **IMAP over JMAP** | Universal. Every target provider supports it. JMAP would be cleaner for Fastmail but limits provider support. |
| **No daemon, app must be running** | Simpler than a background launchd/systemd service. Menu bar icon makes "running" feel ambient. Trade-off: no sync when app fully quit. |

## Architectural decisions deferred

- Whether to expose a CLI alongside the GUI (no for v1)
- Whether to support importing from formats other than Apple Mail mbox (no for v1)
- Whether the LLM runtime is replaceable by user's own Ollama install (no for v1, maybe v2)
- Whether to add a "watch folder" for forwarded mail from unsupported providers (no, scope creep)

These will become ADRs in `docs/decisions/` if they come up.
