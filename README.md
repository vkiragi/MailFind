# MailFind

A local-first desktop app that performs private semantic search and Q&A over your iCloud Mail. Your mail, embeddings, and the LLM all stay on your machine — the only network calls leaving the device are IMAP fetches to iCloud.

Built with Tauri 2 (Rust) + React. Embeddings and answers are powered by a local [Ollama](https://ollama.com) instance.

## Requirements

- macOS (uses the system Keychain for credential storage)
- Rust >= 1.77
- Node 20+
- [Ollama](https://ollama.com) running locally, with these models pulled:

  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:3b-instruct
  ```

- An iCloud app-specific password — generate one at <https://appleid.apple.com>.

## Run

```bash
cd packages/desktop
npm install
npm run tauri:dev
```

See [packages/desktop/README.md](packages/desktop/README.md) for details on syncing iCloud, trying the app with bundled `.eml` fixtures (no iCloud needed), and producing a release build.

## Repository layout

- [packages/desktop/](packages/desktop/) — the Tauri desktop app (React UI + Rust core).
  - [src/](packages/desktop/src/) — React UI; calls Rust via Tauri `invoke`.
  - [src-tauri/](packages/desktop/src-tauri/) — Rust core: IMAP client, SQLite + FTS5 store, Ollama client, hybrid search, RAG Q&A.

There is no backend service. There is nothing to deploy. The app runs entirely on your machine.
