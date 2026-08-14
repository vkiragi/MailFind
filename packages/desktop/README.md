# MailFind Desktop

A standalone Tauri 2 + React desktop app that performs **private, local
semantic search** over your iCloud Mail.

- iCloud Mail sync over IMAP (read-only, app-specific password).
- All mail stored locally in SQLite under your user data directory.
- Embeddings via [Ollama](https://ollama.com) using `nomic-embed-text`.
- Question answering via a small Qwen instruct model, also via Ollama.
- Tauri command surface in `src-tauri/src/commands.rs`; React calls them
  through the typed wrapper in `src/lib/api.ts`.

## Requirements

- Rust >= 1.77
- Node 20+
- [Ollama](https://ollama.com) running locally, with the embedder pulled:

  ```bash
  ollama pull nomic-embed-text
  ```

  Semantic search runs on any Mac (~1.5 GB). The **Ask** tab's chat model is
  chosen automatically to fit your RAM — search-only on 8 GB up to a 35B-class
  model on 48 GB+ — and the app shows the right `ollama pull` command if the
  pick isn't installed. See the model table in the [root README](../../README.md#choosing-an-ask-model--automatic).
  You can override the choice under Accounts → Ask model.

- macOS recommended for v1 (uses macOS Keychain for credential storage via
  the `keyring` crate; other platforms use the platform-native backend).

## Run

From the repo root:

```bash
./dev.sh
```

## Try Without iCloud (Fixtures)

You can demo search/Q&A end-to-end before configuring an iCloud account:

1. Launch the app, add an account with any IMAP host (you can keep the
   default `imap.mail.me.com`).
2. Open the **Accounts** tab.
3. Click **Import .eml file** and pick one of the fixtures from
   `packages/desktop/src-tauri/fixtures/`.
4. Switch to **Search** or **Ask** to query the imported messages.

## Configure iCloud Sync

1. Generate an app-specific password at <https://appleid.apple.com>.
2. In the app, add an account with your iCloud address and that password.
3. Click **Sync Now** to pull recent INBOX messages.
4. The app will automatically embed new messages with `nomic-embed-text`
   in batches after each sync.

## Build

For production icons, run once:

```bash
npx tauri icon path/to/your-icon-1024.png
```

Then:

```bash
npm run tauri:build
```

## Architecture

```
React UI  -->  Tauri commands (Rust)
                  |- db (SQLite, FTS5, embeddings as BLOBs)
                  |- mail (IMAP + RFC822 parser + fixture ingest)
                  |- models (Ollama HTTP client)
                  |- search (FTS5 + cosine RRF fusion)
                  |- qa (Qwen answer with email citations)
```

The data layer is a port of the prior backend's Postgres schema
(`packages/backend/migrations/mailfindv2_init.sql`) into local SQLite.
Search/Q&A behavior is a Rust port of the relevant pieces of
`packages/backend/main.py` and `packages/backend/rag/*`.
