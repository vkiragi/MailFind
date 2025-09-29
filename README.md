# MailFind

MailFind is an MVP that lets you index and semantically search your Gmail, and generate concise summaries of threads. It consists of:

- packages/backend: FastAPI service handling Google OAuth, Gmail API access, Supabase storage, embeddings, search, and OpenAI-powered summarization.
- packages/chrome-extension: A React + Vite Chrome extension UI.

## Features

- **Gmail Integration**: Secure OAuth authentication with Google
- **Semantic Search**: AI-powered search through your emails using embeddings
- **Email Summarization**: Generate concise summaries of email threads
- **Chatbot Interface**: Ask natural language questions about your emails (e.g., "What emails did I get this week about NYT news?")
- **Time-aware Queries**: Smart detection of time-based questions (this week, last month, today, etc.)
- **Streaming Responses**: Real-time chat responses for better user experience

## Prerequisites

- Python 3.10+
- Node.js 18+ and npm
- Google Cloud project with OAuth credentials (Web application)
- Supabase project (URL + Service Role key)
- OpenAI API key
- Chrome browser (for loading the extension)

## Repository layout

- packages/backend: FastAPI app (`main.py`), `requirements.txt`
- packages/chrome-extension: React + Vite + Tailwind extension

## 1) Backend setup

1. Create a virtualenv and install dependencies:

```bash
cd packages/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` in `packages/backend/` (see required variables below). The backend will also attempt to load a repo-root `.env` if present.

Required environment variables:

- SUPABASE_PUBLIC_URL (or SUPABASE_URL): Your Supabase project URL (e.g. https://xyzcompany.supabase.co)
- SERVICE_ROLE (or SUPABASE_SERVICE_ROLE or SUPABASE_KEY): Supabase Service Role key
- ENCRYPTION_KEY: 32-byte Fernet key for encrypting tokens (generate via `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
- GOOGLE_CLIENT_ID: OAuth 2.0 Client ID from Google Cloud
- GOOGLE_CLIENT_SECRET: OAuth 2.0 Client Secret
- OPENAI_API_KEY: OpenAI API key used for summarization

Optional configuration:

- SUPABASE_ANON_KEY: If you prefer anon usage for some ops

3. Configure Google OAuth:

- In Google Cloud Console, create OAuth 2.0 credentials for a Web application.
- Set Authorized redirect URI to: `http://localhost:8000/callback`
- Add Authorized JavaScript origins if necessary: `http://localhost:8000`

4. Run the backend locally:

```bash
uvicorn main:app --reload --port 8000
```

Health check: `GET http://localhost:8000/` returns `{ "status": "ok" }`.

### Authenticate with Google

- Navigate to `http://localhost:8000/login` in your browser. Complete the consent screen.
- On success, tokens are encrypted with `ENCRYPTION_KEY` and stored in Supabase `users` table.
- Check status: `GET http://localhost:8000/auth/status`
- Clear tokens and state: `POST http://localhost:8000/logout`

### Index, search, and summarize endpoints

- POST `/sync-inbox` → Fetch recent Gmail threads and upsert into Supabase. Body: `{ "range": "24h|7d|30d", "userId": "optional-google-user-id" }`.
- POST `/search` → Semantic search. Body: `{ "query": "...", "userId": "optional" }`.
- POST `/summarize` → Summarize a thread by message ID (backend resolves thread). Body: `{ "messageId": "...", "userId": "optional" }`.
- POST `/summarize-content` → Summarize raw content. Body: `{ "content": "..." }`.
- POST `/chat` → Chat with your emails using natural language. Body: `{ "message": "What emails did I get this week about NYT news?", "userId": "optional" }`.

Note: The backend expects at least one authenticated user stored in Supabase to access Gmail API.

## 2) Supabase setup

1. Create a Supabase project and obtain:

- Project URL (e.g., https://xyzcompany.supabase.co)
- Service Role key (Settings → API)

2. Create required tables and RPC (minimal schema):

- `users` table with columns: `id uuid default uuid_generate_v4() primary key`, `google_user_id text unique`, `email text`, `encrypted_credentials text`, `created_at timestamptz`, `updated_at timestamptz`.
- `emails` table to store indexed threads and embeddings suitable for your `match_emails` RPC.
- `match_emails` RPC function that accepts `query_embedding float8[]`, `match_threshold float8`, `match_count int` and returns matched rows from `emails` ordered by similarity.

Tip: You can adapt an existing pgvector setup; the backend calls `sb.rpc('match_emails', ...)`.

## 3) Chrome extension setup

The extension is a Vite + React app intended to be loaded as an unpacked extension in Chrome.

1. Install dependencies and build:

```bash
cd packages/chrome-extension
npm install
npm run build
```

This will produce a `dist/` folder.

2. Load the extension in Chrome:

- Open Chrome → chrome://extensions
- Enable Developer mode (toggle in top-right)
- Click “Load unpacked” and select the `packages/chrome-extension/dist` directory

3. Development mode (optional):

- `npm run dev` to start Vite at `http://localhost:5173`. Some Chrome extension features require a build to run as an extension; prefer building and reloading for realistic behavior.

## 4) Running end-to-end locally

- Start backend on port 8000.
- Ensure `.env` includes Supabase, Google, OpenAI, and ENCRYPTION_KEY values.
- Load the Chrome extension build into Chrome.
- Visit `http://localhost:8000/login` to authenticate once; then use the extension or curl the endpoints to sync/search/summarize.

## Troubleshooting

- OAuth error: Verify `GOOGLE_CLIENT_ID/SECRET` and redirect URI `http://localhost:8000/callback` match in Google Cloud.
- Supabase init error: Ensure `SUPABASE_PUBLIC_URL` and `SERVICE_ROLE` are set. URL must be `https://<ref>.supabase.co`.
- ENCRYPTION_KEY missing/invalid: Generate a Fernet key as shown above.
- OpenAI errors/timeouts: Confirm `OPENAI_API_KEY` is set; transient errors can happen.
- Chrome extension not loading: Make sure you load the built `dist/` folder, not the `src/` directory.

## Notes

- The backend CORS allows `http://localhost:5173`, `http://127.0.0.1:5173`, and `https://mail.google.com`.
- The backend also attempts to load `.env` from repo root if present.
