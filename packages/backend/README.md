# MailFind Backend

FastAPI backend for MailFind MVP with AI-powered smart search.

## Features

✨ **Smart Search** (NEW) - Natural language query parsing that understands:
- People filters: "emails from john"
- Time ranges: "last 3 days", "this month", "since January"
- Topics: "about the contract", "receipts from DoorDash"
- Intent detection: fetch, summarize, or list receipts
- 15+ query patterns with AI-powered parsing

See [SMART_SEARCH_GUIDE.md](SMART_SEARCH_GUIDE.md) for details.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick Start with Smart Search

```bash
./quick_start_smart_search.sh
```

This will:
1. Check dependencies
2. Verify environment configuration
3. Test the query parser
4. Test the smart search endpoint

## Environment Variables

Required in `.env`:

```bash
# Core
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
SUPABASE_URL=...
SUPABASE_KEY=...

# Smart Search (NEW) - Uses OpenAI GPT-4o-mini
OPENAI_API_KEY=sk-...
```

## Run

### Option 1: Smart Start (Recommended)
Automatically finds an available port if 8000 is in use:

```bash
python start_server.py
```

### Option 2: Manual Start
Directly specify the port:

```bash
uvicorn main:app --reload --port 8000
```
