# Smart Search Implementation Guide

## Overview

The Smart Search feature addresses "it's not fetching what I need" issues by parsing natural language queries into structured search parameters. It uses AI (OpenAI GPT-4o-mini) to understand user intent and execute appropriate search strategies.

## Architecture

```
Natural Language Query
        ↓
[QueryParser] ← Uses OpenAI API with JSON schema mode
        ↓
Structured Query: {
  intent: "fetch_emails" | "summarize" | "list_receipts",
  people: ["john@example.com"],
  topics: ["project", "update"],
  time_range: {since: "2025-10-27T...", until: "..."},
  limits: {k: 50}
}
        ↓
[SmartSearchEngine] ← Executes appropriate search strategy
        ↓
Results + Metadata
```

## Components

### 1. QueryParser (`query_parser.py`)

Converts natural language into structured queries using:

- **LLM-based extraction**: Uses OpenAI's GPT-4o-mini with JSON schema mode for structured output
- **Few-shot prompting**: 15 diverse examples covering common query patterns
- **Deterministic helpers**:
  - Date normalization (relative → absolute timestamps)
  - Name resolution against contacts database
- **Regex fallback**: Simple pattern matching when LLM unavailable
- **Model flexibility**: Easy to switch to GPT-4o or GPT-5 nano when available

#### Example Parsing

```python
from query_parser import parse_query

# Input
query = "all updates from john in the last 3 days"

# Output
{
    "intent": "fetch_emails",
    "people": ["john@company.com"],  # Normalized via contacts
    "time_range": {
        "since": "2025-10-27T10:30:00Z",
        "until": "2025-10-30T10:30:00Z"
    },
    "topics": [],
    "limits": {"k": 50}
}
```

### 2. SmartSearchEngine (`smart_search.py`)

Executes searches based on parsed queries:

- **Intent-based routing**: Different strategies for fetch/summarize/receipts
- **Filter composition**: Combines people, topics, time filters
- **Hybrid search**: Vector (semantic) + keyword + filters
- **Result post-processing**: Client-side filtering when needed

### 3. Backend Integration (`main.py`)

New endpoint: `POST /smart-search`

```json
Request:
{
  "query": "show me receipts from DoorDash this month",
  "userId": "user123"
}

Response:
{
  "results": [...],  // Array of email objects
  "parsed_query": {
    "intent": "list_receipts",
    "people": ["DoorDash"],
    "time_range": {"last_n": "1 month"},
    "topics": ["receipt", "order"],
    "limits": {"k": 100}
  },
  "search_strategy": "Filter by senders: DoorDash | Time range: last 1 month | Topics: receipt, order | Max results: 100",
  "result_count": 42
}
```

## Query Examples

### 1. Person + Time
```
"all updates from john in the last 3 days"
→ people: ["john"], time_range: {last_n: "3 days"}
```

### 2. Receipts/Transactions
```
"show me receipts from DoorDash this month"
→ intent: "list_receipts", people: ["DoorDash"], topics: ["receipt", "order"]
```

### 3. Topic-based
```
"emails about the contract negotiation"
→ topics: ["contract", "negotiation"]
```

### 4. Summarization
```
"what did alice say about the bug fix?"
→ intent: "summarize", people: ["alice"], topics: ["bug", "fix"]
```

### 5. Specific Counts
```
"show me the last 10 emails from my boss"
→ people: ["boss"], limits: {k: 10}
```

### 6. Amount Filters
```
"all invoices over $500 since January"
→ topics: ["invoice"], filters: {amount_min: 500}, time_range: {since: "2025-01-01"}
```

### 7. Relative Time
```
"meeting notes from yesterday"
→ topics: ["meeting", "notes"], time_range: {last_n: "1 day"}
```

### 8. Multiple Filters
```
"package delivery notifications this week"
→ topics: ["package", "delivery", "notification"], time_range: {last_n: "1 week"}
```

## Setup

### 1. Install Dependencies

```bash
cd packages/backend
pip install -r requirements.txt
```

Required packages:
- `openai>=1.0.0` - For query parsing (already in requirements.txt)
- `python-dateutil>=2.9.0` - For date parsing

### 2. Configure API Keys

Add to `.env`:

```bash
OPENAI_API_KEY=sk-...
```

**Note:** OpenAI API keys start with `sk-`. You can get one at https://platform.openai.com/api-keys

### 3. Test the Parser

```bash
python test_query_parser.py
```

This will:
1. Test LLM-based parsing on 12 sample queries
2. Test contact normalization with a mock database
3. Show parsed structures for each query

## Usage

### From Frontend

```typescript
const response = await fetch('http://localhost:8000/smart-search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "all updates from john in the last 3 days",
    userId: currentUserId
  })
});

const data = await response.json();
console.log(`Found ${data.result_count} emails`);
console.log(`Strategy: ${data.search_strategy}`);
console.log(`Results:`, data.results);
```

### Programmatically

```python
from smart_search import smart_search

results = smart_search(
    query="show me receipts from DoorDash this month",
    supabase_client=sb,
    embedding_model=model,
    user_id="user123"
)

print(f"Found {results['result_count']} emails")
for email in results['results']:
    print(f"- {email['subject']}")
```

## How It Solves "Not Fetching What I Need"

### Problem 1: Vague Time References

**Before:** "last week" might not parse correctly

**Now:**
```python
"emails from last week"
→ time_range: {since: "2025-10-23T00:00:00Z", until: "2025-10-30T23:59:59Z"}
```

### Problem 2: Name Ambiguity

**Before:** "john" might not match "John Smith <john@company.com>"

**Now:** Normalized against contacts database
```python
"emails from john"
→ people: ["john@company.com"]  # Resolved from contacts
```

### Problem 3: Intent Confusion

**Before:** "summarize" vs "find" treated the same

**Now:** Different intents trigger different strategies
```python
"summarize emails about project" → intent: "summarize", limits: {k: 20}
"find emails about project"     → intent: "fetch_emails", limits: {k: 50}
```

### Problem 4: Missing Context

**Before:** Results lacked transparency about what was searched

**Now:** Response includes parsed query and strategy
```json
{
  "search_strategy": "Filter by senders: john@company.com | Time range: last 3 days | Max results: 50",
  "parsed_query": {...}
}
```

## Customization

### Adding New Query Patterns

Edit `query_parser.py`, add to `FEW_SHOT_EXAMPLES`:

```python
Example 16:
Query: "urgent emails from the CEO"
Output: {"intent": "fetch_emails", "people": ["CEO"], "topics": ["urgent"], "filters": {"importance": "high"}, "limits": {"k": 20}}
```

### Custom Search Strategies

Edit `smart_search.py`, modify `_fetch_emails()`:

```python
# Add custom filter
if parsed.get("filters", {}).get("importance") == "high":
    query_builder = query_builder.eq("importance", "high")
```

### Adjusting Thresholds

In `smart_search.py`:

```python
# For more/fewer vector search results
'match_threshold': 0.35,  # Lower = more results
'match_count': 50,        # Higher = more candidates
```

## Monitoring

### Debug Output

The parser logs extensively:

```
=== SMART SEARCH REQUEST START ===
[SmartSearch] Query: 'all updates from john in the last 3 days'
[SmartSearch] Parsed query: {...}
[SmartSearch] Strategy: Filter by senders: john@company.com | Time range: 2025-10-27 to 2025-10-30
[SmartSearch] Found 15 results
=== SMART SEARCH REQUEST END ===
```

### Error Handling

- If Anthropic API fails → Falls back to regex parsing
- If regex parsing fails → Returns empty structured query
- Response always includes `parsed_query` for debugging

## Performance

### LLM Call Latency

- Typical: 300-800ms for query parsing (GPT-4o-mini)
- Cached embeddings: No additional latency
- Total: ~0.5-1.5s for end-to-end search

### Optimization Tips

1. **Cache parsed queries** (same query → same structure)
2. **Use regex fallback** for simple queries (no API call)
3. **Batch searches** if multiple queries needed

### Cost

OpenAI API costs (GPT-4o-mini):
- ~$0.0001 per query parse (10x cheaper than GPT-4o)
- Typical: 150-250 input tokens, 50-100 output tokens
- Pricing: $0.150 per 1M input tokens, $0.600 per 1M output tokens
- **~1,000 queries for $1** 💰

If you upgrade to GPT-4o or GPT-5 nano:
- GPT-4o: ~$0.001 per query (better accuracy)
- GPT-5 nano: TBD (expected to be similar to GPT-4o-mini)

## Troubleshooting

### "Smart search not available"

**Issue:** Missing dependencies

**Fix:**
```bash
pip install openai python-dateutil
```

### "OPENAI_API_KEY not found"

**Issue:** Environment variable not set

**Fix:**
```bash
export OPENAI_API_KEY=sk-...
# Or add to .env file
```

Get your API key at: https://platform.openai.com/api-keys

### "No results found" despite matching emails

**Issue:** Time range too restrictive or name not normalized

**Debug:**
```python
result = smart_search(query, ...)
print("Parsed:", result['parsed_query'])
print("Strategy:", result['search_strategy'])
```

**Fix:** Check `parsed_query.time_range` and `parsed_query.people`

### LLM parsing returns unexpected structure

**Issue:** Query pattern not in few-shot examples

**Fix:** Add similar example to `FEW_SHOT_EXAMPLES` in `query_parser.py`

## Migration from Old Search

### Option 1: Replace `/search`

```python
# In main.py, redirect old search to smart search
@app.post("/search")
async def search_emails(request: dict):
    return await smart_search_emails(request)
```

### Option 2: Parallel Rollout

Keep both endpoints, add feature flag:

```typescript
const endpoint = useSmartSearch
  ? '/smart-search'
  : '/search';
```

### Option 3: A/B Test

```python
import random

@app.post("/search")
async def search_emails(request: dict):
    if random.random() < 0.5:  # 50% traffic
        return await smart_search_emails(request)
    # ... existing search logic
```

## Future Enhancements

1. **Query expansion**: "Python" → also search "py", "python3"
2. **Spell correction**: "johh" → "john"
3. **Conversation history**: Multi-turn refinement
4. **Feedback loop**: Learn from user clicks/ratings
5. **Local caching**: Store (query → parsed structure) in DB

## Support

For issues or questions:
1. Check logs in `backend.log`
2. Run `python test_query_parser.py` to verify setup
3. Review parsed query structure in API response
4. Open issue with query + parsed structure + expected behavior
