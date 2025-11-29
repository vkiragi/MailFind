# Adaptive Search System - Implementation Guide

## Overview

The MailFind Adaptive Search system replaces static cosine similarity thresholds with **quantile-based adaptive thresholds** that learn from user behavior and search patterns over time.

### Key Features

 **Implemented (Phase 1-3)**:
- Quantile-based adaptive thresholds (80th percentile by default)
- Per-query-type threshold selection (sports, news, temporal, latest, default)
- User feedback tracking (clicks, relevance ratings)
- Sender affinity scoring
- User search preferences and personalization
- Search query logging and analytics
- Real-time quantile updates with exponential moving average

=§ **Pending (Phase 4-7)**:
- Learning-to-Rank (LTR) module with feature extraction
- Frontend precision controls (strict/balanced/broad)
- Metrics aggregation and auto-tuning
- Scheduled jobs for model updates

---

## Architecture

### Database Schema

**6 New Tables** (see `migrations/adaptive_search_schema.sql`):

1. **query_type_quantiles** - Rolling quantile statistics per query type
2. **search_queries** - Historical search log with metadata
3. **search_feedback** - User interaction tracking
4. **user_search_preferences** - Per-user preferences and learned parameters
5. **sender_affinity** - Engagement metrics per sender
6. **search_metrics** - Aggregated performance metrics for auto-tuning

### Core Modules

**`adaptive_search.py`** - New module containing:
- `AdaptiveSearchEngine` class
- `perform_adaptive_search()` function (main entry point)
- Quantile calculation and storage
- User preference management

**`main.py`** - Updated chat endpoint:
- Replaced fixed thresholds with adaptive search
- Added `/search-feedback` endpoint for user interactions
- Returns search metadata with results

---

## How It Works

### 1. Query Classification

Queries are automatically classified into types:

```python
query_types = {
    'sports': ['fantasy', 'football', 'basketball', ...],
    'news': ['news', 'nyt', 'breaking', ...],
    'temporal': ['recent', 'latest', 'today', ...],
    'latest': ['most recent', 'newest', ...],
    'default': (everything else)
}
```

### 2. Adaptive Threshold Selection

Instead of fixed thresholds (0.15, 0.25, 0.3), the system:

1. **Calculates quantiles** from similarity scores (10th, 20th, 50th, 80th, 90th percentiles)
2. **Stores quantiles** per query type in database
3. **Uses 80th percentile** as threshold (configurable per user)
4. **Updates quantiles** using exponential moving average (±=0.1)

**Example**:
```
Sports query similarity scores: [0.85, 0.78, 0.65, 0.42, 0.31, 0.28, 0.15, 0.10]
80th percentile = 0.65
’ Threshold set to 0.65 (adaptive, not fixed!)
```

### 3. User Feedback Loop

When users interact with results:

```javascript
// Frontend sends feedback
fetch('/search-feedback', {
  method: 'POST',
  body: JSON.stringify({
    search_query_id: 'uuid',
    google_user_id: 'user123',
    thread_id: 'gmail-thread-456',
    action: 'clicked',  // or 'relevant', 'irrelevant', 'skipped'
    similarity_score: 0.85,
    rank_position: 1,
    dwell_time_ms: 5000
  })
})
```

Backend then:
- Records feedback in `search_feedback` table
- Updates `sender_affinity` for that sender
- Recalculates user's CTR (click-through rate)
- Adjusts future thresholds based on engagement

### 4. Sender Affinity Scoring

Tracks user engagement with specific senders:

```python
affinity_score = (clicks + 2*marked_relevant) / total_emails_from_sender
```

Higher affinity = emails from that sender ranked higher in future searches.

---

## API Reference

### Chat Endpoint (Updated)

**POST** `/chat`

Request:
```json
{
  "message": "recent fantasy football emails",
  "userId": "google-user-id-123"
}
```

Response (streaming):
```
data: {"emails": [...], "search_metadata": {"query_type": "sports", "threshold": 0.28, "search_query_id": "uuid", "results_count": 15}}
data: {"content": "You"}
data: {"content": " have"}
data: {"content": " 15"}
...
data: [DONE]
```

### Search Feedback Endpoint (New)

**POST** `/search-feedback`

Request:
```json
{
  "search_query_id": "uuid-from-search-metadata",
  "google_user_id": "user-123",
  "thread_id": "gmail-thread-id",
  "action": "clicked",
  "similarity_score": 0.85,
  "rank_position": 1,
  "dwell_time_ms": 5000,
  "metadata": {}
}
```

Response:
```json
{
  "success": true,
  "feedback_id": "uuid",
  "message": "Feedback recorded: clicked"
}
```

**Actions**:
- `clicked` - User clicked to open the email
- `relevant` - User explicitly marked as relevant
- `irrelevant` - User marked as not relevant
- `skipped` - User scrolled past without interacting
- `dismissed` - User dismissed from results

---

## Migration Instructions

### Step 1: Run Database Migration

**Option A: Supabase Dashboard (Recommended)**

1. Go to https://supabase.com/dashboard/project/YOUR_PROJECT/sql
2. Copy contents of `migrations/adaptive_search_schema.sql`
3. Paste into SQL Editor
4. Click "Run"

**Option B: Migration Script**

```bash
cd packages/backend
python run_migration.py migrations/adaptive_search_schema.sql
```

Note: This script provides instructions but doesn't execute directly (Supabase Python client doesn't support DDL).

### Step 2: Verify Tables Created

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_name IN (
  'query_type_quantiles',
  'search_queries',
  'search_feedback',
  'user_search_preferences',
  'sender_affinity',
  'search_metrics'
);
```

Should return 6 rows.

### Step 3: Check Default Quantiles

```sql
SELECT * FROM query_type_quantiles;
```

Should show 5 rows with initialized percentile values.

### Step 4: Restart Backend

```bash
cd packages/backend
python main.py
```

Look for log message:
```
[Chat] Using adaptive quantile-based search...
[AdaptiveSearch] Query classified as: sports
[AdaptiveSearch] Using adaptive threshold 0.2800 for 'sports' (p80)
```

---

## Configuration

### User Preferences

Users can customize their search behavior via `user_search_preferences`:

```sql
UPDATE user_search_preferences
SET
  precision_level = 'strict',  -- 'strict', 'balanced', 'broad'
  preferred_quantile = 0.90,   -- Use 90th percentile (fewer, higher quality)
  custom_threshold_offset = 0.05  -- Add +0.05 to all thresholds
WHERE google_user_id = 'user-123';
```

**Precision Levels** (for future frontend UI):
- **Strict**: 90th percentile, fewer high-quality results
- **Balanced**: 80th percentile (default)
- **Broad**: 50th percentile, more results, lower quality threshold

### Quantile Update Rate

Exponential moving average smoothing factor in `adaptive_search.py`:

```python
alpha = 0.1  # 10% new data, 90% historical
```

- Lower ± = slower adaptation (more stable)
- Higher ± = faster adaptation (more responsive to recent patterns)

---

## Monitoring & Analytics

### View Search Performance

```sql
-- Overall search stats
SELECT
  query_type,
  COUNT(*) as total_searches,
  AVG(results_count) as avg_results,
  AVG(threshold_used) as avg_threshold
FROM search_queries
GROUP BY query_type;
```

### User Engagement

```sql
-- User CTR and engagement
SELECT
  google_user_id,
  total_searches,
  total_clicks,
  avg_ctr,
  precision_level
FROM user_search_preferences
ORDER BY avg_ctr DESC;
```

### Sender Affinity

```sql
-- Top senders by affinity for a user
SELECT
  sender_email,
  affinity_score,
  clicked_count,
  marked_relevant,
  total_emails
FROM sender_affinity
WHERE google_user_id = 'user-123'
ORDER BY affinity_score DESC
LIMIT 10;
```

### Feedback Distribution

```sql
-- What actions do users take?
SELECT
  action,
  COUNT(*) as count,
  AVG(similarity_score) as avg_similarity
FROM search_feedback
GROUP BY action;
```

---

## Adaptive Threshold Example

### Before (Fixed Thresholds)

```python
# main.py (old)
if is_sports_query:
    threshold = 0.25  # STATIC
elif is_news_query:
    threshold = 0.20  # STATIC
else:
    threshold = 0.30  # STATIC
```

**Problem**: These values were guessed and never change based on actual data.

### After (Adaptive Quantiles)

```python
# main.py (new)
search_results, metadata = perform_adaptive_search(
    supabase_client=sb,
    query_embedding=query_embedding,
    query_text=message,
    google_user_id=user_id,
    preferred_percentile=80  # Use 80th percentile
)
```

**Benefits**:
- Threshold adapts to actual similarity score distribution
- Different thresholds for different query types
- Learns from user feedback over time
- Per-user customization

---

## Data Flow

```
User Query: "recent fantasy football emails"
    “
1. Query Classification
   ’ query_type: 'sports'
    “
2. Get Adaptive Threshold
   ’ Fetch quantiles from DB for 'sports'
   ’ Use 80th percentile: 0.28
    “
3. Semantic Search
   ’ cosine_similarity >= 0.28
   ’ Returns 15 emails with similarity scores
    “
4. Log Search Query
   ’ Insert into search_queries table
   ’ search_query_id: 'uuid-abc-123'
    “
5. Update Quantiles
   ’ Calculate new quantiles from scores
   ’ Update with EMA (±=0.1)
    “
6. Return Results + Metadata
   ’ emails: [...]
   ’ search_metadata: {query_type, threshold, search_query_id}
    “
7. User Clicks Email #3
   ’ Frontend sends /search-feedback
   ’ action: 'clicked', rank_position: 3
    “
8. Update Affinity & Preferences
   ’ sender_affinity.clicked_count++
   ’ user_search_preferences.total_clicks++
   ’ Recalculate affinity_score and avg_ctr
```

---

## Next Steps (Pending Implementation)

### Phase 4: Learning-to-Rank (LTR)

Create `learning_to_rank.py` module:

**Features to extract**:
- Sender affinity score
- Historical CTR for this sender
- Time decay (recency boost)
- Email importance score
- Has attachments
- Thread depth
- Subject keyword match

**Ranking model**:
- Train on user feedback data
- Use gradient boosting (XGBoost/LightGBM)
- Predict probability of click/relevance
- Re-rank results by predicted score

### Phase 5: Frontend Precision Controls

Add to Settings tab:

```jsx
<select value={precisionLevel} onChange={handlePrecisionChange}>
  <option value="strict">Strict (fewer, higher quality)</option>
  <option value="balanced">Balanced (default)</option>
  <option value="broad">Broad (more results)</option>
</select>
```

### Phase 6: Metrics Aggregation

Scheduled job (daily):

```python
# Aggregate daily metrics
INSERT INTO search_metrics (query_type, date, total_searches, avg_ctr, ...)
SELECT
  query_type,
  CURRENT_DATE,
  COUNT(*) as total_searches,
  ...
FROM search_queries
WHERE created_at >= CURRENT_DATE
GROUP BY query_type;
```

### Phase 7: Auto-Tuning

Detect when thresholds need adjustment:

```python
if avg_ctr < 0.10:  # Low CTR indicates threshold too high
    decrease_threshold(query_type, offset=-0.05)
elif avg_results_count < 5:  # Too few results
    decrease_threshold(query_type, offset=-0.03)
```

---

## Testing

### Manual Testing

1. **Run a search**:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "recent fantasy football emails"}'
   ```

2. **Check logs** for:
   ```
   [AdaptiveSearch] Query classified as: sports
   [AdaptiveSearch] Using adaptive threshold 0.2800 for 'sports' (p80)
   [AdaptiveSearch] Found 15 results with threshold 0.2800
   [AdaptiveSearch] Logged search query: 'recent fantasy...' (id: uuid)
   [AdaptiveSearch] Updated quantiles for 'sports' (total samples: 150)
   ```

3. **Send feedback**:
   ```bash
   curl -X POST http://localhost:8000/search-feedback \
     -H "Content-Type: application/json" \
     -d '{
       "search_query_id": "uuid-from-search",
       "google_user_id": "user-123",
       "thread_id": "gmail-thread-id",
       "action": "clicked"
     }'
   ```

4. **Verify in database**:
   ```sql
   -- Check search was logged
   SELECT * FROM search_queries ORDER BY created_at DESC LIMIT 1;

   -- Check feedback recorded
   SELECT * FROM search_feedback ORDER BY created_at DESC LIMIT 1;

   -- Check quantiles updated
   SELECT * FROM query_type_quantiles WHERE query_type = 'sports';
   ```

### Unit Testing

Create `test_adaptive_search.py`:

```python
import unittest
from adaptive_search import AdaptiveSearchEngine

class TestAdaptiveSearch(unittest.TestCase):
    def test_query_classification(self):
        engine = AdaptiveSearchEngine(mock_supabase)

        self.assertEqual(
            engine.classify_query_type("recent fantasy football emails"),
            'sports'
        )

        self.assertEqual(
            engine.classify_query_type("latest news from NYT"),
            'news'
        )

    def test_quantile_calculation(self):
        engine = AdaptiveSearchEngine(mock_supabase)
        scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        quantiles = engine.calculate_quantiles(scores)

        self.assertAlmostEqual(quantiles['percentile_50'], 0.5, places=2)
        self.assertAlmostEqual(quantiles['percentile_80'], 0.74, places=2)
```

---

## Troubleshooting

### Issue: Threshold too high, no results

**Symptom**: Search returns 0-2 results consistently

**Solution**:
```sql
-- Check current threshold
SELECT percentile_80 FROM query_type_quantiles WHERE query_type = 'sports';

-- Manually lower if needed
UPDATE query_type_quantiles
SET percentile_80 = 0.20  -- Lower from 0.28
WHERE query_type = 'sports';
```

Or adjust user preference:
```sql
UPDATE user_search_preferences
SET custom_threshold_offset = -0.05  -- Reduce threshold by 0.05
WHERE google_user_id = 'user-123';
```

### Issue: Quantiles not updating

**Check**:
1. Search queries are being logged?
   ```sql
   SELECT COUNT(*) FROM search_queries WHERE created_at > NOW() - INTERVAL '1 hour';
   ```

2. Look for errors in backend logs:
   ```
   [AdaptiveSearch] Error updating quantiles: ...
   ```

3. Verify sample_count is increasing:
   ```sql
   SELECT query_type, sample_count, last_updated
   FROM query_type_quantiles;
   ```

### Issue: Sender affinity not updating

**Possible causes**:
- Email is encrypted and sender field is NULL
- thread_id not matching
- Feedback action is 'skipped' or 'dismissed' (doesn't update affinity)

**Check**:
```sql
-- Find emails with NULL sender
SELECT COUNT(*) FROM emails WHERE sender IS NULL;

-- Check if thread_id exists
SELECT * FROM emails WHERE thread_id = 'your-thread-id';
```

---

## Performance Considerations

### Database Indexes

Already created in migration:
```sql
CREATE INDEX idx_search_queries_user ON search_queries(google_user_id);
CREATE INDEX idx_search_feedback_query ON search_feedback(search_query_id);
CREATE INDEX idx_sender_affinity_score ON sender_affinity(affinity_score DESC);
```

### Query Optimization

- Search queries are logged asynchronously (doesn't block response)
- Quantile updates use exponential moving average (O(1) update time)
- Sender affinity lookups use indexed queries

### Scaling

For high-volume usage:
1. Move quantile updates to background job (Redis queue)
2. Cache user preferences in memory
3. Batch feedback processing
4. Use materialized views for search_metrics

---

## References

- Original feature request: "Replace static thresholds with adaptive quantile-based cutoffs"
- Migration file: `migrations/adaptive_search_schema.sql`
- Module: `adaptive_search.py`
- Endpoint: `/chat` (updated), `/search-feedback` (new)

**Created**: 2025-10-10
**Status**: Phase 1-3 Complete 
**Next**: Learning-to-Rank module (Phase 4)
