# Multi-User Database Design

## Overview

The MailFind database is **designed from the ground up for multiple users**. Let me show you exactly how it stays organized.

---

## Database Structure with Multiple Users

### Current State (3 Users Example)

```
┌─────────────────────────────────────────────────────────────┐
│                         USERS TABLE                          │
├──────────┬──────────────────┬─────────────────────────────────┤
│ id (PK)  │ google_user_id   │ email                          │
├──────────┼──────────────────┼─────────────────────────────────┤
│ uuid-001 │ google-alice-123 │ alice@gmail.com                │
│ uuid-002 │ google-bob-456   │ bob@gmail.com                  │
│ uuid-003 │ google-carol-789 │ carol@gmail.com                │
└──────────┴──────────────────┴─────────────────────────────────┘
                    │
                    │ Each user has their own data
                    ▼
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    Alice's      Bob's      Carol's
     Data         Data        Data
```

---

## How Data is Partitioned Per User

### 1. Emails Table (Many:1 with users)

```sql
┌─────────────────────────────────────────────────────────────────┐
│                         EMAILS TABLE                             │
├─────────┬───────────────┬──────────────┬──────────┬─────────────┤
│ id (PK) │ user_id (FK)  │ google_user_ │ subject  │ sender      │
│         │ → users.id    │ id           │          │             │
├─────────┼───────────────┼──────────────┼──────────┼─────────────┤
│ e-001   │ uuid-001      │ google-alice │ "Meeting"│ boss@co.com │ ← Alice's
│ e-002   │ uuid-001      │ google-alice │ "Invoice"│ acct@co.com │ ← Alice's
│ e-003   │ uuid-002      │ google-bob   │ "Hello"  │ friend@...  │ ← Bob's
│ e-004   │ uuid-002      │ google-bob   │ "News"   │ news@...    │ ← Bob's
│ e-005   │ uuid-003      │ google-carol │ "Update" │ team@...    │ ← Carol's
└─────────┴───────────────┴──────────────┴──────────┴─────────────┘
```

**Key Points**:
- ✅ Each email linked to ONE user via `user_id` foreign key
- ✅ Users can only see their own emails (enforced by queries)
- ✅ If Alice is deleted → e-001 and e-002 CASCADE deleted
- ✅ Bob's and Carol's emails unaffected

**Query to get Alice's emails**:
```sql
SELECT * FROM emails
WHERE user_id = 'uuid-001'  -- Alice's UUID
-- OR
WHERE google_user_id = 'google-alice-123';  -- Also works
```

---

### 2. Search Queries Table (Many:1 with users)

```sql
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH_QUERIES TABLE                          │
├─────────┬───────────────┬──────────────┬──────────┬─────────────┤
│ id (PK) │ user_id (FK)  │ query_text   │ query_   │ threshold_  │
│         │ → users.id    │              │ type     │ used        │
├─────────┼───────────────┼──────────────┼──────────┼─────────────┤
│ q-001   │ uuid-001      │ "fantasy fb" │ sports   │ 0.28        │ ← Alice
│ q-002   │ uuid-001      │ "recent nyt" │ news     │ 0.23        │ ← Alice
│ q-003   │ uuid-002      │ "work emails"│ default  │ 0.30        │ ← Bob
│ q-004   │ uuid-003      │ "invoices"   │ default  │ 0.30        │ ← Carol
│ q-005   │ uuid-002      │ "fantasy fb" │ sports   │ 0.28        │ ← Bob
└─────────┴───────────────┴──────────────┴──────────┴─────────────┘
```

**Key Points**:
- ✅ Each search query belongs to ONE user
- ✅ Alice can't see Bob's search history
- ✅ Same query text ("fantasy fb") creates separate records for Alice and Bob
- ✅ Analytics can aggregate across all users or per-user

**Query for Alice's search history**:
```sql
SELECT query_text, query_type, created_at
FROM search_queries
WHERE user_id = 'uuid-001'  -- Alice only
ORDER BY created_at DESC
LIMIT 10;
```

---

### 3. User Search Preferences (1:1 with users)

```sql
┌─────────────────────────────────────────────────────────────────┐
│                USER_SEARCH_PREFERENCES TABLE                     │
├─────────┬───────────────┬──────────────┬──────────┬─────────────┤
│ id (PK) │ user_id (FK)  │ precision_   │ total_   │ avg_ctr     │
│         │ → users.id    │ level        │ searches │             │
│         │ UNIQUE!       │              │          │             │
├─────────┼───────────────┼──────────────┼──────────┼─────────────┤
│ p-001   │ uuid-001      │ balanced     │ 42       │ 0.35        │ ← Alice
│ p-002   │ uuid-002      │ strict       │ 18       │ 0.42        │ ← Bob
│ p-003   │ uuid-003      │ broad        │ 67       │ 0.28        │ ← Carol
└─────────┴───────────────┴──────────────┴──────────┴─────────────┘
```

**Key Points**:
- ✅ ONE preferences record per user (enforced by UNIQUE constraint)
- ✅ Alice prefers "balanced" (default)
- ✅ Bob prefers "strict" (fewer, high-quality results)
- ✅ Carol prefers "broad" (more results, lower threshold)
- ✅ Each user has independent CTR (click-through rate)

**Query for Bob's preferences**:
```sql
SELECT precision_level, avg_ctr, total_searches
FROM user_search_preferences
WHERE user_id = 'uuid-002';  -- Bob only
```

---

### 4. Sender Affinity (Many:1 with users)

```sql
┌─────────────────────────────────────────────────────────────────┐
│                    SENDER_AFFINITY TABLE                         │
├─────────┬───────────────┬──────────────┬──────────┬─────────────┤
│ id (PK) │ user_id (FK)  │ sender_email │ clicked_ │ affinity_   │
│         │ → users.id    │              │ count    │ score       │
├─────────┼───────────────┼──────────────┼──────────┼─────────────┤
│ a-001   │ uuid-001      │ boss@co.com  │ 15       │ 0.85        │ ← Alice
│ a-002   │ uuid-001      │ spam@ads.com │ 0        │ 0.10        │ ← Alice
│ a-003   │ uuid-002      │ boss@co.com  │ 2        │ 0.45        │ ← Bob
│ a-004   │ uuid-002      │ friend@gm... │ 20       │ 0.95        │ ← Bob
│ a-005   │ uuid-003      │ team@co.com  │ 8        │ 0.70        │ ← Carol
└─────────┴───────────────┴──────────────┴──────────┴─────────────┘
                                         ▲
                                         │
                          UNIQUE(user_id, sender_email)
```

**Key Points**:
- ✅ **Same sender, different affinity per user**
- ✅ Alice clicks boss@co.com often → 0.85 affinity
- ✅ Bob rarely clicks boss@co.com → 0.45 affinity
- ✅ Each user has their own engagement pattern with each sender
- ✅ UNIQUE constraint prevents duplicate (user, sender) pairs

**Insight**: Alice and Bob both receive emails from "boss@co.com", but:
- Alice engages frequently → High affinity → Boss emails ranked higher for Alice
- Bob ignores them → Low affinity → Boss emails ranked lower for Bob

---

### 5. Search Feedback (Many:1 with users)

```sql
┌─────────────────────────────────────────────────────────────────┐
│                    SEARCH_FEEDBACK TABLE                         │
├─────────┬───────────────┬──────────────┬──────────┬─────────────┤
│ id (PK) │ user_id (FK)  │ search_      │ action   │ thread_id   │
│         │ → users.id    │ query_id (FK)│          │             │
├─────────┼───────────────┼──────────────┼──────────┼─────────────┤
│ f-001   │ uuid-001      │ q-001        │ clicked  │ thread-abc  │ ← Alice
│ f-002   │ uuid-001      │ q-001        │ relevant │ thread-def  │ ← Alice
│ f-003   │ uuid-002      │ q-003        │ clicked  │ thread-xyz  │ ← Bob
│ f-004   │ uuid-002      │ q-005        │ skipped  │ thread-123  │ ← Bob
│ f-005   │ uuid-003      │ q-004        │ clicked  │ thread-456  │ ← Carol
└─────────┴───────────────┴──────────────┴──────────┴─────────────┘
```

**Key Points**:
- ✅ Each feedback action belongs to ONE user
- ✅ Linked to specific search query via `search_query_id`
- ✅ Users can't see each other's feedback
- ✅ Enables per-user learning and personalization

---

## Global Tables (Shared Across All Users)

### Query Type Quantiles

```sql
┌─────────────────────────────────────────────────────────────────┐
│                  QUERY_TYPE_QUANTILES TABLE                      │
│                   (NO user_id - GLOBAL)                          │
├──────────┬──────────────┬──────────────┬──────────┬─────────────┤
│ id (PK)  │ query_type   │ percentile_80│ sample_  │ last_       │
│          │              │              │ count    │ updated     │
├──────────┼──────────────┼──────────────┼──────────┼─────────────┤
│ qt-001   │ sports       │ 0.28         │ 450      │ 2025-10-10  │
│ qt-002   │ news         │ 0.23         │ 320      │ 2025-10-10  │
│ qt-003   │ temporal     │ 0.18         │ 680      │ 2025-10-10  │
│ qt-004   │ default      │ 0.30         │ 1200     │ 2025-10-10  │
└──────────┴──────────────┴──────────────┴──────────┴─────────────┘
```

**Why Global?**:
- ✅ **Aggregate learning**: All users contribute to improving thresholds
- ✅ When Alice searches "sports", threshold is informed by Bob's and Carol's searches too
- ✅ Sample count grows faster → Better statistical confidence
- ✅ New users benefit from existing users' data

**Example Flow**:
1. Alice searches "fantasy football" (sports query)
2. System uses global 80th percentile for sports: **0.28**
3. Alice's similarity scores: [0.85, 0.78, 0.65, 0.42, 0.31]
4. System recalculates: new p80 = 0.1 * 0.75 + 0.9 * 0.28 = **0.327**
5. Bob searches "basketball" (also sports)
6. Bob now uses updated threshold: **0.327** (benefits from Alice's search!)

---

## Real-World Multi-User Scenario

### Scenario: 3 Users, 1 Week of Activity

**Monday**:
```
Alice syncs 500 emails → emails table has 500 rows (user_id = uuid-001)
Bob syncs 300 emails   → emails table has 800 rows total
Carol syncs 700 emails → emails table has 1500 rows total
```

**Tuesday**:
```
Alice searches "fantasy football" (sports)
  → search_queries: 1 row (user_id = uuid-001)
  → Uses global threshold: 0.28 (sports)
  → Finds 12 emails from her 500 emails
  → Clicks 3 emails
  → search_feedback: 3 rows (user_id = uuid-001)
  → sender_affinity: Updates for "espn@fantasy.com" (Alice)

Bob searches "fantasy football" (sports)
  → search_queries: 1 row (user_id = uuid-002)
  → Uses global threshold: 0.28 (sports, slightly updated from Alice)
  → Finds 8 emails from his 300 emails
  → Clicks 1 email
  → search_feedback: 1 row (user_id = uuid-002)
  → sender_affinity: Updates for "yahoo@fantasy.com" (Bob)
```

**Data Isolation**:
- ✅ Alice's 12 results ≠ Bob's 8 results (different email collections)
- ✅ Alice's clicks don't appear in Bob's feedback
- ✅ Alice's sender affinity for ESPN ≠ Bob's sender affinity for Yahoo
- ✅ But both benefit from improved global sports threshold

**Wednesday**:
```
Carol searches "invoices" (default)
  → Uses global threshold: 0.30 (default)
  → Finds 25 invoices from her 700 emails
  → Has preference: precision_level = 'broad'
  → Custom offset: -0.05
  → Actual threshold used: 0.30 - 0.05 = 0.25
  → Gets MORE results due to her preference
```

**Result**:
- ✅ Carol gets personalized results (lower threshold)
- ✅ Doesn't affect Alice's or Bob's searches
- ✅ Carol's preference stored separately: `user_search_preferences` (user_id = uuid-003)

---

## Database Organization Guarantees

### 1. Data Isolation (Privacy)

```sql
-- Alice can ONLY see her own emails
SELECT * FROM emails WHERE user_id = 'uuid-001';
-- Returns: Alice's 500 emails only

-- Alice can ONLY see her own search history
SELECT * FROM search_queries WHERE user_id = 'uuid-001';
-- Returns: Alice's searches only

-- Alice can ONLY see her own feedback
SELECT * FROM search_feedback WHERE user_id = 'uuid-001';
-- Returns: Alice's clicks/ratings only
```

**Enforcement**:
- ✅ Application layer: Backend filters by `user_id` from authenticated session
- ✅ Database layer: Foreign keys ensure referential integrity
- ✅ Row-Level Security (optional): Supabase RLS can enforce at DB level

---

### 2. Data Integrity (Foreign Keys)

```sql
-- If Alice is deleted:
DELETE FROM users WHERE id = 'uuid-001';

-- CASCADE DELETE automatically removes:
├─ emails (Alice's 500 emails)
├─ search_queries (Alice's search history)
├─ search_feedback (Alice's clicks/ratings)
├─ user_search_preferences (Alice's preferences)
└─ sender_affinity (Alice's sender scores)

-- Bob's and Carol's data remains intact!
```

**Benefits**:
- ✅ No orphaned data
- ✅ No manual cleanup required
- ✅ Other users unaffected

---

### 3. Performance with Many Users

**Indexes ensure fast queries**:

```sql
-- Alice searches for "fantasy football"
-- Query:
SELECT * FROM emails
WHERE user_id = 'uuid-001'  -- Uses idx_emails_user_id
  AND similarity >= 0.28
ORDER BY similarity DESC
LIMIT 20;

-- Even with 1,000,000 emails from 1000 users:
-- Index on user_id → Narrows to Alice's 500 emails instantly
-- Then applies similarity filter on 500 rows (fast!)
```

**Scalability**:
- ✅ O(log n) lookup by `user_id` via B-tree index
- ✅ Each user's data partition is small (500-5000 emails typically)
- ✅ Queries don't scan all users' data

---

## Multi-User Query Examples

### 1. Get All Users and Their Email Counts

```sql
SELECT
    u.email AS user_email,
    COUNT(e.id) AS total_emails,
    COUNT(CASE WHEN e.created_at > NOW() - INTERVAL '7 days' THEN 1 END) AS emails_this_week
FROM users u
LEFT JOIN emails e ON e.user_id = u.id
GROUP BY u.id, u.email
ORDER BY total_emails DESC;
```

**Result**:
```
user_email          | total_emails | emails_this_week
--------------------|--------------|------------------
alice@gmail.com     | 500          | 42
carol@gmail.com     | 700          | 68
bob@gmail.com       | 300          | 15
```

---

### 2. Compare Search Behavior Across Users

```sql
SELECT
    u.email AS user_email,
    usp.precision_level,
    usp.total_searches,
    usp.avg_ctr,
    COUNT(DISTINCT sq.query_type) AS unique_query_types
FROM users u
JOIN user_search_preferences usp ON usp.user_id = u.id
LEFT JOIN search_queries sq ON sq.user_id = u.id
GROUP BY u.id, u.email, usp.precision_level, usp.total_searches, usp.avg_ctr
ORDER BY usp.avg_ctr DESC;
```

**Result**:
```
user_email      | precision | searches | avg_ctr | unique_types
----------------|-----------|----------|---------|-------------
bob@gmail.com   | strict    | 18       | 0.42    | 3
alice@gmail.com | balanced  | 42       | 0.35    | 4
carol@gmail.com | broad     | 67       | 0.28    | 5
```

**Insights**:
- Bob: Strict precision, fewer searches, high CTR (clicks relevant results)
- Carol: Broad precision, many searches, lower CTR (browses more)

---

### 3. Top Senders Across All Users

```sql
SELECT
    sa.sender_email,
    COUNT(DISTINCT sa.user_id) AS users_who_engage,
    AVG(sa.affinity_score) AS avg_affinity,
    SUM(sa.clicked_count) AS total_clicks
FROM sender_affinity sa
GROUP BY sa.sender_email
HAVING COUNT(DISTINCT sa.user_id) >= 2  -- At least 2 users
ORDER BY avg_affinity DESC
LIMIT 10;
```

**Result**:
```
sender_email        | users_who_engage | avg_affinity | total_clicks
--------------------|------------------|--------------|-------------
team@company.com    | 3                | 0.75         | 35
boss@company.com    | 2                | 0.65         | 17
news@nytimes.com    | 3                | 0.60         | 28
```

**Insights**:
- "team@company.com": High engagement across all 3 users
- Different users have different affinities for same sender

---

## Edge Cases and Solutions

### Edge Case 1: Same Email Content, Different Users

**Scenario**: Alice and Bob both subscribe to NYT newsletter, receive same email.

**Storage**:
```sql
emails table:
├─ id: e-001, user_id: uuid-001 (Alice), subject: "NYT Daily Brief", sender: "nyt@..."
└─ id: e-002, user_id: uuid-002 (Bob),   subject: "NYT Daily Brief", sender: "nyt@..."
```

**Why Separate Rows?**
- ✅ Alice might delete her copy, Bob keeps his
- ✅ Alice's embedding might differ (personalized via her encryption key)
- ✅ Simpler query isolation (no shared rows to protect)

**Storage Impact**: Minimal - text is compressed, embeddings are main cost

---

### Edge Case 2: User Switches Devices/Browsers

**Scenario**: Alice uses MailFind on laptop, then on phone.

**How It Works**:
- ✅ Both devices authenticate via Google OAuth
- ✅ Same `google_user_id` retrieved
- ✅ Queries use `WHERE google_user_id = 'google-alice-123'`
- ✅ Alice sees same emails, preferences, history on both devices

**Encryption Consideration**:
- User's encryption key stored in browser local storage
- Must unlock with password on each device
- After unlock, same `google_user_id` → same data

---

### Edge Case 3: 10,000 Users

**Scalability**:

```
Users: 10,000
Avg emails per user: 2,000
Total emails: 20,000,000

Indexes ensure:
├─ Query for Alice's emails: SELECT WHERE user_id = 'uuid-001'
│  → Index seek: O(log 20M) = ~24 comparisons
│  → Fetch 2000 rows (Alice's partition)
│  → Total time: <50ms
│
└─ Search queries, feedback, affinity all indexed by user_id
   → Same O(log n) performance per user
```

**Database stays organized because**:
- ✅ Every query filters by `user_id` first (indexed)
- ✅ Each user's partition is small (2000 rows << 20M total)
- ✅ No cross-user data scans

---

## Summary: Why It Stays Organized

### 1. Clear Ownership
```
Every data row belongs to exactly ONE user
(except global tables like query_type_quantiles)
```

### 2. Foreign Key Relationships
```
user_id → users.id (enforced by database)
Prevents orphaned data, ensures referential integrity
```

### 3. Indexed Partitioning
```
All queries filter by user_id first
Indexes make this O(log n) even with millions of users
```

### 4. Cascade Cleanup
```
DELETE user → All their data deleted automatically
Other users' data unaffected
```

### 5. Logical Grouping
```
users (root)
├── emails (user's inbox)
├── search_queries (user's search history)
├── search_feedback (user's clicks)
├── user_search_preferences (user's settings)
└── sender_affinity (user's engagement patterns)
```

---

## Visual: 3 Users, Full Data Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-USER DATABASE                           │
└─────────────────────────────────────────────────────────────────┘

┌──────────┐  ┌──────────┐  ┌──────────┐
│  Alice   │  │   Bob    │  │  Carol   │
│ uuid-001 │  │ uuid-002 │  │ uuid-003 │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │
     ├─────────────┼─────────────┼──────────────────┐
     │             │             │                  │
     ▼             ▼             ▼                  ▼
┌─────────┐  ┌─────────┐  ┌─────────┐       ┌──────────────┐
│ emails  │  │ emails  │  │ emails  │       │ query_type_  │
│ (500)   │  │ (300)   │  │ (700)   │       │ quantiles    │
└─────────┘  └─────────┘  └─────────┘       │              │
     │             │             │           │ (GLOBAL)     │
     ▼             ▼             ▼           │ All users    │
┌─────────┐  ┌─────────┐  ┌─────────┐       │ contribute   │
│ search_ │  │ search_ │  │ search_ │       └──────────────┘
│ queries │  │ queries │  │ queries │
│ (42)    │  │ (18)    │  │ (67)    │
└─────────┘  └─────────┘  └─────────┘
     │             │             │
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ search_ │  │ search_ │  │ search_ │
│ feedback│  │ feedback│  │ feedback│
└─────────┘  └─────────┘  └─────────┘
     │             │             │
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ sender_ │  │ sender_ │  │ sender_ │
│ affinity│  │ affinity│  │ affinity│
└─────────┘  └─────────┘  └─────────┘
     │             │             │
     ▼             ▼             ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│ user_   │  │ user_   │  │ user_   │
│ prefs   │  │ prefs   │  │ prefs   │
│ (1:1)   │  │ (1:1)   │  │ (1:1)   │
└─────────┘  └─────────┘  └─────────┘

DATA ISOLATION: Alice can't see Bob's or Carol's data
GLOBAL LEARNING: All users improve shared quantile thresholds
PERFORMANCE: Each user queries only their partition (indexed)
```

---

## Conclusion

✅ **Yes, the database stays extremely organized with multiple users!**

**Why?**
1. **Clear ownership**: Every row belongs to one user (via `user_id`)
2. **Foreign keys**: Database enforces relationships
3. **Indexes**: Fast queries even with millions of rows
4. **Cascade deletes**: Automatic cleanup
5. **Logical partitioning**: Each user has their own data slice

**Benefits for 1000+ users**:
- Each user's queries remain fast (O(log n) via indexes)
- Data isolation ensures privacy
- Global tables enable collective learning
- No cross-user data contamination
- Automatic cleanup when users leave

**This is a production-grade multi-tenant design!** 🎉
