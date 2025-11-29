# MailFind Database Schema

## Overview

The MailFind database follows a **user-centric hierarchical design** where most tables relate back to the `users` table.

## Core Tables

### 1. `users` (Primary table)
```
users
├── id (UUID, PRIMARY KEY)
├── google_user_id (VARCHAR, UNIQUE)  ← Google OAuth user ID
├── email (VARCHAR)
├── encrypted_credentials (TEXT)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)
```

**Purpose**: Store authenticated users and their OAuth credentials

**Key Fields**:
- `id` - Internal UUID primary key (used for foreign keys)
- `google_user_id` - Google's unique user identifier (used for lookups)

---

### 2. `emails` (Primary email data)
```
emails
├── id (UUID, PRIMARY KEY)
├── google_user_id (VARCHAR) → users.google_user_id
├── thread_id (VARCHAR)  ← Gmail thread ID
├── thread_id_hash (VARCHAR)  ← Hashed thread ID for encrypted mode
├── subject (TEXT, nullable if encrypted)
├── sender (VARCHAR, nullable if encrypted)
├── content (TEXT, nullable if encrypted)
├── encrypted_content (TEXT)  ← Encrypted email data
├── iv (TEXT)  ← Initialization vector for encryption
├── embedding (vector(384))  ← Semantic embedding
├── importance_score (INT)
├── is_automated (BOOLEAN)
├── has_unsubscribe (BOOLEAN)
├── categories (JSONB)
└── created_at (TIMESTAMP)
```

**Purpose**: Store all user emails (encrypted or plain text)

**Relationships**:
- Links to `users` via `google_user_id` (not a formal FK due to mixed encryption modes)

---

## User-Related Tables

### 3. `user_settings`
```
user_settings
├── id (UUID, PRIMARY KEY)
├── user_id (UUID) → users.id (FK, CASCADE DELETE)
├── auto_sync_enabled (BOOLEAN)
├── sync_frequency (VARCHAR)
├── created_at (TIMESTAMP)
└── updated_at (TIMESTAMP)

UNIQUE(user_id)  ← One settings record per user
```

**Relationship**: **1:1 with users**
- `user_id` → `users.id` (foreign key with CASCADE DELETE)
- If user is deleted, their settings are deleted

---

### 4. `user_search_preferences`
```
user_search_preferences
├── id (UUID, PRIMARY KEY)
├── user_id (UUID) → users.id (FK, CASCADE DELETE)
├── google_user_id (VARCHAR, denormalized)
├── precision_level (VARCHAR)  ← 'strict', 'balanced', 'broad'
├── custom_threshold_offset (DECIMAL)
├── preferred_quantile (DECIMAL)
├── feature_weights (JSONB)
├── total_searches (INT)
├── total_clicks (INT)
├── avg_ctr (DECIMAL)
├── last_updated (TIMESTAMP)
└── created_at (TIMESTAMP)

UNIQUE(user_id)
UNIQUE(google_user_id)
```

**Relationship**: **1:1 with users**
- `user_id` → `users.id` (foreign key with CASCADE DELETE)
- `google_user_id` denormalized for query performance

---

## Search & Feedback Tables

### 5. `search_queries`
```
search_queries
├── id (UUID, PRIMARY KEY)
├── user_id (UUID) → users.id (FK, CASCADE DELETE)
├── google_user_id (VARCHAR, denormalized)
├── query_text (TEXT)
├── query_type (VARCHAR)  ← 'sports', 'news', etc.
├── results_count (INT)
├── threshold_used (DECIMAL)
├── percentile_used (DECIMAL)
├── avg_similarity (DECIMAL)
├── max_similarity (DECIMAL)
├── min_similarity (DECIMAL)
└── created_at (TIMESTAMP)
```

**Relationship**: **Many:1 with users**
- One user can have many search queries
- `user_id` → `users.id` (foreign key with CASCADE DELETE)

---

### 6. `search_feedback`
```
search_feedback
├── id (UUID, PRIMARY KEY)
├── search_query_id (UUID) → search_queries.id (FK, CASCADE DELETE)
├── user_id (UUID) → users.id (FK, CASCADE DELETE)
├── google_user_id (VARCHAR, denormalized)
├── email_id (UUID)  ← Reference to emails.id (soft link)
├── thread_id (VARCHAR)  ← Gmail thread ID
├── action (VARCHAR)  ← 'clicked', 'relevant', etc.
├── similarity_score (DECIMAL)
├── rank_position (INT)
├── dwell_time_ms (INT)
├── metadata (JSONB)
└── created_at (TIMESTAMP)
```

**Relationships**:
- **Many:1 with search_queries**: `search_query_id` → `search_queries.id`
- **Many:1 with users**: `user_id` → `users.id`
- **Soft link to emails**: via `thread_id` (no formal FK)

---

### 7. `sender_affinity`
```
sender_affinity
├── id (UUID, PRIMARY KEY)
├── user_id (UUID) → users.id (FK, CASCADE DELETE)
├── google_user_id (VARCHAR, denormalized)
├── sender_email (VARCHAR)
├── sender_domain (VARCHAR)
├── total_emails (INT)
├── clicked_count (INT)
├── marked_relevant (INT)
├── marked_irrelevant (INT)
├── affinity_score (DECIMAL)
├── last_interaction (TIMESTAMP)
└── created_at (TIMESTAMP)

UNIQUE(user_id, sender_email)  ← One affinity record per user per sender
```

**Relationship**: **Many:1 with users**
- One user can have many sender affinity records
- `user_id` → `users.id` (foreign key with CASCADE DELETE)

---

## System Tables

### 8. `query_type_quantiles`
```
query_type_quantiles
├── id (UUID, PRIMARY KEY)
├── query_type (VARCHAR)  ← 'sports', 'news', 'temporal', etc.
├── percentile_10 (DECIMAL)
├── percentile_20 (DECIMAL)
├── percentile_50 (DECIMAL)
├── percentile_80 (DECIMAL)
├── percentile_90 (DECIMAL)
├── sample_count (INT)
├── last_updated (TIMESTAMP)
└── created_at (TIMESTAMP)

UNIQUE(query_type)
```

**Relationship**: **Standalone (global)**
- Not linked to specific users
- Stores aggregate statistics for all users

---

### 9. `search_metrics`
```
search_metrics
├── id (UUID, PRIMARY KEY)
├── query_type (VARCHAR)
├── date (DATE)
├── total_searches (INT)
├── total_clicks (INT)
├── total_relevant (INT)
├── total_irrelevant (INT)
├── avg_ctr (DECIMAL)
├── avg_precision (DECIMAL)
├── avg_results_count (DECIMAL)
├── avg_threshold (DECIMAL)
└── created_at (TIMESTAMP)

UNIQUE(query_type, date)
```

**Relationship**: **Standalone (global aggregates)**
- Not linked to specific users
- Daily rollup of search metrics

---

## Entity-Relationship Diagram

```
                    ┌─────────────┐
                    │    users    │ ← PRIMARY ENTITY
                    │  (id, UUID) │
                    └──────┬──────┘
                           │
           ┌───────────────┼───────────────┬───────────────┬──────────────┐
           │               │               │               │              │
           ↓               ↓               ↓               ↓              ↓
   ┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────┐ ┌──────────────┐
   │user_settings │ │user_search_  │ │search_      │ │sender_     │ │  emails      │
   │              │ │preferences   │ │queries      │ │affinity    │ │              │
   │  1:1         │ │  1:1         │ │  1:Many     │ │  1:Many    │ │  1:Many      │
   └──────────────┘ └──────────────┘ └──────┬──────┘ └────────────┘ └──────────────┘
                                            │
                                            ↓
                                     ┌──────────────┐
                                     │search_       │
                                     │feedback      │
                                     │  Many:1      │
                                     └──────────────┘

   GLOBAL TABLES (not user-specific):
   ┌──────────────────┐    ┌──────────────────┐
   │query_type_       │    │search_metrics    │
   │quantiles         │    │                  │
   └──────────────────┘    └──────────────────┘
```

---

## Data Flow & Relationships

### User Creates Account
```
1. User authenticates with Google OAuth
2. INSERT into users (google_user_id, email, encrypted_credentials)
3. Auto-create user_search_preferences (via trigger or app logic)
4. Auto-create user_settings (via app logic)
```

### User Syncs Emails
```
1. Fetch emails from Gmail API
2. Encrypt email content with user's encryption key
3. INSERT/UPDATE emails (google_user_id = user's google_user_id)
   - Links to user via google_user_id (not formal FK)
```

### User Performs Search
```
1. POST /chat with message
2. Classify query type (sports, news, etc.)
3. Get adaptive threshold from query_type_quantiles
4. Perform semantic search on emails
5. INSERT into search_queries (user_id, query_text, threshold_used, ...)
6. Return results + search_query_id
```

### User Clicks Email Result
```
1. POST /search-feedback (search_query_id, user_id, thread_id, action='clicked')
2. INSERT into search_feedback
3. UPDATE sender_affinity (increment clicked_count, recalculate affinity_score)
4. UPDATE user_search_preferences (increment total_clicks, recalculate avg_ctr)
```

---

## Key Design Decisions

### Why `user_id` (UUID) AND `google_user_id` (VARCHAR)?

**Two different use cases**:

1. **`users.id` (UUID)**: Internal database primary key
   - Used for foreign key relationships
   - Ensures referential integrity
   - CASCADE DELETE works properly

2. **`users.google_user_id` (VARCHAR)**: Google's OAuth identifier
   - Used for authentication lookups
   - Matches what Google provides
   - Used in application logic for queries

**Denormalization Strategy**:
- Tables store BOTH `user_id` (FK to users.id) AND `google_user_id` (denormalized)
- Why? Query performance - avoid JOIN to users table just to get google_user_id
- Trade-off: Slight data redundancy for significant performance gain

### Why Soft Links to `emails` table?

The `emails` table relationship is **intentionally loose**:

**Reasons**:
1. **Encryption complexity**: Some emails are encrypted, some plain text
2. **Mixed storage modes**: Different encryption keys per user
3. **thread_id is stable**: Gmail thread IDs don't change
4. **Avoid circular dependencies**: emails → users, feedback → emails would create cycles

**Solution**: Use `thread_id` as a soft link instead of formal foreign key

---

## Migration Strategy

### Current State (After All Migrations)

```sql
-- 1. Core tables (exist)
users (id, google_user_id, email, ...)
emails (id, google_user_id, thread_id, ...)

-- 2. User preferences (exist)
user_settings (user_id → users.id)

-- 3. Adaptive search tables (NEW - use v2 migration)
user_search_preferences (user_id → users.id)
search_queries (user_id → users.id)
search_feedback (user_id → users.id, search_query_id → search_queries.id)
sender_affinity (user_id → users.id)
query_type_quantiles (global)
search_metrics (global)
```

### Recommended Migration Path

**Use `adaptive_search_schema_v2.sql`** instead of v1:

✅ **v2 improvements**:
- Proper foreign keys: `user_id → users.id`
- Denormalized `google_user_id` for performance
- Cascade deletes when user is removed
- Clear parent-child relationships

❌ **v1 problems**:
- Only used `google_user_id` (VARCHAR)
- No formal foreign keys to users table
- Manual cascade delete handling required

---

## Database Cohesion Summary

### Is the database cohesive? **Yes, with v2 migration!**

**Hierarchy**:
```
users (root)
├── user_settings (1:1, FK)
├── user_search_preferences (1:1, FK)
├── search_queries (1:many, FK)
│   └── search_feedback (1:many, FK)
├── sender_affinity (1:many, FK)
└── emails (1:many, soft link via google_user_id)

Global tables (no user FK):
- query_type_quantiles
- search_metrics
```

**Key Principles**:
1. ✅ All user data links to `users.id` via foreign keys
2. ✅ CASCADE DELETE ensures cleanup when user is deleted
3. ✅ Denormalized `google_user_id` for performance
4. ✅ Soft links where appropriate (emails via thread_id)
5. ✅ Global tables separate from user data

**What to do**: Use `adaptive_search_schema_v2.sql` for the migration!
