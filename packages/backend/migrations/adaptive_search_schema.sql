-- Migration: Adaptive Search and Learning-to-Rank Schema
-- This migration adds tables for adaptive quantile thresholds, user feedback, and ranking features

-- 1. Query Type Quantiles Table
-- Stores rolling quantile statistics per query type
CREATE TABLE IF NOT EXISTS query_type_quantiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_type VARCHAR(50) NOT NULL, -- 'sports', 'news', 'temporal', 'latest', 'default'
    percentile_10 DECIMAL(5,4),
    percentile_20 DECIMAL(5,4),
    percentile_50 DECIMAL(5,4),
    percentile_80 DECIMAL(5,4),
    percentile_90 DECIMAL(5,4),
    sample_count INT DEFAULT 0,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(query_type)
);

-- 2. Search Queries Table
-- Track all searches for analytics and quantile updates
CREATE TABLE IF NOT EXISTS search_queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    google_user_id VARCHAR(255),
    query_text TEXT NOT NULL,
    query_type VARCHAR(50),
    results_count INT,
    threshold_used DECIMAL(5,4),
    percentile_used DECIMAL(5,4),
    avg_similarity DECIMAL(5,4),
    max_similarity DECIMAL(5,4),
    min_similarity DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Search Feedback Table
-- Record user interactions with search results
CREATE TABLE IF NOT EXISTS search_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    search_query_id UUID REFERENCES search_queries(id) ON DELETE CASCADE,
    user_id VARCHAR(255),
    google_user_id VARCHAR(255),
    email_id UUID,
    thread_id VARCHAR(255),
    action VARCHAR(50) NOT NULL, -- 'clicked', 'relevant', 'irrelevant', 'skipped', 'dismissed'
    similarity_score DECIMAL(5,4),
    rank_position INT,
    dwell_time_ms INT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. User Search Preferences Table
-- Store per-user search precision preferences and learned parameters
CREATE TABLE IF NOT EXISTS user_search_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    google_user_id VARCHAR(255) NOT NULL,
    precision_level VARCHAR(20) DEFAULT 'balanced', -- 'strict', 'balanced', 'broad'
    custom_threshold_offset DECIMAL(5,4) DEFAULT 0.0,
    preferred_quantile DECIMAL(5,4) DEFAULT 0.80,
    feature_weights JSONB, -- For learning-to-rank personalization
    total_searches INT DEFAULT 0,
    total_clicks INT DEFAULT 0,
    avg_ctr DECIMAL(5,4),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(google_user_id)
);

-- 5. Sender Affinity Table
-- Track user engagement with specific senders for ranking
CREATE TABLE IF NOT EXISTS sender_affinity (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255),
    google_user_id VARCHAR(255) NOT NULL,
    sender_email VARCHAR(500) NOT NULL,
    sender_domain VARCHAR(255),
    total_emails INT DEFAULT 0,
    clicked_count INT DEFAULT 0,
    marked_relevant INT DEFAULT 0,
    marked_irrelevant INT DEFAULT 0,
    affinity_score DECIMAL(5,4),
    last_interaction TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(google_user_id, sender_email)
);

-- 6. Search Metrics Aggregation Table
-- Daily/weekly aggregated metrics for auto-tuning
CREATE TABLE IF NOT EXISTS search_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_type VARCHAR(50),
    date DATE NOT NULL,
    total_searches INT DEFAULT 0,
    total_clicks INT DEFAULT 0,
    total_relevant INT DEFAULT 0,
    total_irrelevant INT DEFAULT 0,
    avg_ctr DECIMAL(5,4),
    avg_precision DECIMAL(5,4),
    avg_results_count DECIMAL(5,2),
    avg_threshold DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(query_type, date)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_search_queries_user ON search_queries(google_user_id);
CREATE INDEX IF NOT EXISTS idx_search_queries_created ON search_queries(created_at);
CREATE INDEX IF NOT EXISTS idx_search_queries_type ON search_queries(query_type);

CREATE INDEX IF NOT EXISTS idx_search_feedback_user ON search_feedback(google_user_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_query ON search_feedback(search_query_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_action ON search_feedback(action);
CREATE INDEX IF NOT EXISTS idx_search_feedback_created ON search_feedback(created_at);

CREATE INDEX IF NOT EXISTS idx_sender_affinity_user ON sender_affinity(google_user_id);
CREATE INDEX IF NOT EXISTS idx_sender_affinity_sender ON sender_affinity(sender_email);
CREATE INDEX IF NOT EXISTS idx_sender_affinity_score ON sender_affinity(affinity_score DESC);

CREATE INDEX IF NOT EXISTS idx_search_metrics_type_date ON search_metrics(query_type, date);

-- Initialize default quantiles for each query type
INSERT INTO query_type_quantiles (query_type, percentile_10, percentile_20, percentile_50, percentile_80, percentile_90, sample_count)
VALUES
    ('temporal', 0.10, 0.12, 0.15, 0.18, 0.20, 0),
    ('news', 0.15, 0.17, 0.20, 0.23, 0.25, 0),
    ('sports', 0.20, 0.22, 0.25, 0.28, 0.30, 0),
    ('latest', 0.15, 0.17, 0.20, 0.23, 0.25, 0),
    ('default', 0.25, 0.27, 0.30, 0.33, 0.35, 0)
ON CONFLICT (query_type) DO NOTHING;

-- Comments for documentation
COMMENT ON TABLE query_type_quantiles IS 'Rolling quantile statistics for adaptive threshold selection per query type';
COMMENT ON TABLE search_queries IS 'Historical log of all search queries with metadata for analytics';
COMMENT ON TABLE search_feedback IS 'User feedback on search results (clicks, relevance ratings)';
COMMENT ON TABLE user_search_preferences IS 'Per-user search precision preferences and learned parameters';
COMMENT ON TABLE sender_affinity IS 'User engagement metrics per sender for ranking personalization';
COMMENT ON TABLE search_metrics IS 'Aggregated search performance metrics for auto-tuning';

COMMENT ON COLUMN search_feedback.action IS 'User action: clicked, relevant, irrelevant, skipped, dismissed';
COMMENT ON COLUMN user_search_preferences.precision_level IS 'User preference: strict (fewer), balanced, broad (more results)';
COMMENT ON COLUMN sender_affinity.affinity_score IS 'Calculated score based on engagement (0.0-1.0)';
