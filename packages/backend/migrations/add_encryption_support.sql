-- Migration: Add encryption support for zero-knowledge architecture
-- This migration adds fields for client-side encrypted email storage

-- Add new fields to emails table for encryption
ALTER TABLE emails
  ADD COLUMN IF NOT EXISTS encrypted_content TEXT,
  ADD COLUMN IF NOT EXISTS encrypted_embedding TEXT,
  ADD COLUMN IF NOT EXISTS iv TEXT,
  ADD COLUMN IF NOT EXISTS sender_domain VARCHAR(255),
  ADD COLUMN IF NOT EXISTS thread_id_hash VARCHAR(64);

-- Create index on sender_domain for faster filtering
CREATE INDEX IF NOT EXISTS idx_emails_sender_domain ON emails(sender_domain);

-- Create index on thread_id_hash for deduplication
CREATE INDEX IF NOT EXISTS idx_emails_thread_id_hash ON emails(thread_id_hash);

-- Add storage_mode to user_settings
ALTER TABLE user_settings
  ADD COLUMN IF NOT EXISTS storage_mode VARCHAR(20) DEFAULT 'encrypted_cloud',
  ADD COLUMN IF NOT EXISTS retention_days INT DEFAULT NULL,
  ADD COLUMN IF NOT EXISTS encryption_enabled BOOLEAN DEFAULT TRUE;

-- Create audit_logs table for compliance
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address VARCHAR(45),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for audit log queries
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp);

-- Add comment explaining the encryption model
COMMENT ON COLUMN emails.encrypted_content IS 'AES-GCM encrypted JSON blob containing subject, sender, and content. Server cannot decrypt.';
COMMENT ON COLUMN emails.encrypted_embedding IS 'AES-GCM encrypted vector embedding. Server cannot decrypt.';
COMMENT ON COLUMN emails.iv IS 'Initialization vector for AES-GCM decryption. Unique per email.';
COMMENT ON COLUMN emails.sender_domain IS 'Plain-text sender domain (e.g., github.com) for server-side filtering. No PII.';
COMMENT ON COLUMN emails.thread_id_hash IS 'SHA-256 hash of thread_id for deduplication without revealing actual ID.';
