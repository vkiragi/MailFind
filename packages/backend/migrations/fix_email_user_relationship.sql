-- Migration: Fix email-user relationship
-- This ensures emails are properly linked to users

-- Step 1: Make sure google_user_id column exists in emails table
ALTER TABLE emails
  ADD COLUMN IF NOT EXISTS google_user_id VARCHAR(255);

-- Step 2: Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_emails_google_user_id ON emails(google_user_id);

-- Step 3: For existing NULL google_user_id values, populate from the first user
-- (Only run this if you have emails with NULL google_user_id)
UPDATE emails
SET google_user_id = (SELECT google_user_id FROM users LIMIT 1)
WHERE google_user_id IS NULL;

-- Step 4: Add foreign key constraint (optional - uncomment if you want strict referential integrity)
-- ALTER TABLE emails
--   ADD CONSTRAINT fk_emails_users
--   FOREIGN KEY (google_user_id)
--   REFERENCES users(google_user_id)
--   ON DELETE CASCADE;

-- Step 5: Verify the relationship
-- Run this query to see how many emails are linked:
-- SELECT
--   u.email,
--   u.google_user_id,
--   COUNT(e.id) as email_count
-- FROM users u
-- LEFT JOIN emails e ON e.google_user_id = u.google_user_id
-- GROUP BY u.email, u.google_user_id;
