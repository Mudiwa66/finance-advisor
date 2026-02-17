-- Migration: Add statement_uploads table for tracking uploaded PDFs
-- Run this in Supabase SQL Editor

CREATE TABLE IF NOT EXISTS statement_uploads (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    date_from   DATE NOT NULL,
    date_to     DATE NOT NULL,
    transaction_count INTEGER NOT NULL DEFAULT 0,
    storage_path TEXT,  -- path in Supabase Storage (nullable if storage failed)
    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Prevent duplicate statements for the same user and date range
CREATE UNIQUE INDEX IF NOT EXISTS idx_statement_uploads_unique
    ON statement_uploads(user_id, date_from, date_to);

-- Index for querying a user's upload history
CREATE INDEX IF NOT EXISTS idx_statement_uploads_user
    ON statement_uploads(user_id, uploaded_at DESC);

-- Allow upsert on transactions to handle re-uploads gracefully
-- (transactions table needs a unique constraint on user_id + date + description + amount)
ALTER TABLE transactions
    ADD CONSTRAINT IF NOT EXISTS uq_transaction
    UNIQUE (user_id, date, description, amount);
