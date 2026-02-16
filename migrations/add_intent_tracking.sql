-- Migration: Add intent classification tracking to metrics table
-- Run this in Supabase SQL Editor

-- Add intent and confidence columns to metrics table
ALTER TABLE metrics
ADD COLUMN IF NOT EXISTS intent TEXT,
ADD COLUMN IF NOT EXISTS confidence REAL;

-- Create index for intent analytics queries
CREATE INDEX IF NOT EXISTS idx_metrics_intent ON metrics(intent, timestamp DESC);

-- Add comment to document intent values
COMMENT ON COLUMN metrics.intent IS 'Detected user intent: greeting, account_balance, spending_query, debt_advice, budget_check, document_upload, general_financial_advice, unknown';
COMMENT ON COLUMN metrics.confidence IS 'Confidence score for intent classification (0.0 - 1.0)';
