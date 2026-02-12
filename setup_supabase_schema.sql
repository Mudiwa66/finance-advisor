-- Supabase Schema Setup
-- Run this in your Supabase Dashboard → SQL Editor

-- Create users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_hash TEXT UNIQUE NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    total_messages INTEGER DEFAULT 0,
    CONSTRAINT phone_hash_length CHECK (length(phone_hash) = 12)
);
CREATE INDEX idx_users_phone_hash ON users(phone_hash);

-- Create transactions table
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    amount DECIMAL(12, 2) NOT NULL,
    balance DECIMAL(12, 2) NOT NULL,
    merchant TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    statement_source TEXT
);
CREATE INDEX idx_transactions_user_date ON transactions(user_id, date DESC);
CREATE INDEX idx_transactions_merchant ON transactions(user_id, merchant);

-- Create metrics table
CREATE TABLE metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    message_text TEXT NOT NULL,
    used_llm BOOLEAN NOT NULL DEFAULT FALSE,
    llm_response_time REAL,
    total_response_time REAL NOT NULL,
    success BOOLEAN NOT NULL DEFAULT TRUE,
    tokens_used INTEGER DEFAULT 0,
    llm_model TEXT DEFAULT 'llama3.2'
);
CREATE INDEX idx_metrics_user_timestamp ON metrics(user_id, timestamp DESC);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp DESC);
