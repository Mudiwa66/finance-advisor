-- Supabase Row-Level Security Setup
-- Run this AFTER running setup_supabase_schema.sql

-- Enable RLS on all tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;

-- Users table policies
CREATE POLICY "Users can view own data" ON users FOR SELECT
    USING (phone_hash = current_setting('app.current_user_hash', TRUE));

CREATE POLICY "Anyone can insert users" ON users FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Users can update own data" ON users FOR UPDATE
    USING (phone_hash = current_setting('app.current_user_hash', TRUE));

-- Transactions table policies
CREATE POLICY "Users can view own transactions" ON transactions FOR SELECT
    USING (user_id IN (
        SELECT id FROM users
        WHERE phone_hash = current_setting('app.current_user_hash', TRUE)
    ));

CREATE POLICY "Anyone can insert transactions" ON transactions FOR INSERT
    WITH CHECK (true);

-- Metrics table policies
CREATE POLICY "Users can view own metrics" ON metrics FOR SELECT
    USING (user_id IN (
        SELECT id FROM users
        WHERE phone_hash = current_setting('app.current_user_hash', TRUE)
    ));

CREATE POLICY "Anyone can insert metrics" ON metrics FOR INSERT
    WITH CHECK (true);
