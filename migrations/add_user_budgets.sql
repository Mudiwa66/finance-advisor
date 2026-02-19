-- Run this in Supabase Dashboard → SQL Editor

CREATE TABLE IF NOT EXISTS user_budgets (
    id          UUID          DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id     UUID          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category    TEXT          NOT NULL,
    amount      DECIMAL(12,2) NOT NULL,
    period      TEXT          NOT NULL CHECK (period IN ('monthly', 'weekly')),
    created_at  TIMESTAMPTZ   DEFAULT NOW(),
    updated_at  TIMESTAMPTZ   DEFAULT NOW(),
    active      BOOLEAN       DEFAULT TRUE,

    CONSTRAINT uq_user_budget_category_period UNIQUE (user_id, category, period)
);

CREATE INDEX IF NOT EXISTS idx_user_budgets_user_active
    ON user_budgets(user_id) WHERE active = TRUE;
