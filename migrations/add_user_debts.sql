-- Debt tracking table

CREATE TABLE IF NOT EXISTS user_debts (
    id              UUID          DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         UUID          NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    creditor_name   TEXT          NOT NULL,
    debt_type       TEXT          NOT NULL DEFAULT 'other'
                        CHECK (debt_type IN ('credit_card', 'personal_loan', 'overdraft',
                                             'store_credit', 'payday_loan', 'other')),
    amount_owed     DECIMAL(12,2) NOT NULL,
    interest_rate   DECIMAL(5,2)  NOT NULL,
    minimum_payment DECIMAL(12,2),
    payment_day     INTEGER       CHECK (payment_day BETWEEN 1 AND 31),
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    updated_at      TIMESTAMPTZ   DEFAULT NOW(),
    active          BOOLEAN       DEFAULT TRUE,

    CONSTRAINT uq_user_creditor UNIQUE (user_id, creditor_name)
);

CREATE INDEX IF NOT EXISTS idx_user_debts_user_active
    ON user_debts(user_id) WHERE active = TRUE;
