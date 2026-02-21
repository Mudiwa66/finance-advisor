-- Conversational onboarding state on the users table

-- Add columns; DEFAULT TRUE so existing users skip onboarding
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_step     TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS monthly_income      DECIMAL(12,2);

-- New users default to FALSE (must complete onboarding)
ALTER TABLE users ALTER COLUMN onboarding_completed SET DEFAULT FALSE;
