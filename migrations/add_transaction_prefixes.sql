-- Transaction description prefix stripping — moves hardcoded list out of app.py

CREATE TABLE IF NOT EXISTS transaction_prefixes (
    id          UUID  DEFAULT gen_random_uuid() PRIMARY KEY,
    bank        TEXT  NOT NULL,
    prefix      TEXT  NOT NULL,
    category    TEXT,                          -- nullable, for future use
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_bank_prefix UNIQUE (bank, prefix)
);

CREATE INDEX IF NOT EXISTS idx_transaction_prefixes_bank ON transaction_prefixes(bank);

-- Seed FNB prefixes (longest first to ensure correct match priority)
INSERT INTO transaction_prefixes (bank, prefix) VALUES
    ('FNB', 'Card Purchase With Cashback'),
    ('FNB', 'Chq Card ATM Local Cash Advanc Cash'),
    ('FNB', 'Refund Chq Card Purchase Cr Vc'),
    ('FNB', 'Rtc Express Credit'),
    ('FNB', 'Rtc Express Pmt To'),
    ('FNB', 'Paypal Withdrawal'),
    ('FNB', 'Electricity Prepaid'),
    ('FNB', 'Internet Airtime'),
    ('FNB', 'Airtime Topup Airtime'),
    ('FNB', 'Payment 1Day Cr'),
    ('FNB', 'Payshap Credit'),
    ('FNB', 'Fuel Purchase'),
    ('FNB', 'Card Cashback Cashb'),
    ('FNB', 'Card Purchase'),
    ('FNB', 'Internet Pmt To'),
    ('FNB', 'POS Purchase'),
    ('FNB', 'Magtape Credit'),
    ('FNB', 'Magtape Debit'),
    ('FNB', 'Send Money App Dr Send'),
    ('FNB', 'Send Money Dr Send'),
    ('FNB', 'FNB App Transfer From'),
    ('FNB', 'FNB App Payment To'),
    ('FNB', 'FNB App Payment From'),
    ('FNB', 'FNB App Rtc Pmt To'),
    ('FNB', 'FNB OB Pmt'),
    ('FNB', 'Payment To'),
    ('FNB', 'Rtc Credit'),
    ('FNB', 'Byc Debit'),
    ('FNB', 'ATM Cash')
ON CONFLICT DO NOTHING;
