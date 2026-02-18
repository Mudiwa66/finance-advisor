-- Conversational memory: stores message exchanges per user
CREATE TABLE IF NOT EXISTS chat_history (
    id              UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id         UUID        NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_message    TEXT        NOT NULL,
    bot_response    TEXT        NOT NULL,
    intent_detected TEXT,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_history_user_time
    ON chat_history(user_id, timestamp DESC);

-- Auto-delete history older than 30 days (runs on insert via trigger)
CREATE OR REPLACE FUNCTION cleanup_old_chat_history()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    DELETE FROM chat_history
    WHERE user_id = NEW.user_id
      AND timestamp < NOW() - INTERVAL '30 days';
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_cleanup_chat_history ON chat_history;
CREATE TRIGGER trg_cleanup_chat_history
    AFTER INSERT ON chat_history
    FOR EACH ROW EXECUTE FUNCTION cleanup_old_chat_history();
