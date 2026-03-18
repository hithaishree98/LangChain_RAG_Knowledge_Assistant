CREATE INDEX IF NOT EXISTS idx_logs_user_session
    ON application_logs(user_id, session_id);

CREATE INDEX IF NOT EXISTS idx_logs_user_created
    ON application_logs(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_docs_user
    ON document_store(user_id);