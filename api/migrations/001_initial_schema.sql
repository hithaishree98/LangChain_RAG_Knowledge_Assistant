
CREATE TABLE IF NOT EXISTS application_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,
    user_query   TEXT,
    gpt_response TEXT,
    model        TEXT,
    confidence   REAL    DEFAULT 0.0,
    escalated    INTEGER DEFAULT 0,
    sources      TEXT    DEFAULT '',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_store (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    filename         TEXT,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);