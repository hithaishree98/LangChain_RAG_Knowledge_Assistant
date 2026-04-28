CREATE TABLE IF NOT EXISTS brief_logs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id      TEXT    NOT NULL,
    query            TEXT    NOT NULL,
    brief_json       TEXT    NOT NULL,
    faithfulness_score REAL  DEFAULT 0.0,
    loop_count       INTEGER DEFAULT 0,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_brief_logs_customer
    ON brief_logs(customer_id, created_at DESC);
