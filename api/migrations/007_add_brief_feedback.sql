-- Migration 007: FDE feedback on brief sections

CREATE TABLE IF NOT EXISTS brief_feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    brief_log_id  INTEGER REFERENCES brief_logs(id),
    customer_id   TEXT    NOT NULL,
    section       TEXT    NOT NULL,
    rating        INTEGER NOT NULL CHECK(rating IN (1, -1)),
    flagged_claim TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_feedback_customer ON brief_feedback(customer_id);
CREATE INDEX IF NOT EXISTS idx_feedback_brief ON brief_feedback(brief_log_id);
