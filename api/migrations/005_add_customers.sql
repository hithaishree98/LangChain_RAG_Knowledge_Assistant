-- Migration 005: Add customer workspaces table
-- Each customer account is a tenant. FDE is the operator.

CREATE TABLE IF NOT EXISTS customers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL,
    slug           TEXT    NOT NULL UNIQUE,
    fde_user_id    TEXT    NOT NULL,
    last_call_date TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_fde ON customers(fde_user_id);
CREATE INDEX IF NOT EXISTS idx_customers_slug ON customers(slug);

-- Extend document_store with versioning and type metadata
-- SQLite does not support IF NOT EXISTS on ALTER TABLE, but the migration
-- runner tracks applied migrations via schema_migrations, so each file runs
-- exactly once and these statements are safe to include directly.
ALTER TABLE document_store ADD COLUMN doc_type TEXT;
ALTER TABLE document_store ADD COLUMN doc_date TEXT;
ALTER TABLE document_store ADD COLUMN is_latest_version INTEGER DEFAULT 1;
ALTER TABLE document_store ADD COLUMN doc_version_group TEXT;
-- doc_version_group = "{user_id}::{doc_type}" groups all versions of same type per customer
