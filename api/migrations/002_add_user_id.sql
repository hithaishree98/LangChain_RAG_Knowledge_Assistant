ALTER TABLE application_logs ADD COLUMN user_id TEXT DEFAULT 'default';
ALTER TABLE document_store   ADD COLUMN user_id TEXT DEFAULT 'default';