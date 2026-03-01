# Future Improvements & Production Path

# Where current app stands today

This is a working POC that does the core job well. You can upload documents, ask questions in plain English, get grounded answers with source references and confidence scores, and have multi-turn conversations that actually maintain context. The architecture is clean, data is isolated per workspace, and there's a full audit trail.

# Scaling improvements

**1. PostgreSQL instead of SQLite**

Right now the chat logs, document records, session history, and audit trail all live in a single SQLite file on disk. I added WAL mode (Write-Ahead Logging) which lets reads happen while writes are in progress, and that handles low concurrency fine.

But if many users are uploading documents at the same time, writes start queuing and some fail. And also SQLite can't replicate, can't distribute across machines.

I'd replace this with PostgreSQL.

**Pinecone instead of Embedded ChromaDB**

