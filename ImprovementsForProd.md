# Future Improvements & Production Path

# Where current app stands today

This is a working POC that does the core job well. You can upload documents, ask questions in plain English, get grounded answers with source references and confidence scores, and have multi-turn conversations that actually maintain context. The architecture is clean, data is isolated per workspace, and there's a full audit trail.

# Scaling improvements

**1. PostgreSQL instead of SQLite**

Right now the chat logs, document records, session history, and audit trail all live in a single SQLite file on disk. I added WAL mode (Write-Ahead Logging) which lets reads happen while writes are in progress, and that handles low concurrency fine.

But if many users are uploading documents at the same time, writes start queuing and some fail. And also SQLite can't replicate, can't distribute across machines.

I'd replace this with PostgreSQL.

**2. Failed document index to DLQ**

Right now if a document fails to index, the user gets an error and that's it.

I'd replace this with allowing retry for 3 times and then adding the document to Dead Letter Queue for human review and sending a notification.

**3. Slack notification triggering**

Right now the Slack notification fires on every query where the user has enabled it.

I'd replace with configurable triggering like only when confidence is low or when it needs human review or according to user needs.

**4. Async document processing**

Right now everything runs in the same process. One large PDF upload can block the API while it parses, chunks, embeds, and indexes. In prod, if someone uploads a file while another person is asking a question, the chat response slows down.

I'd replace this by decoupling uploads from chat. 
Upload → S3 (store file) → SQS (queue job) → Worker (process in background)
Chat   → FastAPI → Groq → Response (always fast, unaffected by uploads)

**5.  Real Authentication**

Right now the workspace and passkey system works well for a demo.

I'd replace this with proper suthentication flows, JWT tokens.

**6. Memory of Previous sessions**

Right now chat history exists only within a session. Every session starts from scratch.

I'd replace this with call memory per customer account. After a session ends, a summary gets saved — what was asked, what was answered, what was unresolved, what was promised. That summary becomes part of the context for the next session.

**7. Export option for bulk questionnaire**

Right now the bulk questionnaire mode generates answers but they stay inside the tool.

I'd add export option that makes the bulk mode actually usable end to end.
