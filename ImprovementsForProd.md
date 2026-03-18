# Future Improvements & Production Path

# Where current app stands today

This is a prototype that supports document ingestion, retrieval-augmented question answering, multi-turn conversations, and logging/analytics. This works well for local deployments and small teams. 

If the system were deployed in prod with many users and larger document collections, several improvements would be required.

# Scaling improvements

**1. PostgreSQL instead of SQLite**

Right now the chat logs, document metadata, session history are stored in a single SQLite file. SQLite works well for local development but does not fit when the requirement is for more users, has to handle many concurrent writes and horizontally scale.

For prod, SQLite needs to be replaced with PostgreSQL which can handle concurrent writes better and better indexing for large datasets.

In current verison, raw document itself is not being stored as embeddings are being stored in Chroma.
If we need to store raw user documents then object storage S3 needs to be used.

**2. Async document processing**

Currently document uploads are processed synchronously. When a file is uploaded, the system immediately parses it, splits it into chunks, generates embeddings, and stores them in the vector store. For large documents, this can block API requests.

In prod, this would have to be handled asynchronously using message queues and worker.
```
Upload request
      ↓
Message queue (SQS / Redis)
      ↓
Worker service
      ↓
Parsing + chunking + embedding
      ↓
Vector database
```

And Chromadb need to be replaced with distributed vector databases like Pinecone for prod.

**3. Slack notification triggering**

Currently Slack notifications are sent for every query when the user enables the feature.

For prod, this triggering will have to be tied to specific events like when confidence is low, document indexing fails or spikes in query volume.

**4.  Real Authentication**

The current workspace and passkey combination works for basic workspace isolation.

For prod, this has to be replaced with proper authentication.

