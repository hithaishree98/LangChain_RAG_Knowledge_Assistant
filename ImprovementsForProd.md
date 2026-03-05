# Future Improvements & Production Path

# Where current app stands today

This is a prototype that supports document ingestion, retrieval-augmented question answering, multi-turn conversations, and logging/analytics. This works well for local deployments and small teams. 

If the system were deployed in prod with many users and larger document collections, several improvements would be required.

# Scaling improvements

**1. PostgreSQL instead of SQLite**

Right now the chat logs, document metadata, session history are stored in a single SQLite file. SQLite works well for local development but does not fit when the requirement is for more users and has to handle many concurrnet writes.

For prod, SQLite needs to be replaced with PostgreSQL which can handle concurrent writes better and better indexing for large datasets.

In current verison, document itself is not stored as embedding are being stored in Chroma.
If we need to store user documents then propbably S3 needs to be used.

**2. Async document processing**

Cureenlty document uploads are processng synchronously. When a file is uploaded, the system immediately parses it, splits it into chunks, generates embeddings, and stores them in the vector store.

and chromadb need to be replaved woth a betetr vectore database for prod
 ad if we are scaling we would need messgae queus for workers 


**3. Slack notification triggering**

Currently Slack notifications are sent for every query when the user enables the feature.

For prod, this triggering will have to be tied to specific events like when confidence is low, document indexing fails or spikes in query volume.

**5.  Real Authentication**

The current workspace and passkey combination works for basic workspace isolation.

For prod, this has to be replaced with proper authentication.

