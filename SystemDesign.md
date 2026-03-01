# Architecture & Design Decisions

This document explains the architectural choices, tradeoffs, problems that forced me to rethink things.

## High Level Architecture

```
        
   Streamlit Frontend  ──────▶         FastAPI Backend           
   (Port 8501)           HTTP            (Port 8000)               
                                                                  
  - Chat UI                            - /chat                         
  - File upload                        - /upload-doc                   
  - Doc management                     - /list-docs                    
                                       - /delete-doc                    
                                       - /analytics                                  
                                       - /audit-log                     
                                       - /answer-questionnaire          
                          
                                            │
                          ┌─────────────────┼─────────────────┐
                   
                      ChromaDB           SQLite           Groq API   
                     (vectors)         (logs/docs)         (LLM)     

```

## How a document becomes queryable

This is the core of the system and where I spent most of my time getting things right. When you upload a PDF, it goes through a pipeline before it's ready to answer questions.

- **Parsing** — the file gets read by the right loader depending on its type. PDFs use PyPDF, Word docs use Docx2txt, HTML uses Unstructured. This gives raw text.

- **Chunking** — the raw text gets split into overlapping chunks. The overlap matters. If a sentence spans the boundary between two chunks and you split it cleanly, you lose context. The overlap means each chunk shares a little content with the next one so nothing important falls through the gap.

- **Embedding** — each chunk is converted into a vector (a list of numbers representing its meaning) using the HuggingFace nomic-embed-text-v1.5 model. Two chunks about similar topics will have vectors that are mathematically close to each other. This is what makes semantic search work, we're not matching words, we're matching meaning.

- **Storing** — the vectors and original chunk text are stored in ChromaDB with metadata like which file they came from, which user uploaded them.

When someone asks a question, it goes through the same embedding step and ChromaDB finds the chunks whose vectors are closest. Those chunks become the context the LLM uses to answer.

## System design concepts in this project

- **Client-Server Architecture**
  
The frontend and backend are completely separate services. Streamlit handles the UI, FastAPI handles everything else. They communicate over HTTP.

 Someone could replace the Streamlit frontend with a Slack bot or a Chrome extension without touching a single line of the RAG logic. That separation was intentional from the start.

- **Multi-Tenancy and Data Isolation**
  
One deployment, many workspaces, zero data leakage between them. Every document chunk and every database row is tagged with a user_id.

Chroma filters on user_id at query time. SQLite queries include WHERE user_id = ?.

- **Retrieval Augmented Generation (RAG)**
  
RAG stores documents as vectors and retrieves only the relevant chunks at query time.

Since customer data changes constantly, fine-tuning would require retraining every time anything changes.

- **Vector Similarity Search**
  
Text gets converted into high-dimensional vectors where semantic similarity maps to mathematical closeness.

- **Fault Tolerance and Graceful Degradation**
  
Things fail in production like LLMs go down, databases lock, networks timeout.

 - The system handles each failure independently. 
   - LLM failures retry with exponential backoff (wait 1s, then 2s, then return 503).
   - Slack notification failures don't affect the answer.
   - Database write failures don't corrupt the session.

The pattern throughout: non-critical paths fail silently and log the error, critical paths retry and surface the error cleanly.

- **Idempotency and Atomic Operations**
  
If document indexing fails halfway through, the database record gets cleaned up.

Temp files are deleted in a finally block so they're always cleaned up even if an exception is thrown. 

Duplicate uploads are rejected with a 409 before any work is done. 

- **Observability**

  - Structured JSON logs for debugging individual requests
  - business metrics via /analytics for understanding usage patterns
  - audit trail of every query and response.
