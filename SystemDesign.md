# Architecture & Design Decisions

## High Level Architecture

Two separate services — a FastAPI backend and a Streamlit frontend, talking to each other over HTTP.

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

- **Data Isolation**
  
   Every document chunk and every database row is tagged with a user_id.
   Chroma filters on user_id at query time. SQLite queries include WHERE user_id = ?.

- **Retrieval Augmented Generation (RAG)**
  
   RAG stores documents as vectors and retrieves only the relevant chunks at query time.

   Since customer data changes constantly, fine-tuning would require retraining every time anything changes.

- **Vector Similarity Search**
  
  Text gets converted into high-dimensional vectors where semantic similarity maps to mathematical closeness.

- **LLM Determinism — temperature set to zero**
  
  Temperature controls how random the LLM's output is.

  For an FDE using this in front of a customer, answers that randomly vary on every run would completely undermine trust. Hence temperature is set to 0 deliberately to reduce randomness.

- **Prompt and Guardrails**

  Prompt is structured as role, context, task, constraints and output.

  The model is strictly instructed to use only retrieved context, never rely on outside knowledge, never fabricate details, and return a fixed fallback sentence when evidence is missing.
  
- **Fault Tolerance and Degradation**
  
  Things fail in production like LLMs go down, databases lock, networks timeout.

   - The system handles each failure independently. 
     - LLM failures retry with exponential backoff (wait 1s, then 2s, then return 503).
     - Slack notification failures don't affect the answer.
     - Database write failures don't corrupt the session.

- **Idempotency and Atomic Operations**
  
  If document indexing fails halfway through, the database record gets cleaned up.

  Temp files are deleted in a finally block so they're always cleaned up even if an exception is thrown. 

  Duplicate uploads are rejected with a 409 before any work is done.

- **Input Validation**

  Every request is checked before any work starts. A 200MB file or a 10,000-character question gets rejected immediately with a clear 400 error. 
  

- **Observability**

  - Structured JSON logs for debugging individual requests
  - business metrics via /analytics for understanding usage patterns
  - audit trail of every query and response.

- **Health check**
- 
    To check whether the database, vector store, and LLM key are all working.
 
- **Security**
  
  - Input validation blocks malformed requests (max file size, allowed extensions, max question length).
   - Authentication via API key header.
   - Authorization via user_id filtering at the database level
   - LLM guardrail prompt prevents hallucination
 
- **Deterministic Hashing**
  
  The workspace and passkey get hashed together to produce a consistent user_id.

  Same inputs always produce the same hash. Different passkey produces a completely different hash. 

  This gives persistent workspace identity without a user table, password storage, or session management.

- **Logging**
  
  Custom StructuredLogger that writes every event as a single line of JSON. Every API call, LLM retry, upload, deletion, and error gets logged.

  JSON was preferred as it is parseable. We can filter by event type, timestamp, and also answer questions like "how many queries had confidence below 0.4 this week?".

## Tools used
- Groq
- nomic-embed-text
- ChromaDB
- SQLite
- LangChain 
