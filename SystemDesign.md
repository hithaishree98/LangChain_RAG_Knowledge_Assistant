# Architecture & Design Decisions

## High Level Architecture

Two separate services, a FastAPI backend and a Streamlit frontend, talking to each other over HTTP.


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
                   
                      ChromaDB           SQLite             Groq API   
                     (vectors)    (logs/analytics/audit)      (LLM)     

```

## How a document becomes queryable

This is the core of the system and where I spent most of my time getting things right. When user uploads a PDF, it goes through a pipeline before it's ready to answer questions.

- **Parsing** — the file gets read by the right loader depending on its type. PDFs use PyPDF, Word docs use Docx2txt, HTML uses Unstructured. This gives raw text.

- **Chunking** — the raw text gets split into overlapping chunks. The overlap matters for context. The overlap means each chunk shares a little content with the next one so nothing important falls through the gap.

- **Embedding** — each chunk is converted into a vector (a list of numbers representing its meaning) and HuggingFace nomic-embed-text-v1.5 model is used for this.

  **Why Embedding**
  Two chunks about similar topics will have vectors that are mathematically close to each other. This is what makes semantic search work, we're not matching words, we're matching meaning.

- **Storing** — the vectors and original chunk text are stored in ChromaDB with metadata like which file they came from, which user uploaded them.

  When someone asks a question, it goes through the same embedding step and ChromaDB finds the chunks whose vectors are closest. Those chunks become the context the LLM uses to answer.

- **History aware retrieval** - users often ask follow-up questions that depend on earlier context.

  Chat history for each session is stored in SQLite. When a new question arrives, the system retrieves the previous messages and passes them into a history-aware retriever. 

  If the question depends on earlier context, the model first rewrites it into a standalone query before performing vector search. This ensures that even short follow-up questions retrieve the correct document context.
  
## System design concepts in this project

- **Client-Server Architecture**
  
  The frontend and backend are designed to be completely separate services.
  Streamlit handles the UI, FastAPI handles the backend logic such as document ingestion, retrieval, and LLM interaction.
  They communicate thorugh API endpoints.
        
  This seperation is intentional as it allows the UI layer and the backend logic to evolve independently.

   For example, the Streamlit frontend could be replaced with another interface such as a Slack bot or a browser extension without changing the RAG pipeline. Similarly, changes to the LLM model or retrieval logic can be made in the backend without affecting the frontend.

- **Multi Tenancy & Data Isolation (Deterministic Hashing)**

  I did not want to implement a full authentication system at this level for this project, but I still needed a way to isolate user data especially to make sure uploaded documents are not visible to other users.

  Deterministic hashing is used to secure by producing user_id using the workspace name and passkey.

  Same workspace_name + same passkey = same hash

  Every document chunk stored in ChromaDB and every row written to SQLite is tagged with this `user_id`.

   Retrieval queries in Chroma filter by `user_id`, and SQLite queries include `WHERE user_id = ?`, ensuring that users can only access their own documents and conversation history.

- **Retrieval Augmented Generation (RAG) with Vector Similarity Search**
  
   Customer data changes constantly and data changes from one customer to another. Fine tuning a model requires training the model frequently to know everything about your customers which is not feasible.

  Hence in this usecase I found RAG as the right approach. Here documents are stored as vectors. At query time, find only the relevant chunks and send those to the LLM. Works with unlimited documents, documents can be added any time, no retraining needed.

  Text gets converted into high-dimensional vectors where semantic similarity maps to mathematical closeness. This finds the right chunk even when the question uses different words than the document unlike keyword search would miss it.

- **Data Modelling and Migration**

  The core workflow is that users upload documents and ask questions about them, and the system retrieves relevant document content to generate answers. The system also keeps logs of interactions for auditing and analytics.
  
  So main entities the application needs to store include document metadata, interaction logs and document chunks. Structured data is stored in SQLite and embeddings in Chroma.
  
  The data isolation requirement was introduced after intial schema creation database migration is used to safely update schema.

- **LLM Determinism — temperature set to zero**
  
  Temperature controls how random the LLM's output is.

  For an FDE using this in front of a customer, answers that randomly vary on every run would completely undermine trust. Hence temperature is set to 0 deliberately to reduce randomness.

- **Prompt Design and Guardrails**

  The structure followed in the propmpt is as follows role, context, task, constraints and output.

  This reduces hallucination and also model is instructed to answer only using the retrieved document context, avoid relying on outside knowledge, and never fabricate details.

  If the answer cannot be found in the provided context, it returns a fixed fallback response indicating that the information is not available.

- **Confidence_Score**

  Currently confidence score of every answer is computed based on how many chunks were retrieved and answer length in characters.

  Answers which have confidence scores less than 0.4 are marked as escalated in application_logs.
  /analytics aggregates from those logs through which we can determine
    - average confidence
    - how often questions are unsupported (low confidence / escalations)
    - which sessions/users are hitting lots of low-confidence answers
  
- **Fault Tolerance and Degradation**
  
  Few point of failures that have been handled to be resilient

     - LLM failures retry with exponential backoff then returns 503.
     - Slack notification failures logs the error but still continues to give the answer, does not break the system or chat.
     - Database write failures logs the error but still contiues to give answer to the user.
     - If Chroma indexing fails after SQLite record creation for the document, then SQLite record is deleted.
     - In bulk questionnaire mode, if any question triggers failure then it returns an fallback answer " Failed to get answer, return manually" and continues with next question.

- **Validations**
  
  - Request structure validation at API endpoints like /chat and /delete-doc using Pydantic.
    
    If the request is missing a field or has the wrong type, the API immediately returns HTTP 422 and stops processing.

  - User question validation to check if the question is empty or too long.
 
    If the question fails these checks, the API returns HTTP 400
  
  - Document upload validation to check file extension, file not empty, file size.

    If any of these checks fail, the upload is rejected and the API returns HTTP 400.

    Duplicate uploads are rejected based on filename with a 409.

  - Bulk questionnaire uploads validation checks file format, rows and columns.
 
    If any of these checks fail, the API returns HTTP 400.

  - Delete document validation if document not present.

    If no document is found, the API returns HTTP 404.     
      
- **Observability**

    Application_logs use helper functions in db_utils.py. These logs store session_id, user_id, user_query, gpt_response, model, confidence, escalated, sources.

    /audit-log endpoint shows the raw interaction logs stored in the application_logs table.
    - This is helpful in debugging user conversations, reviewing past interactions and auditing how the system responded to questions.

    /analytics endpoint shows aggregated information derived from the logs such as total number of chat requests, number of escalations, average confidence score, usage patterns.
    - This is helful in how the system is being used, how often questions cannot be answered confidently and overall performance of the RAG system.
 
    /logs shows the last N JSON log entries written by StructuredLogger.
    - This is helful in showing what operational errors happened (retries, failures, endpoint errors)
      

- **Health check**

    /health endpoint verifies SQLite connectivity, Chroma vector store availability and configuration of required environment variables.

    returns "healthy" if all checks are acceptable, else "degraded"
 

## Tools used
- Groq
- nomic-embed-text
- ChromaDB
- SQLite
- LangChain 
