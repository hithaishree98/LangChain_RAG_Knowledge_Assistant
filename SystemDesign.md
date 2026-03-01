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

- Parsing — the file gets read by the right loader depending on its type. PDFs use PyPDF, Word docs use Docx2txt, HTML uses Unstructured. This gives raw text.

- Chunking — the raw text gets split into overlapping chunks. The overlap matters. If a sentence spans the boundary between two chunks and you split it cleanly, you lose context. The overlap means each chunk shares a little content with the next one so nothing important falls through the gap.

- Embedding — each chunk is converted into a vector (a list of numbers representing its meaning) using the HuggingFace nomic-embed-text-v1.5 model. Two chunks about similar topics will have vectors that are mathematically close to each other. This is what makes semantic search work, we're not matching words, we're matching meaning.

- Storing — the vectors and original chunk text are stored in ChromaDB with metadata like which file they came from, which user uploaded them.

When someone asks a question, it goes through the same embedding step and ChromaDB finds the chunks whose vectors are closest. Those chunks become the context the LLM uses to answer.
