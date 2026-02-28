# LangChain RAG Knowledge Assistant (FastAPI · Streamlit · Chroma · Groq)

A Retrieval-Augmented Generation (RAG) chatbot that turns uploaded **PDF/DOCX/HTML** into grounded, citation-style answers.  
Backend powered by **FastAPI + LangChain**, vector search with **Chroma** (Hugging Face **nomic-embed-text-v1.5**), and generation with **Groq Llama-3.1-8B**.  
Frontend built with **Streamlit** for easy chat and file ingestion. Includes a **CSV-driven evaluation harness** (no API changes) that reports **semantic similarity**, **key-facts coverage**, **p50/p95 latency**, and **error rate**.

---

## Key Features
- **Document ingestion:** Upload **PDF/DOCX/HTML**; parse, chunk, embed, and index into **Chroma**.  
- **Grounded Q&A:** Retrieve top-k relevant chunks and answer **only from context** using **Groq Llama-3.1-8B** + guardrail prompts.  
- **Session history:** Threaded conversations via `session_id`.  
- **Doc management:** List and delete indexed documents.  
- **Eval harness:** CSV-driven script tracking semantic similarity, key-facts coverage, p50/p95 latency, and error rate.

---

## How It Works
1. **Upload:** File → parse (loader) → chunk (splitter) → embed (HF embeddings) → store in **Chroma** with metadata (`file_id`).  
2. **Ask:** User question → (optional) history-aware rewrite → retrieve top-k chunks from **Chroma** → prompt LLM with **context + question**.  
3. **Answer:** LLM returns a concise answer grounded by retrieved text. If not in context, it can say **“I don’t know.”**  
4. **Persist:** Chat logs and document metadata stored in **SQLite**.

---

## Tech Stack
- **Backend:** FastAPI, LangChain  
- **Frontend:** Streamlit  
- **Vector DB:** Chroma (persisted at `./chroma_db`)  
- **Embeddings:** Hugging Face `nomic-ai/nomic-embed-text-v1.5`  
- **LLM:** Groq `llama-3.1-8b-instant` (via `langchain_groq`)  
- **Parsers:** `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredHTMLLoader`  
- **DB:** SQLite (chat logs & doc list)
