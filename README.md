# FDE Customer Context Assistant (FastAPI · Streamlit · Chroma · Groq)

The FDE Customer Context Assistant is a RAG-powered document Q&A tool built for customer-facing technical teams. Upload PDFs, Word docs, or HTML files and query them in plain english. The assistant only answers from what's in your documents and if the answer isn't there, it says so. 

In such customer context, a confident wrong answer is worse than admitting uncertainty.

## Problem Context

While learning about RAG systems, LangChain and SE/FDE workflows, one specific pain point I'd observed is that before every customer call you need to know what was last discussed, what was promised, what's broken right now, what their tech stack looks like. That information exists but it's scattered across slack, email, notion, git, perosnal notes.

That's what this project is. It started as a way for me to deeply learn RAG architecture, but I built it with a real SE/FDE use case in mind, a tool that lets you upload your customer documents that in various formats and from various platforms and ask questions against them in plain English, so you can pull all of this context together before you get on a call.

## What it does

- Upload and index PDF, DOCX, and HTML documents
- Documents are stored with user-level metadata, ensuring retrieval is scoped to the uploading user
- Ask questions in natural language, get answers with source references
- Multi-turn conversation with session memory
- Confidence scoring on every response
- Bulk mode let's you upload a CSV of questions, get all answers back at once (useful for RFPs or questionnaires)
- List and delete indexed files per user
- Optional Slack notifications when enabled and a webhook is configured.
- Full audit log of every query, answer, and confidence score
- Usage analytics per session

## How It Works
```
1. Upload
   User uploads a PDF / DOCX / HTML file
        ↓
   File is parsed by the appropriate loader (PyPDF, Docx2txt, Unstructured)
        ↓
   Content is split into overlapping chunks 
        ↓
   Each chunk is embedded using HuggingFace nomic-embed-text-v1.5
        ↓
   Embeddings + metadata stored in ChromaDB

2. Query
   User asks a question
        ↓
   If there's chat history → LangChain rewrites question to be standalone
        ↓
   Question is embedded and similarity-searched against ChromaDB
        ↓
   Top-k most relevant chunks are retrieved (scoped to the current user via metadata filtering)
        ↓
   Chunks + question are sent to llama-3.1-8b-instant with a guardrail prompt
        ↓
   Model answers strictly from context and is instructed to say "I don't know" if not found

3. Response
   Answer returned with confidence score and source file references
        ↓
   Everything logged to SQLite (session, question, answer, confidence, sources)
```
## Why this approach

Customer data changes constantly and data changes from one customer to another. Fine tuning a model requires training the model frequently to know everything about your customers which is not feasible.

Hence in this usecase I found RAG as the right approach. Here documents are stored as vectors. At query time, find only the relevant chunks and send those to the LLM. Works with unlimited documents, documents can be added any time, no retraining needed.

## Setup Instructions

Docker and Docker Compose needs to be installed on your machine

Set these variables in .env
- API_BASE_URL
- Groq API key
- Slack url

### docker-compose up --build 
Use this command to bring up the docker container and the app to be hosted locally.

## Using the App

1. Go to the sidebar and upload a PDF, DOCX, or HTML file
2. Wait a few seconds for it to index
3. Type a question in the chat about your document
4. You'll get an answer with a confidence score and the source file it came from
