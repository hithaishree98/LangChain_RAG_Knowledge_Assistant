# FDE Customer Context Assistant (FastAPI · Streamlit · Chroma · Groq)

The FDE Customer Context Assistant is a RAG-powered document Q&A tool built for customer-facing technical teams. Upload PDFs, Word docs, or HTML files and query them conversationally. The assistant only answers from what's in your documents and if the answer isn't there, it says so. That's intentional. 
In a customer context, a confident wrong answer is worse than admitting uncertainty.

## Problem Context

While learning about RAG systems and LangChain, one specific pain point I'd observed and read about in SE/FDE workflows: customer-facing engineers spend a surprising amount of time during calls or follow-ups just searching for answers that already exist somewhere in a document. A product manual, a technical spec, a contract, an FAQ. The knowledge is not instantly accessible when you need it.

That's what this project is. It started as a way for me to deeply learn RAG architecture, but I built it with a real SE/FDE use case in mind, a tool that lets you upload your customer documents and ask questions against them in plain English. No Ctrl+F, no digging through folders. Just ask and get a grounded answer with a source reference.

## Core capabilities

- Upload and index PDF, DOCX, and HTML documents
- Ask questions in natural language, get answers with source references
- Multi-turn conversation with session memory
- Confidence scoring on every response
- Bulk mode let's you upload a CSV of questions, get all answers back at once (useful for RFPs or questionnaires)
- List and delete indexed files per user
- Slack notifications when confidence drops below threshold, escalate to a human automatically
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
   Top-k most relevant chunks are retrieved
        ↓
   Chunks + question are sent to Groq Llama-3.1-8B with a guardrail prompt
        ↓
   Model answers strictly from context — instructed to say "I don't know" if not found

3. Response
   Answer returned with confidence score and source file references
        ↓
   If confidence < 0.4 → flagged for escalation, Slack notification sent (if configured)
        ↓
   Everything logged to SQLite — session, question, answer, confidence, sources
```

## Setup Instructions

Docker and Docker Compose installed on your machine

Set these variables in .env
- API_BASE_URL=http://localhost:8000
- A free Groq API key
- Slack url
- API_KEY (if configured)

### docker-compose up --build

## Open the app
- Chat interface: http://localhost:8501
- API documentation: http://localhost:8000/docs

## First time using it

1. Go to the sidebar and upload a PDF, DOCX, or HTML file
2. Wait a few seconds for it to index
3. Type a question in the chat about your document
4. You'll get an answer with a confidence score and the source file it came from
