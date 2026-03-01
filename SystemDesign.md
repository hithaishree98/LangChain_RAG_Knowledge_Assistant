# Architecture & Design Decisions

This document explains the architectural choices, tradeoffs, problems that forced me to rethink things.

## High Level Architecture

```
┌─────────────────────┐         ┌──────────────────────────────────┐
│   Streamlit Frontend │ ──────▶ │        FastAPI Backend           │
│   (Port 8501)        │  HTTP   │        (Port 8000)               │
│                      │         │                                  │
│  - Chat UI           │         │  - /chat                         │
│  - File upload       │         │  - /upload-doc                   │
│  - Doc management    │         │  - /list-docs                    │
└─────────────────────┘          │  - /delete-doc                    │
                                 │  - /analytics                     │
                                 │  - /audit-log                     │
                                 │  - /answer-questionnaire          │
                                 └──────────┬────────────────────────┘
                                            │
                          ┌─────────────────┼─────────────────┐
                          │                 │                 │
                    ┌─────▼──────┐   ┌──────▼─────┐   ┌──────▼──────┐
                    │  ChromaDB  │   │   SQLite    │   │  Groq API   │
                    │ (vectors)  │   │ (logs/docs) │   │   (LLM)     │
                    └────────────┘   └─────────────┘   └─────────────┘
```
