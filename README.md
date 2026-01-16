# LLM Embeddings & RAG Basics

Training materials for learning to work with Large Language Models, embeddings, and Retrieval Augmented Generation (RAG) patterns in research workflows.

## What is this?

A hands-on training repository that teaches:
- How to make API calls to LLMs from Python
- What embeddings are and how to use them
- How to build RAG (Retrieval Augmented Generation) systems
- Practical applications: qualitative coding and knowledge chatbots

## Who is this for?

Research staff who want to:
- Use LLMs programmatically (not just through chat interfaces)
- Work with their own data and documents
- Build AI-augmented research tools
- Understand embeddings and semantic search

No prior LLM experience required. Basic Python knowledge helpful.

## Repository structure
```
llm-embeddings-rag-basics/
├── docs/                    # Session guides with step-by-step instructions
│   ├── session_01.md       # Local setup & first API call
│   ├── session_02.md       # Introduction to embeddings & RAG
│   ├── session_03.md       # Qualitative coding with embeddings
│   └── session_04.md       # Building a knowledge chatbot
├── examples/                # Runnable Python scripts
│   ├── 01_test_connection.py
│   ├── 02_translate_text.py
│   ├── 03_embeddings_intro.py
│   ├── 04_semantic_search.py
│   ├── 05_qualitative_coding.py
│   └── 06_knowledge_chatbot.py
├── src/                     # Reusable code modules
│   ├── client/             # LLM client setup
│   ├── embeddings/         # Embedding utilities
│   └── rag/                # RAG pipeline components
└── data/                    # Sample data for exercises
    ├── sample_interviews/
    └── project_lessons/
```

## Training sessions

### Morning
1. **Session 01** (1 hour): Local setup and your first LLM API call
2. **Session 02** (1 hour): Introduction to embeddings and RAG models

### Afternoon
3. **Session 03** (1.5 hours): Qualitative coding with embeddings
4. **Session 04** (1.5 hours): Building an internal knowledge chatbot

## Getting started

1. Clone this repository
2. Follow the setup instructions in `docs/session_01.md`
3. Complete sessions in order

Each session builds on the previous one.

## Requirements

- Python 3.11+
- OpenAI API key
- Code editor (VS Code or Positron recommended)
- Basic command line familiarity

## Support

- Session guides contain detailed instructions and troubleshooting
- Example scripts include inline comments
- Common issues documented in each session guide

## Learning outcomes

By the end of this training, you will:
- Have a working local LLM development environment
- Understand how to use LLM APIs programmatically
- Know what embeddings are and when to use them
- Understand the RAG architecture pattern
- Have built two practical applications using these concepts

---

**Start here:** `docs/session_02_embeddings_rag.md`