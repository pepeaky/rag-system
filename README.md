# Technical Docs RAG System

A Retrieval-Augmented Generation system that indexes technical documentation into a vector store, retrieves relevant chunks via cosine similarity, and generates answers using Claude's API with source citations.

---

## Architecture

```
  Markdown Documents
         │
         ▼
┌──────────────────┐
│     Chunker       │  Word-level splitting with overlap
│  src/chunker.py   │  Preserves source + position metadata
└────────┬─────────┘
         │ chunks[]
         ▼
┌──────────────────┐
│   TF-IDF Embedder │  Fit vocabulary → embed to dense vectors
│  src/embeddings.py │  (swap for OpenAI/Cohere in production)
└────────┬─────────┘
         │ vectors
         ▼
┌──────────────────┐
│   Vector Store    │  Cosine similarity search
│ src/vectorstore.py│  Persist: numpy + JSON
└────────┬─────────┘
         │ top-K chunks
         ▼
┌──────────────────┐
│    RAG Pipeline   │  Build prompt → Claude API → answer
│   src/rag.py      │  System prompt enforces context-only answers
└──────────────────┘
```

## Features

- **Document chunking** — word-level with configurable overlap for context continuity
- **TF-IDF embeddings** — zero-cost local embeddings (swappable for production embedding APIs)
- **Vector similarity search** — cosine similarity with top-K retrieval
- **Claude integration** — Claude Sonnet for answer generation with source citations
- **Persistent store** — numpy arrays + JSON for fast reload
- **Source tracking** — every answer includes ranked source documents with similarity scores

## Quick Start

```bash
git clone <repo-url> && cd 12-rag-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Anthropic API key

# index documents
python main.py index --docs docs

# search (vector similarity only, no LLM)
python main.py search "What is a Kubernetes pod?"

# ask (retrieval + Claude)
python main.py ask "How do Docker bridge networks work?"
```

## Testing

```bash
pytest -v
```

**15 tests** — chunking, TF-IDF embeddings, vector store indexing/search, prompt building. All tests run without an API key.

## Project Structure

```
├── main.py              # CLI: index, search, ask
├── docs/
│   ├── kubernetes_basics.md
│   └── docker_networking.md
├── src/
│   ├── config.py        # .env loader
│   ├── chunker.py       # Document splitting with overlap
│   ├── embeddings.py    # TF-IDF embedder (fit/embed/save/load)
│   ├── vectorstore.py   # Vector store (index/search/persist)
│   └── rag.py           # RAG pipeline (retrieve + prompt + Claude)
└── tests/
    └── test_rag.py      # 15 tests — full pipeline
```

## Design Decisions

- **TF-IDF over API embeddings** — demonstrates the vector search pattern without API costs or dependencies. In production, replace `TFIDFEmbedder` with Voyage/OpenAI embeddings for better semantic understanding.
- **Word-level chunking** — simpler and more predictable than sentence-level; overlap ensures no context is lost at boundaries.
- **Claude as generator** — system prompt restricts answers to provided context, reducing hallucination.
