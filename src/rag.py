"""RAG pipeline — retrieval + prompt construction + LLM generation."""

from __future__ import annotations
import logging

import anthropic

from src.config import get_config
from src.vectorstore import VectorStore

logger = logging.getLogger("rag")

SYSTEM_PROMPT = """You are a technical documentation assistant. Answer questions based ONLY on the provided context. If the context doesn't contain enough information to answer, say so. Be precise and cite the source document when possible."""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {source}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Context:
{context}

---

Question: {query}

Answer based on the context above:"""


def query_rag(query: str, store: VectorStore, top_k: int = 5, api_key: str | None = None, model: str | None = None) -> dict:
    cfg = get_config()
    api_key = api_key or cfg["api_key"]
    model = model or cfg["model"]

    results = store.search(query, top_k=top_k)
    if not results:
        return {"answer": "No relevant documents found.", "sources": [], "query": query}

    prompt = build_prompt(query, results)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = message.content[0].text

    sources = [{"source": r["source"], "chunk_id": r["id"], "score": r["score"]} for r in results]

    return {"answer": answer, "sources": sources, "query": query, "model": model}
