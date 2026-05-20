"""In-memory vector store with cosine similarity search."""

from __future__ import annotations
import json
import logging
from pathlib import Path

import numpy as np

from src.embeddings import TFIDFEmbedder

logger = logging.getLogger("rag.vectorstore")


class VectorStore:
    def __init__(self, embedder: TFIDFEmbedder | None = None):
        self.embedder = embedder or TFIDFEmbedder()
        self.documents: list[dict] = []
        self.vectors: np.ndarray | None = None

    def index(self, chunks: list[dict]) -> int:
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.vectors = self.embedder.embed_batch(texts)
        self.documents = chunks
        logger.info("Indexed %d chunks (%d-dim vectors)", len(chunks), self.vectors.shape[1] if self.vectors.ndim > 1 else 0)
        return len(chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.vectors is None or len(self.documents) == 0:
            return []

        query_vec = self.embedder.embed(query)
        similarities = self.vectors @ query_vec

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append({
                    **self.documents[idx],
                    "score": round(float(similarities[idx]), 4),
                })
        return results

    def save(self, store_dir: str | Path) -> None:
        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        self.embedder.save(store_dir / "embedder.json")
        np.save(store_dir / "vectors.npy", self.vectors)
        (store_dir / "documents.json").write_text(json.dumps(self.documents))

    def load(self, store_dir: str | Path) -> None:
        store_dir = Path(store_dir)
        self.embedder.load(store_dir / "embedder.json")
        self.vectors = np.load(store_dir / "vectors.npy")
        self.documents = json.loads((store_dir / "documents.json").read_text())
