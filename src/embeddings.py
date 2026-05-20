"""Simple TF-IDF based vector embeddings (no external embedding API dependency).

For production, swap this with OpenAI/Cohere/Voyage embeddings.
This implementation demonstrates the vector search pattern without API costs.
"""

from __future__ import annotations
import json
import math
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger("rag.embeddings")


class TFIDFEmbedder:
    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self._fitted = False

    def fit(self, documents: list[str]) -> None:
        doc_freq = Counter()
        all_terms = set()

        for doc in documents:
            terms = set(_tokenize(doc))
            for term in terms:
                doc_freq[term] += 1
            all_terms.update(terms)

        n_docs = len(documents)
        self.vocabulary = {term: idx for idx, term in enumerate(sorted(all_terms))}
        self.idf = {term: math.log(n_docs / (1 + freq)) for term, freq in doc_freq.items()}
        self._fitted = True

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocabulary))
        tokens = _tokenize(text)
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1

        for term, count in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                vec[idx] = (count / total) * self.idf.get(term, 0)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.array([self.embed(t) for t in texts])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"vocabulary": self.vocabulary, "idf": self.idf}
        path.write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.vocabulary = data["vocabulary"]
        self.idf = data["idf"]
        self._fitted = True


def _tokenize(text: str) -> list[str]:
    return [w.lower().strip(".,;:!?\"'()[]{}") for w in text.split() if len(w) > 2]
