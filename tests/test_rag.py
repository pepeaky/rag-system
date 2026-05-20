import pytest
import numpy as np
from src.chunker import chunk_text, load_and_chunk
from src.embeddings import TFIDFEmbedder
from src.vectorstore import VectorStore
from src.rag import build_prompt


class TestChunker:
    def test_basic_chunking(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) >= 3

    def test_empty_text(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_small_text_single_chunk(self):
        chunks = chunk_text("hello world", chunk_size=100)
        assert len(chunks) == 1

    def test_overlap_creates_redundancy(self):
        text = " ".join([f"w{i}" for i in range(50)])
        chunks_no_overlap = chunk_text(text, chunk_size=20, overlap=0)
        chunks_with_overlap = chunk_text(text, chunk_size=20, overlap=5)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)

    def test_load_and_chunk_from_dir(self, tmp_path):
        (tmp_path / "test.md").write_text("This is a test document with some content for chunking.")
        chunks = load_and_chunk(tmp_path, chunk_size=5, overlap=1)
        assert len(chunks) >= 1
        assert chunks[0]["source"] == "test.md"
        assert "id" in chunks[0]


class TestEmbedder:
    def test_fit_and_embed(self):
        embedder = TFIDFEmbedder()
        embedder.fit(["hello world", "foo bar baz"])
        vec = embedder.embed("hello world")
        assert isinstance(vec, np.ndarray)
        assert len(vec) > 0

    def test_normalized_vectors(self):
        embedder = TFIDFEmbedder()
        embedder.fit(["kubernetes pods deployments", "docker networking bridge"])
        vec = embedder.embed("kubernetes pods")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01 or norm == 0

    def test_similar_texts_closer(self):
        embedder = TFIDFEmbedder()
        docs = ["kubernetes pod deployment", "docker container network", "kubernetes service ingress"]
        embedder.fit(docs)
        v1 = embedder.embed("kubernetes pod")
        v2 = embedder.embed("docker container")
        v3 = embedder.embed("kubernetes service")
        sim_k_k = float(v1 @ v3)
        sim_k_d = float(v1 @ v2)
        assert sim_k_k > sim_k_d

    def test_save_load(self, tmp_path):
        embedder = TFIDFEmbedder()
        embedder.fit(["test document"])
        embedder.save(tmp_path / "emb.json")
        loaded = TFIDFEmbedder()
        loaded.load(tmp_path / "emb.json")
        assert loaded.vocabulary == embedder.vocabulary


class TestVectorStore:
    def test_index_and_search(self):
        chunks = [
            {"id": "a", "source": "k8s.md", "chunk_index": 0, "text": "Kubernetes is a container orchestration platform for deploying applications"},
            {"id": "b", "source": "docker.md", "chunk_index": 0, "text": "Docker networking uses bridge networks for container communication"},
            {"id": "c", "source": "k8s.md", "chunk_index": 1, "text": "Pods are the smallest deployable units in Kubernetes clusters"},
        ]
        store = VectorStore()
        store.index(chunks)
        results = store.search("kubernetes pods", top_k=2)
        assert len(results) <= 2
        assert results[0]["score"] > 0

    def test_empty_store_returns_empty(self):
        store = VectorStore()
        assert store.search("anything") == []

    def test_save_load(self, tmp_path):
        chunks = [{"id": "a", "source": "test.md", "chunk_index": 0, "text": "test document content"}]
        store = VectorStore()
        store.index(chunks)
        store.save(tmp_path / "store")
        loaded = VectorStore()
        loaded.load(tmp_path / "store")
        results = loaded.search("test document")
        assert len(results) >= 1


class TestPromptBuilder:
    def test_builds_prompt_with_context(self):
        chunks = [{"source": "k8s.md", "text": "Kubernetes is great"}]
        prompt = build_prompt("What is K8s?", chunks)
        assert "Kubernetes is great" in prompt
        assert "What is K8s?" in prompt
        assert "[Source 1: k8s.md]" in prompt

    def test_empty_context(self):
        prompt = build_prompt("test?", [])
        assert "test?" in prompt
