"""Document chunker — splits markdown files into overlapping text chunks."""

from __future__ import annotations
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap if end < len(words) else len(words)

    return chunks


def load_and_chunk(docs_dir: str | Path, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    docs_path = Path(docs_dir)
    all_chunks = []

    for md_file in sorted(docs_path.glob("*.md")):
        text = md_file.read_text()
        chunks = chunk_text(text, chunk_size, overlap)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{md_file.stem}_{i:03d}",
                "source": md_file.name,
                "chunk_index": i,
                "text": chunk,
            })

    return all_chunks
