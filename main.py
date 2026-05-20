"""CLI for the Technical Docs RAG System."""

import argparse
import json
import logging

from src.config import get_config
from src.chunker import load_and_chunk
from src.vectorstore import VectorStore
from src.rag import query_rag


def cmd_index(args):
    cfg = get_config()
    chunks = load_and_chunk(args.docs or cfg["docs_dir"], cfg["chunk_size"], cfg["chunk_overlap"])
    store = VectorStore()
    count = store.index(chunks)
    store.save(args.store or cfg["vectorstore_dir"])
    print(f"Indexed {count} chunks from {len(set(c['source'] for c in chunks))} documents")


def cmd_search(args):
    cfg = get_config()
    store = VectorStore()
    store.load(args.store or cfg["vectorstore_dir"])
    results = store.search(args.query, top_k=args.top_k or cfg["top_k"])
    for r in results:
        print(f"  [{r['score']:.4f}] {r['source']} (chunk {r['chunk_index']})")
        print(f"          {r['text'][:120]}...")
        print()


def cmd_ask(args):
    cfg = get_config()
    store = VectorStore()
    store.load(args.store or cfg["vectorstore_dir"])
    result = query_rag(args.query, store, top_k=args.top_k or cfg["top_k"])
    print(f"\n{result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  [{s['score']:.4f}] {s['source']}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Technical Docs RAG System")
    sub = parser.add_subparsers(dest="command", required=True)

    p_idx = sub.add_parser("index", help="Index documents into vector store")
    p_idx.add_argument("--docs", help="Documents directory")
    p_idx.add_argument("--store", help="Vector store directory")

    p_search = sub.add_parser("search", help="Search without LLM (vector similarity only)")
    p_search.add_argument("query")
    p_search.add_argument("--store", help="Vector store directory")
    p_search.add_argument("--top-k", type=int)

    p_ask = sub.add_parser("ask", help="Ask a question (retrieval + LLM)")
    p_ask.add_argument("query")
    p_ask.add_argument("--store", help="Vector store directory")
    p_ask.add_argument("--top-k", type=int)

    args = parser.parse_args()
    {"index": cmd_index, "search": cmd_search, "ask": cmd_ask}[args.command](args)


if __name__ == "__main__":
    main()
