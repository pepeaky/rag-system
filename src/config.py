import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def get_config() -> dict:
    return {
        "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "model": os.getenv("MODEL_NAME", "claude-sonnet-4-6"),
        "vectorstore_dir": os.getenv("VECTORSTORE_DIR", "vectorstore"),
        "docs_dir": os.getenv("DOCS_DIR", "docs"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "500")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "50")),
        "top_k": int(os.getenv("TOP_K", "5")),
        "log_level": os.getenv("LOG_LEVEL", "INFO"),
    }
