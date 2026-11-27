
"""Central settings. Values come from environment variables or a .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

# ChromaDB absolute storage path
_raw_chroma_dir = Path(os.getenv("CHROMA_DIR", "chroma_db"))

PERSIST_DIR = (
    _raw_chroma_dir
    if _raw_chroma_dir.is_absolute()
    else ROOT / _raw_chroma_dir
).resolve()

COLLECTION = os.getenv("COLLECTION_NAME", "lexiai")

# Models
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
SEPARATORS = ["\n\n", "\n", " ", ""]

TOP_K = int(os.getenv("TOP_K", "4"))

DISCLAIMER = (
    "This is legal information for research, "
    "not legal advice. Verify against the official text."
)
