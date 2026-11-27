"""Step 3: Embeddings. Same local model as the notebooks: qwen3-embedding:0.6b via Ollama."""
from langchain_ollama import OllamaEmbeddings

from .config import EMBED_MODEL, OLLAMA_URL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
