"""Step 2: Chunking with RecursiveCharacterTextSplitter (as in RAG_Chuncking)."""
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS

_SECTION = re.compile(r"(?:Section|Sec\.|Article)\s+(\d+[A-Z]?)", re.I)


def split_documents(docs: list[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(docs)
    counters: dict[str, int] = {}
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        n = counters.get(src, 0)
        counters[src] = n + 1
        c.metadata["chunk_id"] = f"{src}::{n}"
        m = _SECTION.search(c.page_content)
        if m:
            c.metadata["section"] = m.group(1)  # helps citations
    return chunks
