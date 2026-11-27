
"""Step 4: ChromaDB vector store and CRUD operations."""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import COLLECTION, PERSIST_DIR


def get_vector_store(
    embedding=None,
    persist_directory=None,
    collection_name=None
) -> Chroma:

    if embedding is None:
        from .embeddings import get_embeddings
        embedding = get_embeddings()

    # Always use an absolute path
    db_path = Path(persist_directory or PERSIST_DIR).resolve()

    db_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=collection_name or COLLECTION,
        embedding_function=embedding,
        persist_directory=str(db_path),
        collection_metadata={"hnsw:space": "cosine"},
    )


# Database information
def get_store_count(vs: Chroma) -> int:
    return vs._collection.count()


def show_store_info(vs: Chroma) -> None:
    print("Collection:", COLLECTION)
    print("Database count:", get_store_count(vs))


# Reset database
def reset_store(persist_directory=None) -> None:

    path = Path(persist_directory or PERSIST_DIR).resolve()

    if path.exists():
        shutil.rmtree(path)

    print(f"Database reset: {path}")


# CREATE
def add_chunks(
    vs: Chroma,
    chunks: list[Document]
) -> list[str]:

    if not chunks:
        raise ValueError("No chunks provided for storage.")

    ids = []

    for chunk in chunks:
        chunk_id = chunk.metadata.get("chunk_id")

        if not chunk_id:
            raise ValueError("A chunk is missing its chunk_id.")

        ids.append(str(chunk_id))

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return vs.add_documents(chunks, ids=ids)


# READ
def load_chunks_from_store(vs: Chroma) -> list[Document]:

    data = vs.get(include=["documents", "metadatas"])

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    return [
        Document(
            page_content=text,
            metadata=metadata or {}
        )
        for text, metadata in zip(documents, metadatas)
        if text is not None
    ]


def get_by_ids(
    vs: Chroma,
    ids: list[str]
) -> list[Document]:

    return vs.get_by_ids(ids)


# UPDATE
def update_chunk(
    vs: Chroma,
    chunk_id: str,
    document: Document
) -> None:

    vs.update_document(
        document_id=chunk_id,
        document=document
    )


# DELETE
def delete_chunks(
    vs: Chroma,
    ids: list[str]
) -> None:

    if ids:
        vs.delete(ids=ids)
