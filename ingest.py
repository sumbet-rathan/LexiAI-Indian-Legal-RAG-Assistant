
"""Run: python -m lexiai.ingest (load -> chunk -> embed -> store)"""

import argparse

from .chunking import split_documents
from .config import DATA_DIR
from .embeddings import get_embeddings
from .loaders import load_documents
from .vector_store import add_chunks, get_vector_store, reset_store


def run(path=DATA_DIR, reset=True, embedding=None, persist_directory=None) -> int:

    docs = load_documents(path)
    print(f"Loaded {len(docs)} document(s)/page(s)")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunk(s)")

    if not chunks:
        raise ValueError("No chunks generated. Check your documents.")

    if reset:
        reset_store(persist_directory)

    vs = get_vector_store(
        embedding or get_embeddings(),
        persist_directory
    )

    # Store chunks in smaller batches
    BATCH_SIZE = 8
    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        add_chunks(vs, batch)

        processed = min(start + BATCH_SIZE, total_chunks)

        print(f"Stored {processed}/{total_chunks} chunks")

    print("Successfully stored all chunks in ChromaDB.")

    return len(chunks)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--path", default=str(DATA_DIR))

    ap.add_argument(
        "--keep",
        action="store_true",
        help="Do not wipe the existing store first"
    )

    a = ap.parse_args()

    run(a.path, reset=not a.keep)
