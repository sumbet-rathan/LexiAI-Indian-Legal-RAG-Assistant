"""Offline tests: no Ollama, no OpenAI key needed (uses a fake embedder)."""
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from lexiai.chunking import split_documents
from lexiai.ingest import run as ingest_run
from lexiai.legal import rewrite_query
from lexiai.loaders import load_documents
from lexiai.rag import ABSTAIN, ask, invalid_citations
from lexiai.retrievers import MODES, bm25_retriever
from lexiai.vector_store import (add_chunks, delete_chunks, get_by_ids, get_vector_store,
                                 load_chunks_from_store, update_chunk)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
EMB = DeterministicFakeEmbedding(size=64)


@pytest.fixture()
def vs(tmp_path):
    ingest_run(RAW, embedding=EMB, persist_directory=tmp_path / "db")
    return get_vector_store(EMB, tmp_path / "db")


def test_loader_reads_txt():
    docs = load_documents(RAW)
    assert docs and docs[0].metadata["source"] == "demo_bns.txt"


def test_chunker_adds_ids_and_sections():
    chunks = split_documents(load_documents(RAW))
    assert all("chunk_id" in c.metadata for c in chunks)
    assert any(c.metadata.get("section") == "318" for c in chunks)


def test_rewrite_ipc_to_bns():
    q, mapped = rewrite_query("What is the punishment for cheating under IPC 420?")
    assert ("420", "318") in mapped and "Section 318" in q
    assert rewrite_query("Section 498A of IPC")[1] == [("498A", "85")]
    assert rewrite_query("plain question")[1] == []


def test_bm25_finds_section_number():
    chunks = split_documents(load_documents(RAW))
    top = bm25_retriever(chunks, k=2).invoke("Section 318 cheating")[0]
    assert top.metadata.get("section") == "318"


def test_crud(vs):
    ids = [c.metadata["chunk_id"] for c in load_chunks_from_store(vs)][:1]
    assert get_by_ids(vs, ids)
    update_chunk(vs, ids[0], Document(page_content="updated text", metadata={"source": "x", "chunk_id": ids[0]}))
    assert get_by_ids(vs, ids)[0].page_content == "updated text"
    delete_chunks(vs, ids)
    assert not get_by_ids(vs, ids)


@pytest.mark.parametrize("mode", [m for m in MODES if m not in ("multi_query",)])
def test_every_mode_runs(vs, mode):
    r = ask("cheating Section 318", mode=mode, k=3, vs=vs, embeddings=EMB, use_llm=False)
    assert "answer" in r and isinstance(r["sources"], list)


def test_hybrid_returns_sources_without_llm(vs):
    r = ask("What is the punishment for cheating under IPC 420?", vs=vs, embeddings=EMB, use_llm=False)
    assert r["ipc_to_bns"] and any(s["section"] == "318" for s in r["sources"])
    assert r["grounded"] is False


def test_multi_query_needs_llm(vs):
    with pytest.raises(ValueError):
        ask("anything", mode="multi_query", vs=vs, embeddings=EMB, use_llm=False)


def test_citation_check():
    assert invalid_citations("Cheating is punished [1] and [5].", 3) == [5]
    assert ABSTAIN.startswith("I could not")
