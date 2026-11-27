"""Step 5: Retrievers (as in RAG_Retrivers and RAG_Retrival_2).

similarity | threshold | mmr | bm25 | hybrid (ensemble) | multi_query | compression
"""
import re

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MultiQueryRetriever,
)
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.retrievers import BM25Retriever

MODES = ["similarity", "threshold", "mmr", "bm25", "hybrid", "multi_query", "compression"]


def _tokens(text: str) -> list[str]:
    # Keeps numbers like "318" searchable, which matters for legal section lookups.
    return re.findall(r"[a-z0-9]+", text.lower())


def similarity_retriever(vs, k=4):
    return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})


def threshold_retriever(vs, k=4, score_threshold=0.3):
    return vs.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )


def mmr_retriever(vs, k=4, fetch_k=10, lambda_mult=0.5):
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(fetch_k, k), "lambda_mult": lambda_mult},
    )


def bm25_retriever(chunks, k=4):
    return BM25Retriever.from_documents(documents=chunks, k=k, preprocess_func=_tokens)


def hybrid_retriever(vs, chunks, k=4, weights=(0.6, 0.4)):
    """Dense (MMR) + BM25 ensemble with the same 0.6 / 0.4 weights as the notebook."""
    return EnsembleRetriever(
        retrievers=[mmr_retriever(vs, k=k), bm25_retriever(chunks, k=k)],
        weights=list(weights),
    )


def multi_query_retriever(base, llm):
    return MultiQueryRetriever.from_llm(llm=llm, retriever=base)


def compression_retriever(base, embeddings, similarity_threshold=0.3):
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=similarity_threshold)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)


def build_retriever(mode, vs, chunks, k=4, embeddings=None, llm=None):
    if mode == "similarity":
        return similarity_retriever(vs, k)
    if mode == "threshold":
        return threshold_retriever(vs, k)
    if mode == "mmr":
        return mmr_retriever(vs, k)
    if mode == "bm25":
        return bm25_retriever(chunks, k)
    if mode == "hybrid":
        return hybrid_retriever(vs, chunks, k)
    if mode == "multi_query":
        if llm is None:
            raise ValueError("multi_query needs an LLM. Set OPENAI_API_KEY in .env.")
        return multi_query_retriever(hybrid_retriever(vs, chunks, k), llm)
    if mode == "compression":
        if embeddings is None:
            raise ValueError("compression needs the embeddings object.")
        return compression_retriever(hybrid_retriever(vs, chunks, k), embeddings)
    raise ValueError(f"Unknown mode '{mode}'. Choose from {MODES}")
