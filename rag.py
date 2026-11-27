"""Step 6: Retrieval + answer generation with gpt-4o-mini (same chat model as the notebooks)."""
import re

from langchain_core.prompts import ChatPromptTemplate

from .config import CHAT_MODEL, DISCLAIMER, OPENAI_API_KEY, TOP_K
from .legal import rewrite_query
from .retrievers import build_retriever
from .vector_store import get_vector_store, load_chunks_from_store

ABSTAIN = "I could not find this in the indexed documents."

SYSTEM = f"""You are LexiAI, a careful assistant for Indian legal research.
Rules:
1. Answer ONLY from the numbered passages. Do not use outside knowledge.
2. Cite every claim with passage numbers like [1] or [2].
3. If the passages do not contain the answer, reply exactly: "{ABSTAIN}"
4. Quote section and article numbers exactly as written in the passages.
5. Give legal information, not legal advice."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Question: {question}\n\nPassages:\n{context}\n\nAnswer:")]
)


def get_llm():
    if not OPENAI_API_KEY:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0)


def format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        label = d.metadata.get("source", "unknown")
        if d.metadata.get("section"):
            label += f", Section {d.metadata['section']}"
        parts.append(f"[{i}] ({label})\n{d.page_content}")
    return "\n\n".join(parts)


def invalid_citations(answer: str, n_passages: int) -> list[int]:
    cited = {int(x) for x in re.findall(r"\[(\d+)\]", answer)}
    return sorted(c for c in cited if c < 1 or c > n_passages)


def ask(question, mode="hybrid", k=TOP_K, vs=None, embeddings=None, llm=None, use_llm=True) -> dict:
    rewritten, mapped = rewrite_query(question)
    vs = vs or get_vector_store(embeddings)
    chunks = load_chunks_from_store(vs)
    if not chunks:
        raise RuntimeError("The vector store is empty. Run: python -m lexiai.ingest")

    llm = llm if llm is not None else (get_llm() if use_llm else None)
    if embeddings is None:
        embeddings = vs.embeddings if hasattr(vs, "embeddings") else None
    retriever = build_retriever(mode, vs, chunks, k=k, embeddings=embeddings, llm=llm)
    docs = retriever.invoke(rewritten)[:k]

    result = {
        "question": question,
        "rewritten_query": rewritten,
        "ipc_to_bns": mapped,
        "mode": mode,
        "sources": [
            {"n": i, "source": d.metadata.get("source"), "section": d.metadata.get("section"),
             "chunk_id": d.metadata.get("chunk_id"), "text": d.page_content}
            for i, d in enumerate(docs, 1)
        ],
        "disclaimer": DISCLAIMER,
    }
    if not docs:
        result.update(answer=ABSTAIN, invalid_citations=[], grounded=False)
        return result
    if llm is None:  # no API key: return the passages themselves
        result.update(
            answer="(No OPENAI_API_KEY set, showing the best passages only.)\n\n" + format_context(docs),
            invalid_citations=[], grounded=False,
        )
        return result

    answer = (PROMPT | llm).invoke({"question": question, "context": format_context(docs)}).content
    bad = invalid_citations(answer, len(docs))
    result.update(answer=answer, invalid_citations=bad, grounded=(not bad) and ABSTAIN not in answer)
    return result
