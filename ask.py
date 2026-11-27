"""Run: python -m lexiai.ask "your question" --mode hybrid"""
import argparse

from .config import TOP_K
from .rag import ask
from .retrievers import MODES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="hybrid", choices=MODES)
    ap.add_argument("--k", type=int, default=TOP_K)
    a = ap.parse_args()
    r = ask(a.question, mode=a.mode, k=a.k)
    if r["ipc_to_bns"]:
        print("Rewritten:", r["rewritten_query"])
    print("\n" + r["answer"] + "\n")
    for s in r["sources"]:
        print(f"[{s['n']}] {s['source']} Section {s['section']}  ({s['chunk_id']})")
    if r["invalid_citations"]:
        print("WARNING invalid citations:", r["invalid_citations"])
    print("\n" + r["disclaimer"])
