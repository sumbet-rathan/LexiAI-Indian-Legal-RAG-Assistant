"""Legal helper: rewrite old IPC section numbers to the new BNS numbers.

STARTER TABLE ONLY. Verify every entry against the official IPC-to-BNS
correspondence table before real use. A wrong mapping returns the wrong law.
"""
import re

IPC_TO_BNS = {
    "34": "3", "120B": "61", "302": "103", "304B": "80", "307": "109",
    "323": "115", "354": "74", "376": "64", "379": "303", "406": "316",
    "420": "318", "498A": "85",
}

_PATTERNS = [
    re.compile(r"\b(?:section|sec\.?|s\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+(?:the\s+)?)?(?:IPC|Indian Penal Code)\b", re.I),
    re.compile(r"\b(?:IPC|Indian Penal Code)\s*(?:section|sec\.?|s\.?)?\s*(\d+[A-Za-z]?)\b", re.I),
]


def rewrite_query(query: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (rewritten_query, [(ipc, bns), ...]). Adds BNS wording to the query."""
    found: list[tuple[str, str]] = []
    for pat in _PATTERNS:
        for m in pat.finditer(query):
            ipc = m.group(1).upper()
            bns = IPC_TO_BNS.get(ipc)
            if bns and (ipc, bns) not in found:
                found.append((ipc, bns))
    if not found:
        return query, []
    extra = " ".join(f"Bharatiya Nyaya Sanhita BNS Section {b}" for _, b in found)
    return f"{query} {extra}", found
