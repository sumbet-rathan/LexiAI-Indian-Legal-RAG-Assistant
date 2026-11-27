"""Run: streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from lexiai.rag import ask
from lexiai.retrievers import MODES

st.set_page_config(page_title="LexiAI", page_icon="⚖️")
st.title("⚖️ LexiAI")
st.caption("Legal RAG: Ollama qwen3-embedding + Chroma + gpt-4o-mini")

mode = st.sidebar.selectbox("Retrieval mode", MODES, index=MODES.index("hybrid"))
k = st.sidebar.slider("Passages", 1, 10, 4)
q = st.text_input("Ask a legal question", "What is the punishment for cheating under IPC 420?")

if st.button("Ask") and q.strip():
    try:
        r = ask(q, mode=mode, k=k)
    except Exception as e:  # show a friendly error
        st.error(str(e))
        st.stop()
    if r["ipc_to_bns"]:
        st.info(f"Rewritten for BNS: {r['rewritten_query']}")
    st.markdown(r["answer"])
    if r["invalid_citations"]:
        st.warning(f"Invalid citations flagged: {r['invalid_citations']}")
    for s in r["sources"]:
        with st.expander(f"[{s['n']}] {s['source']} Section {s['section']}"):
            st.write(s["text"])
    st.caption(r["disclaimer"])
