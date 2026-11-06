"""Minimal Streamlit front-end: question in, cited answer out, with the retrieval trace."""

import os

import httpx
import streamlit as st

API = os.environ.get("DRONEDOCS_API", "http://localhost:8000")

st.set_page_config(page_title="DroneDocs RAG", page_icon="🛩️")
st.title("DroneDocs RAG")
st.caption("India's Drone Rules 2021 (+ amendments) and ArduPilot / PX4 docs. Every answer is cited, or refused.")

q = st.text_input("Question", placeholder="What is the default RTL altitude in ArduPilot Copter?")
if st.button("Ask", type="primary") and q:
    with st.spinner("Retrieving, reranking, generating..."):
        r = httpx.post(f"{API}/ask", json={"question": q, "include_debug": True}, timeout=120)
    if r.status_code != 200:
        st.error(r.text)
    else:
        d = r.json()
        (st.warning if d["refused"] else st.success)(d["answer"])
        for c in d["citations"]:
            with st.expander(f"[{c['chunk_id']}] {c['source']} - {c['locator']}"):
                st.write(c["snippet"] + "...")
        st.caption(
            f"prompt {d['prompt_version']} - {d['model']} - "
            + ", ".join(f"{k} {v:.0f} ms" for k, v in d["latency_ms"].items())
        )
        with st.expander("Rerank order (debug)"):
            st.dataframe(d["debug"]["reranked"])
