# Resume bullets: DroneDocs production RAG (all numbers measured in this repo)

- Built a cited question-answering system over 33 drone-regulation and autopilot documents (156 chunks):
  hybrid BM25 + Chroma retrieval with RRF, a cross-encoder evidence gate, and a LangGraph pipeline.
  Served via FastAPI, a CLI and Streamlit.
- Enforced grounding with an evidence gate and a citation validator. Scored 0.94 faithfulness and 1.00
  citation validity on the scored golden items, and declined 4/4 unanswerable questions before any LLM call
  (0 tokens spent).
- Curated a 60-question hand-verified golden set and an offline eval (faithfulness, answer relevancy,
  context precision/recall, citation accuracy, refusal correctness, p50/p95 latency), wired into GitHub
  Actions as a quality gate that fails the build below thresholds.
- Ran a retrieval ablation on 52 questions: hybrid RRF lifted MRR@4 from 0.745 (vector-only) to 0.811.
  A general-domain cross-encoder lowered it to 0.728, so it was kept only for the evidence gate, after
  cutting its CPU latency from ~9.5 s to ~2.2 s per query.
