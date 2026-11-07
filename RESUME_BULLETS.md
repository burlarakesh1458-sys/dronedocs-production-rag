# Resume bullets: DroneDocs production RAG (all numbers measured in this repo)

Sources: `reports/latest.json` (live golden eval, 60/60 items), `reports/offline_gate.json` (CI gate),
`reports/retrieval_ablation.json`.

- Built a cited question-answering system over 33 drone-regulation and autopilot documents (156 chunks):
  hybrid BM25 + Chroma retrieval with RRF, a cross-encoder evidence gate, and a LangGraph pipeline.
  Served via FastAPI, a CLI and Streamlit, packaged with Docker.
- Enforced grounding with an evidence gate and a citation validator: on a 60-question hand-verified golden set
  (gpt-oss-20b generator, Qwen judge) scored **0.935 faithfulness**, **1.00 citation validity**, 0.885 context
  recall, and declined **8/8 unanswerable questions** (4 by the evidence gate before any LLM call, 0 tokens).
- Wrote an evaluation harness (faithfulness, answer relevancy, context precision/recall, citation accuracy,
  refusal correctness, p50/p95 latency) and found the weak spot from per-item misses: 6 of 11 false declines
  were PX4 parameter-table questions (PX4 answered 7/13 vs. Drone Rules 20/23).
- Made the eval survive free-tier limits: a rate-limit-aware LLM client (honours `retry-after`, paces from
  `x-ratelimit-*` headers, sqlite response cache, per-item checkpoints) finished the 22 items a daily quota had
  blocked; replaced a CI gate that failed on HTTP 429s with a deterministic offline replay gate over all
  60 items that needs no API key.
- Ran a retrieval ablation on 52 questions: hybrid RRF lifted MRR@4 from 0.745 (vector-only) to 0.811.
  A general-domain cross-encoder lowered it to 0.728, so it was kept only for the evidence gate, after
  cutting its CPU latency from ~9.5 s to ~2.2 s per query.
