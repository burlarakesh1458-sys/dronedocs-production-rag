"""Retrieval ablation on the golden set - no LLM calls.

Compares vector-only, BM25-only, hybrid (RRF) and hybrid + cross-encoder rerank on:
  hit@k (a gold-evidence chunk is among the k=rerank.top_n chunks that go into the prompt), MRR@k,
  recall@20 (candidate pool) and per-stage latency.
Writes reports/retrieval_ablation.json.
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dronedocs.config import get_settings  # noqa: E402
from dronedocs.eval.metrics import chunk_is_relevant, reciprocal_rank  # noqa: E402
from dronedocs.eval.run_eval import load_golden  # noqa: E402
from dronedocs.rerank import CrossEncoderReranker  # noqa: E402
from dronedocs.retrieval import Embedder, HybridIndex  # noqa: E402


def main() -> None:
    cfg = get_settings()
    r = cfg["retrieval"]
    top_n = cfg["rerank"]["top_n"]
    idx = HybridIndex(
        cfg["paths"]["index_dir"], Embedder(cfg["embeddings"]["model"], cfg["embeddings"]["query_instruction"])
    ).load()
    rr = cfg["rerank"]
    ce = CrossEncoderReranker(rr["model"], rr.get("max_length", 256), rr.get("candidates"))
    items = [it for it in load_golden(cfg["eval"]["golden_path"]) if it["answerable"]]
    configs = {"vector": "vector", "bm25": "bm25", "hybrid_rrf": "hybrid", "hybrid_rrf+rerank": "hybrid"}
    results = {}
    for name, mode in configs.items():
        hit5, mrr, rec20, lat = [], [], [], []
        for it in items:
            t0 = time.perf_counter()
            hits, _ = idx.hybrid_search(it["question"], r["vector_k"], r["bm25_k"], r["rrf_k"], r["fused_k"], mode)
            top = ce.rerank(it["question"], hits, top_n) if name.endswith("rerank") else hits[:top_n]
            lat.append(time.perf_counter() - t0)
            rels = [chunk_is_relevant(idx.chunks[h.chunk_id], it) for h in top]
            hit5.append(float(any(rels)))
            mrr.append(reciprocal_rank(rels))
            rec20.append(float(any(chunk_is_relevant(idx.chunks[h.chunk_id], it) for h in hits[:20])))
        results[name] = {
            "hit@k": round(statistics.mean(hit5), 4),
            "mrr@k": round(statistics.mean(mrr), 4),
            "recall@20": round(statistics.mean(rec20), 4),
            "p50_ms": round(1000 * statistics.median(lat), 1),
            "p95_ms": round(1000 * sorted(lat)[int(0.95 * (len(lat) - 1))], 1),
        }
        print(f"{name:<20} {results[name]}", flush=True)
    out = {
        "n_questions": len(items),
        "k": top_n,
        "rerank_candidates": rr.get("candidates"),
        "rerank_max_length": rr.get("max_length"),
        "chunks_in_index": len(idx.chunks),
        "results": results,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "retrieval_ablation.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
