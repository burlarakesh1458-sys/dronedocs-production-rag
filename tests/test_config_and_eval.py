from dronedocs.chunking import Chunk
from dronedocs.config import apply_env_overrides, load_prompt_bundle
from dronedocs.eval.metrics import average_precision, chunk_is_relevant, reciprocal_rank
from dronedocs.eval.run_eval import check_thresholds, is_quota_error, load_golden, summarize


def test_env_overrides_nested_keys():
    cfg = apply_env_overrides({"rerank": {"top_n": 5}}, {"DRONEDOCS__RERANK__TOP_N": "8", "OTHER": "x"})
    assert cfg["rerank"]["top_n"] == 8


def test_prompt_bundles_are_versioned():
    assert load_prompt_bundle("configs/prompts/answer_v1.yaml")["version"].startswith("answer-v")
    assert load_prompt_bundle("configs/prompts/judge_v1.yaml")["version"].startswith("judge-v")


def test_average_precision_and_mrr():
    assert average_precision([True, False, True]) == (1 + 2 / 3) / 2
    assert average_precision([False, False]) == 0.0
    assert reciprocal_rank([False, True]) == 0.5


def test_relevance_by_evidence_or_page():
    item = {"answerable": True, "source": "r.pdf", "locator": "page 3", "evidence": "ten years"}
    c1 = Chunk("r.pdf#c0", "r.pdf", "pdf", "page 9", "valid for ten  years", 5, [9], [])
    c2 = Chunk("r.pdf#c1", "r.pdf", "pdf", "page 3", "other text", 5, [3], [])
    c3 = Chunk("x.pdf#c0", "x.pdf", "pdf", "page 3", "ten years", 5, [3], [])
    assert chunk_is_relevant(c1, item) and chunk_is_relevant(c2, item) and not chunk_is_relevant(c3, item)


def test_golden_set_is_large_enough_and_balanced():
    rows = load_golden("eval/golden.jsonl")
    assert len(rows) >= 50
    assert sum(not r["answerable"] for r in rows) >= 5
    assert all(r["evidence"] for r in rows if r["answerable"])
    small = load_golden("eval/golden.jsonl", limit=10)
    assert len(small) == 10 and any(not r["answerable"] for r in small)


def test_summary_and_threshold_gate():
    base = {"latency_s": {"total": 1.0, "generate": 0.8}, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    rows = [
        {
            **base,
            "answerable": True,
            "refused": False,
            "refusal_correct": True,
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_recall": 1.0,
            "context_precision": 1.0,
            "citation_accuracy": 1.0,
            "citation_validity": 1.0,
            "fused_recall": 1.0,
            "mrr": 1.0,
        },
        {
            **base,
            "answerable": True,
            "refused": True,
            "refusal_correct": False,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "context_precision": 0.0,
            "fused_recall": 0.0,
            "mrr": 0.0,
        },
        {**base, "answerable": False, "refused": True, "refusal_correct": True},
    ]
    s = summarize(rows)
    assert s["faithfulness"] == 1.0 and s["answer_relevancy"] == 0.5
    assert abs(s["refusal_correctness"] - 2 / 3) < 1e-3
    fails = check_thresholds(s, {"faithfulness": 0.8, "refusal_correctness": 0.9})
    assert fails == [f"refusal_correctness={s['refusal_correctness']} < 0.9"]


def test_provider_quota_rows_are_not_scored_as_refusals():
    base = {"latency_s": {"total": 1.0}, "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    ok = {**base, "id": "a", "answerable": False, "refused": True, "refusal_correct": True, "error": None}
    quota = {
        **base,
        "id": "b",
        "answerable": True,
        "refused": True,
        "refusal_correct": False,
        "error": "LLMError: quota exhausted (retry-after 900s): HTTP 429",
    }
    other = {**ok, "id": "c", "error": "LLMError: HTTP 500"}
    assert is_quota_error(quota) and not is_quota_error(ok) and not is_quota_error(other)
    s = summarize([ok, quota])
    assert s["n_items"] == 1 and s["n_not_evaluated_quota"] == 1 and s["not_evaluated_ids"] == ["b"]
    assert s["refusal_correctness"] == 1.0
    assert summarize([quota])["n_items"] == 0


def test_alternate_evidence_counts_as_relevant():
    item = {
        "answerable": True,
        "source": "r.pdf",
        "locator": "page 3",
        "evidence": "ten years",
        "alt_evidence": [{"source": "pib.pdf", "locator": "page 1", "evidence": "valid for 10 years"}],
    }
    alt = Chunk("pib.pdf#c0", "pib.pdf", "pdf", "page 1", "it is valid for 10 years", 5, [1], [])
    assert chunk_is_relevant(alt, item)
