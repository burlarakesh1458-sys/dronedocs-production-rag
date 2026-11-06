import pytest

from dronedocs.retrieval import bm25_tokenize, reciprocal_rank_fusion


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_between_retrievers():
    order = [cid for cid, _ in reciprocal_rank_fusion([["x", "shared", "y"], ["z", "shared", "w"]])]
    assert order[0] == "shared"


def test_rrf_is_deterministic_on_ties_and_handles_empty():
    assert [c for c, _ in reciprocal_rank_fusion([["b"], ["a"]])] == ["a", "b"]
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_weights():
    order = [c for c, _ in reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])]
    assert order == ["b", "a"]


def test_bm25_tokenizer_keeps_parameter_names():
    toks = bm25_tokenize("What does RTL_ALT_M default to? 2.5 m")
    assert "rtl_alt_m" in toks and "2.5" in toks and "what" not in toks
