"""Shared offline fakes: no network, no model downloads."""

import pytest

from dronedocs.chunking import Chunk
from dronedocs.config import load_settings
from dronedocs.llm import LLMResponse
from dronedocs.retrieval import Hit


def make_chunk(cid: str, text: str, source: str = "doc.pdf", pages=(1,)) -> Chunk:
    return Chunk(
        chunk_id=cid,
        source=source,
        doc_type="pdf",
        locator=f"page {pages[0]}",
        text=text,
        n_tokens=len(text.split()),
        pages=list(pages),
        sections=[],
    )


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = {c.chunk_id: c for c in chunks}

    def hybrid_search(self, query, vector_k, bm25_k, rrf_k, fused_k, mode="hybrid"):
        hits = [Hit(cid, 1.0 / (i + 1), i + 1, mode, c.text) for i, (cid, c) in enumerate(self.chunks.items())]
        return hits[:fused_k], {"vector": [], "bm25": []}


class FakeReranker:
    def __init__(self, score=5.0):
        self.score = score

    def rerank(self, query, hits, top_n):
        return [
            Hit(h.chunk_id, self.score - i, i + 1, "fake", h.text, {"pre_rank": h.rank})
            for i, h in enumerate(hits[:top_n])
        ]


class FakeLLM:
    model = "fake-model"

    def __init__(self, text):
        self.text = text
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        return LLMResponse(text=self.text, model=self.model, prompt_tokens=100, completion_tokens=20, latency_s=0.01)


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def chunks():
    return [
        make_chunk(
            "drone_rules_2021.pdf#c000",
            "Nano unmanned aircraft system: weighing less than or equal to 250 grams;",
            "drone_rules_2021.pdf",
            (27,),
        ),
        make_chunk(
            "drone_rules_2021.pdf#c001",
            "Micro unmanned aircraft system: more than 250 grams, up to 2 kilograms.",
            "drone_rules_2021.pdf",
            (27,),
        ),
        make_chunk(
            "px4.md#c000", "COM_DISARM_LAND Time-out for auto disarm after landing. Default: 2s", "px4.md", (1,)
        ),
    ]


@pytest.fixture
def make_pipeline(settings, chunks):
    from dronedocs.pipeline import RAGPipeline

    def _make(answer: str, rerank_score: float = 5.0, observers=None):
        return RAGPipeline(
            settings=settings,
            index=FakeIndex(chunks),
            reranker=FakeReranker(rerank_score),
            llm=FakeLLM(answer),
            observers=observers,
        )

    return _make
