import itertools

from dronedocs.chunking import chunk_sections, get_encoder, make_locator
from dronedocs.loaders import Section


def _sections(n_pages=6, words=400):
    return [
        Section(
            "rules.pdf",
            "pdf",
            f"page {p}",
            "\n\n".join(f"Rule {p}.{i} " + " ".join(f"word{p}_{i}_{k}" for k in range(40)) for i in range(words // 40)),
            page=p,
        )
        for p in range(1, n_pages + 1)
    ]


def test_chunks_respect_token_budget_and_overlap():
    chunks = chunk_sections(_sections(), chunk_tokens=550, overlap_tokens=100)
    assert len(chunks) > 3
    assert all(c.n_tokens <= 550 for c in chunks)
    enc = get_encoder()
    # consecutive chunks share text (the overlap) -> last tokens of chunk i appear in chunk i+1
    for a, b in itertools.pairwise(chunks):
        tail = enc.decode(enc.encode(a.text)[-20:]).strip().split()[-3:]
        assert all(t in b.text for t in tail)


def test_chunks_carry_page_provenance():
    chunks = chunk_sections(_sections(), chunk_tokens=550, overlap_tokens=100)
    assert chunks[0].pages[0] == 1
    assert chunks[-1].pages[-1] == 6
    assert all(c.locator.startswith("page") for c in chunks)
    assert chunks[0].chunk_id == "rules.pdf#c000"
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_overlap_must_be_smaller_than_chunk():
    import pytest

    with pytest.raises(ValueError):
        chunk_sections(_sections(), chunk_tokens=100, overlap_tokens=100)


def test_locator_formats():
    assert make_locator([3], []) == "page 3"
    assert make_locator([3, 4], []) == "pages 3-4"
    assert make_locator([], ["A > B", "A > C"]) == "section: A > B (+1 more)"
    assert make_locator([], []) == "document"


def test_short_document_yields_single_chunk():
    chunks = chunk_sections([Section("a.md", "markdown", "section: A", "tiny text", section="A")])
    assert len(chunks) == 1 and chunks[0].sections == ["A"]
