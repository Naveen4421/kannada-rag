import pytest

from retrieval.config import RetrievalConfig
from retrieval.pipeline.errors import StageError
from retrieval.retrieve.search import build_filter, hybrid_search, rrf_fuse
from tests.conftest import sparse


def hit(chunk_id, score, book="A"):
    return {"chunk_id": chunk_id, "score": score, "payload": {"book_id": book, "text": chunk_id, "page_start": 1}}


def test_rrf_scores_ranks_and_dedup():
    dense = [hit("x", 0.9), hit("y", 0.8), hit("x", 0.1)]  # duplicate within a channel
    sp = [hit("y", 5.0), hit("z", 4.0)]
    fused = rrf_fuse(dense, sp, k=60)

    by_id = {r.chunk_id: r for r in fused}
    assert len(fused) == 3
    assert by_id["x"].rrf_score == pytest.approx(1 / 61)
    assert by_id["y"].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert (by_id["y"].dense_rank, by_id["y"].sparse_rank) == (2, 1)
    assert by_id["z"].dense_rank is None and by_id["z"].dense_score is None
    assert [r.chunk_id for r in fused] == ["y", "x", "z"]


def test_rrf_deterministic_tie_break():
    fused = rrf_fuse([hit("b", 1.0)], [hit("a", 1.0)], k=60)
    assert [r.chunk_id for r in fused] == ["a", "b"]  # equal score/rank -> chunk_id


def test_filter_builder():
    assert build_filter(None) is None
    assert build_filter({"book_id": "A"}).must[0].match.value == "A"
    assert build_filter({"book_id": ["A", "B"]}).must[0].match.any == ["A", "B"]


def search(qdrant, **kw):
    kw.setdefault("question_embedding", [1, 0, 0])
    kw.setdefault("sparse_vector", sparse([0]))
    return hybrid_search("q", client=qdrant, collection_name="kannada_chunks", **kw)


def test_hybrid_returns_both_channels_and_metadata(qdrant):
    out = search(qdrant, top_k=3)
    top = out.results[0]
    assert top["chunk_id"] in ("A_0001", "B_0001")
    assert top["dense_rank"] is not None and top["sparse_rank"] is not None
    assert top["book_id"] and top["title"] and top["page"] == top["page_start"]
    assert top["score"] == top["rrf_score"]
    assert {"dense", "sparse", "rrf", "total"} <= set(out.timings_ms)
    assert out.counts["fused_unique"] == len(out.results) or out.counts["fused_unique"] >= len(out.results)


def test_book_filter_applies_to_both_channels(qdrant):
    out = search(qdrant, top_k=10, where={"book_id": "B"})
    assert {r["book_id"] for r in out.results} == {"B"}
    assert out.counts["dense"] == 1 and out.counts["sparse"] == 1


def test_no_matching_filter_is_empty_not_error(qdrant):
    out = search(qdrant, where={"book_id": "NOPE"})
    assert out.results == [] and out.warnings == []


def test_configurable_channel_limits(qdrant):
    out = search(qdrant, top_k=10, config=RetrievalConfig(dense_limit=1, sparse_limit=2, rrf_k=60))
    assert out.counts["dense"] == 1 and out.counts["sparse"] == 2


def test_one_channel_failure_degrades_with_warning(qdrant):
    out = search(qdrant, sparse_vector=object())  # no .indices -> sparse channel fails
    assert out.results and out.counts["sparse"] == 0
    assert any("sparse channel failed" in w for w in out.warnings)


def test_both_channels_failing_raises_structured_error(qdrant):
    with pytest.raises(StageError) as e:
        search(qdrant, question_embedding=[1, 0], sparse_vector=object())  # wrong dim + bad sparse
    assert e.value.to_dict()["code"] == "VECTOR_SEARCH_FAILED"
