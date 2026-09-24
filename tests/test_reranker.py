from retrieval.retrieve.reranker import normalize_score, rerank, rerank_with_info


def cands():
    return [
        {"chunk_id": "a", "text": "aa", "rrf_score": 0.03},
        {"chunk_id": "b", "text": "bb", "rrf_score": 0.02},
        {"chunk_id": "c", "text": "cc", "rrf_score": 0.01},
    ]


def test_orders_by_score_keeps_metadata_and_does_not_mutate():
    original = cands()
    out = rerank("q", original, top_n=2, scorer=lambda pairs: [0.2, 0.9, 0.5])
    assert [c["chunk_id"] for c in out] == ["b", "c"]
    assert out[0]["rrf_score"] == 0.02 and out[0]["rerank_score"] == 0.9
    assert "rerank_score" not in original[0]


def test_ties_are_deterministic():
    out = rerank("q", cands(), top_n=3, scorer=lambda pairs: [0.5, 0.5, 0.5])
    assert [c["chunk_id"] for c in out] == ["a", "b", "c"]  # then RRF desc


def test_empty_candidates():
    out, info = rerank_with_info("q", [], scorer=lambda p: [])
    assert out == [] and info.candidates == 0


def test_failure_falls_back_to_retrieval_order_with_warning():
    def boom(pairs):
        raise RuntimeError("CUDA OOM")

    out, info = rerank_with_info("q", cands(), top_n=2, scorer=boom)
    assert info.fallback and "CUDA OOM" in info.warnings[0]
    assert [c["chunk_id"] for c in out] == ["a", "b"]
    assert out[0]["rerank_score"] is None


def test_wrong_score_count_is_a_failure_not_a_silent_misalignment():
    _, info = rerank_with_info("q", cands(), scorer=lambda p: [0.1])
    assert info.fallback


def test_normalize_score():
    assert normalize_score(0.7) == 0.7
    assert 0.9 < normalize_score(5.0) < 1.0
    assert 0.0 < normalize_score(-5.0) < 0.1
