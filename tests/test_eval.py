import pytest

from retrieval.eval import metrics
from retrieval.eval.gold import load_gold, verify
from retrieval.eval.run_eval import evaluate_rag, evaluate_retrieval
from retrieval.retrieve.types import RetrievalOutput


def test_rank_metrics_known_values():
    ranked, rel = ["a", "b", "c", "d"], ["c", "x"]
    assert metrics.recall_at_k(ranked, rel, 3) == 0.5
    assert metrics.precision_at_k(ranked, rel, 4) == 0.25
    assert metrics.reciprocal_rank(ranked, rel) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["z"], rel) == 0.0
    # single relevant at rank 3 among 1 relevant: dcg = 1/log2(4) = 0.5, ideal = 1
    assert metrics.ndcg_at_k(ranked, ["c"], 4) == pytest.approx(0.5)
    assert metrics.ndcg_at_k(["c", "a"], ["c"], 4) == 1.0
    assert metrics.recall_at_k(ranked, [], 3) is None


def test_aggregate_is_macro_mean_per_category():
    rows = {"x": [metrics.score_ranking(["a"], ["a"], (1,)), metrics.score_ranking(["b"], ["a"], (1,))]}
    agg = metrics.aggregate(rows)
    assert agg["x"]["recall@1"] == 0.5 and agg["ALL"]["n"] == 2 and agg["x"]["mrr"] == 0.5


def g(supported, unsupported, valid=True, abstained=False, grounded=True, cited=True):
    return {"supported_claims": [0] * supported, "unsupported_claims": [0] * unsupported, "citations_valid": valid,
            "abstained": abstained, "grounded": grounded, "citations": {"has_citations": cited}}


def test_grounding_and_abstention_summaries():
    s = metrics.grounding_summary([g(3, 1), g(2, 0, valid=False, grounded=False), g(0, 0, abstained=True)])
    assert s["answered"] == 2 and s["abstained"] == 1
    assert s["supported_claim_rate"] == pytest.approx(5 / 6) and s["unsupported_claim_rate"] == pytest.approx(1 / 6)
    assert s["citation_valid_rate"] == 0.5 and s["grounded_answer_rate"] == 0.5

    a = metrics.abstention_summary([("abstain", True), ("abstain", False), ("answer", False), ("answer", True)])
    assert a["abstain_when_expected"] == 0.5 and a["false_abstain_rate"] == 0.5


def test_latency_summary():
    s = metrics.latency_summary({"dense": [1.0, 2.0, 3.0, 4.0, 100.0]})
    assert s["dense"]["p50"] == 3.0 and s["dense"]["p95"] == 100.0


def test_gold_file_is_well_formed_and_covers_required_categories():
    cases = load_gold()
    cats = {c["category"] for c in cases}
    assert {"factual", "exact_word", "rare_word", "spelling_ocr_variant", "page_specific", "cross_book",
            "insufficient_evidence", "ambiguous"} <= cats
    assert len({c["id"] for c in cases}) == len(cases)
    assert all(c.get("verification") and c["expected_behavior"] in ("answer", "abstain", "clarify_or_abstain") for c in cases)


def test_verifier_catches_bad_labels():
    chunks = {"A_1": {"chunk_id": "A_1", "book_id": "A", "text": "hello world", "page_start": 1, "page_end": 2}}
    bad = [
        {"id": "m", "category": "rare_word", "relevant_chunks": ["A_2"], "must_contain": ["hello"]},
        {"id": "n", "category": "rare_word", "relevant_chunks": ["A_1"], "must_contain": ["nope"]},
        {"id": "p", "category": "page_specific", "relevant_chunks": [], "book_id": "A", "page": 2},
        {"id": "a", "category": "insufficient_evidence", "relevant_chunks": [], "corpus_absent_terms": ["world"]},
    ]
    assert len(verify(bad, chunks)) >= 5
    assert verify([{"id": "ok", "category": "rare_word", "relevant_chunks": ["A_1"], "must_contain": ["hello"]}], chunks) == []


def fake_result(chunk_id, dense, sparse):
    return {"chunk_id": chunk_id, "book_id": "A", "dense_rank": dense, "sparse_rank": sparse}


def test_retrieval_eval_ablation_channels_and_rerank():
    def search_fn(q, k, where):
        out = RetrievalOutput(results=[fake_result("x", 1, None), fake_result("rel", 2, 1), fake_result("y", None, 2)])
        out.timings_ms = {"dense": 5.0, "sparse": 2.0}
        return out

    cases = [{"id": "c1", "category": "cat", "question": "q", "relevant_chunks": ["rel"], "book_id": "A"},
             {"id": "skip", "category": "cat", "question": "q", "relevant_chunks": []}]
    report = evaluate_retrieval(cases, search_fn, rerank_fn=lambda q, c, n: [c[1]], ks=(1, 3))
    st = report["stages"]
    assert st["sparse"]["ALL"]["mrr"] == 1.0 and st["dense"]["ALL"]["mrr"] == 0.5 and st["fused"]["ALL"]["mrr"] == 0.5
    assert st["reranked"]["ALL"]["recall@1"] == 1.0 and st["fused"]["ALL"]["n"] == 1
    assert report["latency_ms"]["retrieval.dense"]["mean"] == 5.0 and report["filter_violations"] == []


def test_rag_eval_reports_abstention_and_citation_correctness():
    cases = [{"id": "a", "category": "factual", "question": "q", "expected_behavior": "answer", "relevant_chunks": ["c1"]},
             {"id": "b", "category": "insufficient_evidence", "question": "q", "expected_behavior": "abstain", "relevant_chunks": []}]

    def answer_fn(q, where):
        if len(answer_fn.calls) == 0:
            answer_fn.calls.append(1)
            return {"grounding": g(2, 0) | {"citations": {"has_citations": True, "cited_ids": ["E1"]}},
                    "evidence": [{"evidence_id": "E1", "chunk_id": "c1"}], "error": None}
        return {"grounding": g(0, 0, abstained=True), "evidence": [], "error": None}

    answer_fn.calls = []
    r = evaluate_rag(cases, answer_fn)
    assert r["citation_correctness"] == 1.0 and r["abstention"]["abstain_when_expected"] == 1.0
    assert r["abstention"]["false_abstain_rate"] == 0.0
