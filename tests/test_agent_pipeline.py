import pytest

from src.config import AgentConfig
from src.evidence.engine import build_evidence
from src.generation.context_builder import INSUFFICIENT_MARKER
from src.observability.trace import QueryTrace
from src.pipeline.agent import NO_EVIDENCE_ANSWER, UNGROUNDED_CAVEAT, run_agentic_answer
from src.pipeline.errors import StageError
from src.pipeline.rag_core import RetrievalStage, retrieve_evidence
from src.retrieval.search import hybrid_search
from tests.conftest import TEXT_A1, TEXT_A2, TEXT_B1, sparse
from tests.test_evidence import chunk

Q = "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಜನಿಸಿದರು?"
GOOD = "ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1]."
BAD = GOOD + " ಅವರು ಅಮೆರಿಕಾದ ವಿಶ್ವವಿದ್ಯಾಲಯದಲ್ಲಿ ಪ್ರಾಧ್ಯಾಪಕರಾಗಿ ಕೆಲಸ ಮಾಡಿ ನೊಬೆಲ್ ಬಹುಮಾನ ಪಡೆದರು."
CFG = AgentConfig(max_iterations=3, min_grounding_score=0.9, use_llm_judge=False)


def good_stage():
    chunks = [chunk("A_1", "A", TEXT_A1, page=42)]
    return RetrievalStage(retrieved=chunks, reranked=chunks, bundle=build_evidence(chunks))


def empty_stage():
    return RetrievalStage(bundle=build_evidence([]))


class Recorder:
    def __init__(self, stages):
        self.stages, self.calls = list(stages), []

    def __call__(self, query, top_k, top_n, where=None, question_embedding=None, trace=None):
        self.calls.append({"query": query, "top_k": top_k, "top_n": top_n, "where": where})
        return self.stages[min(len(self.calls) - 1, len(self.stages) - 1)]


def run(retrieve, answers, **kw):
    prompts, replies = [], list(answers)

    def gen(prompt):
        prompts.append(prompt)
        return replies[min(len(prompts) - 1, len(replies) - 1)]

    kw.setdefault("variants_fn", lambda q: [q, q + " (ಪರ್ಯಾಯ)"])
    result = run_agentic_answer(Q, 6, 3, where={"book_id": "A"}, normalized_query=Q, config=CFG,
                                generate_fn=gen, retrieve_fn=retrieve, **kw)
    return result, prompts


def test_end_to_end_real_retrieval_rerank_evidence_generation_grounding(qdrant):
    def search_fn(query, top_k, where, question_embedding):
        return hybrid_search(query, top_k=top_k, where=where, question_embedding=question_embedding,
                             sparse_vector=sparse([0]), client=qdrant, collection_name="kannada_chunks")

    trace = QueryTrace(Q)
    stage = retrieve_evidence(Q, 3, 2, question_embedding=[1, 0, 0], search_fn=search_fn, trace=trace,
                              scorer=lambda pairs: [0.95 if "1880" in p else 0.05 for _, p in pairs])
    d = trace.data
    assert {"retrieval", "rerank", "evidence"} <= set(d["latency_ms"])
    assert {"dense", "sparse", "rrf"} <= set(d["retrieval"]["timings_ms"]) and d["retrieval"]["counts"]["dense"] >= 1
    assert d["reranking"]["scores"] and d["evidence"]["evidence_ids"] and d["evidence"]["pages"]
    assert TEXT_A1 not in str(d)  # passage text is never logged
    assert stage.reranked[0]["chunk_id"] in ("A_0001", "B_0001")
    assert stage.reranked[0]["dense_rank"] is not None and stage.bundle.items[0]["page"] == "3"

    result, prompts = run(lambda *a, **k: stage, [GOOD.replace("[E1]", "[E1]")])
    assert result["grounding"]["grounded"] and result["iteration_count"] == 1
    assert result["answer"] == GOOD and "EVIDENCE 1 [E1]" in prompts[0] and Q in prompts[0]
    top = result["evidence"][0]
    assert top["book_id"] in ("A", "B") and top["chunk_id"] and top["components"]["reranker"] == 0.95
    assert result["iterations"][0]["retry_reason"] is None


def test_no_retry_when_first_answer_is_grounded():
    rec = Recorder([good_stage()])
    result, prompts = run(rec, [GOOD])
    assert len(rec.calls) == 1 and len(prompts) == 1 and result["iteration_count"] == 1


def test_insufficient_retrieval_broadens_before_generating():
    rec = Recorder([empty_stage(), good_stage()])
    result, prompts = run(rec, [GOOD])
    assert rec.calls[1] == {"query": Q, "top_k": 12, "top_n": 6, "where": None}
    first = result["iterations"][0]
    assert (first["retry_reason"], first["retry_action"], first["generated"]) == ("NO_EVIDENCE", "broaden", False)
    assert len(prompts) == 1 and result["grounding"]["grounded"]


def test_no_evidence_anywhere_never_calls_llm_and_tries_variant():
    rec = Recorder([empty_stage()])
    result, prompts = run(rec, [GOOD])
    assert prompts == [] and result["answer"] == NO_EVIDENCE_ANSWER and result["grounding"]["abstained"]
    assert [c["query"] for c in rec.calls] == [Q, Q, Q + " (ಪರ್ಯಾಯ)"]
    assert [i["retry_action"] for i in result["iterations"]] == ["broaden", "query_variant", None]


def test_unsupported_claim_regenerates_on_same_evidence_with_feedback():
    rec = Recorder([good_stage()])
    result, prompts = run(rec, [BAD, GOOD])
    assert len(rec.calls) == 1 and len(prompts) == 2  # evidence reused, not re-retrieved
    assert "CORRECTION" not in prompts[0] and "CORRECTION" in prompts[1] and "Unsupported" in prompts[1]
    first = result["iterations"][0]
    assert (first["retry_reason"], first["retry_action"]) == ("UNSUPPORTED_CLAIMS", "regenerate")
    assert result["grounding"]["grounded"] and UNGROUNDED_CAVEAT not in result["answer"]


def test_fabricated_citation_is_a_retry_reason():
    result, _ = run(Recorder([good_stage()]), [GOOD.replace("[E1]", "[E7]"), GOOD])
    assert result["iterations"][0]["retry_reason"] == "INVALID_CITATIONS"


def test_iteration_cap_and_caveat_when_never_grounded():
    rec = Recorder([good_stage()])
    result, prompts = run(rec, [BAD])
    assert result["iteration_count"] == 3 and len(prompts) == 3
    assert not result["grounding"]["grounded"] and result["answer"].endswith(UNGROUNDED_CAVEAT.strip())
    assert result["iterations"][-1]["retry_reason"] is None


def test_model_abstention_is_returned_without_marker():
    result, _ = run(Recorder([good_stage()]), [f"{INSUFFICIENT_MARKER} ಆಧಾರ ಸಾಕಾಗುವುದಿಲ್ಲ"])
    assert result["answer"] == "ಆಧಾರ ಸಾಕಾಗುವುದಿಲ್ಲ" and result["grounding"]["abstained"]


def test_retrieval_failure_propagates_as_structured_error():
    def boom(*a, **k):
        raise StageError("retrieval", "VECTOR_SEARCH_FAILED", "down")

    with pytest.raises(StageError) as e:
        run(boom, [GOOD])
    assert e.value.to_dict() == {"stage": "retrieval", "code": "VECTOR_SEARCH_FAILED", "message": "down"}


def test_llm_failure_propagates_as_structured_error(monkeypatch):
    def fail(prompt):
        raise StageError("generation", "LLM_FAILED", "402")

    with pytest.raises(StageError) as e:
        run_agentic_answer(Q, 6, 3, config=CFG, generate_fn=fail, retrieve_fn=Recorder([good_stage()]))
    assert e.value.code == "LLM_FAILED"


def test_weak_evidence_at_last_iteration_still_generates_with_abstention_allowed():
    weak = build_evidence([chunk("A_1", "A", TEXT_A1, rerank=0.02)])
    stage = RetrievalStage(retrieved=[], reranked=[], bundle=weak)
    rec = Recorder([stage])
    result, prompts = run(rec, [f"{INSUFFICIENT_MARKER} ಸಾಕಾಗುವುದಿಲ್ಲ"])
    assert result["iterations"][0]["sufficiency"]["reason"] == "LOW_RERANK_CONFIDENCE"
    assert len(prompts) == 1 and result["grounding"]["abstained"]
