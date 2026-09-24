import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.cache import store
from src.evidence.engine import build_evidence
from src.generation.context_builder import INSUFFICIENT_MARKER
from src.observability import logger
from src.pipeline import query as q
from src.pipeline.errors import StageError
from src.pipeline.rag_core import RetrievalStage
from src.routing.router import RouteDecision
from tests.conftest import TEXT_A1
from tests.test_evidence import chunk

Q = "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಜನಿಸಿದರು?"
GOOD = "ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1]."


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "cache.sqlite3")
    monkeypatch.setattr(logger, "LOG_PATH", tmp_path / "queries.jsonl")
    monkeypatch.setattr("src.embeddings.model.embed_texts", lambda texts, **k: np.array([[1.0, 0.0]]))
    monkeypatch.setattr(q, "fingerprint", lambda route: "fp1")

    def set_route(route):
        simple = route == "simple"
        monkeypatch.setattr(q, "classify", lambda text, emb: RouteDecision(
            route=route, score=-0.1 if simple else 0.1, top_k=6 if simple else 15, top_n=3 if simple else 6,
            use_query_expansion=not simple, use_self_critique=not simple,
            heuristic_score=0.0, semantic_score=-0.1, combined_score=-0.1, confidence=1.0, reason="test"))

    set_route("simple")
    return SimpleNamespace(set_route=set_route, log=tmp_path / "queries.jsonl")


def stage(**kw):
    chunks = [chunk("A_1", "A", TEXT_A1, page=42)]
    return RetrievalStage(retrieved=chunks, reranked=chunks, bundle=build_evidence(chunks), **kw)


def consume(result):
    text = "".join(result["answer_stream"])
    return text, result["stream_error"]


def last_log(env):
    return json.loads(env.log.read_text(encoding="utf-8").strip().splitlines()[-1])


def test_retrieval_failure_is_structured_and_never_reaches_the_llm(env, monkeypatch):
    def boom(*a, **k):
        raise StageError("retrieval", "VECTOR_SEARCH_FAILED", "connection refused")

    monkeypatch.setattr(q, "retrieve_evidence", boom)
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: pytest.fail("LLM must not be called"))

    r = q.ask(Q)
    assert r["success"] is False and r["answer"] is None and r["answer_stream"] is None
    assert r["error_detail"] == {"stage": "retrieval", "code": "VECTOR_SEARCH_FAILED", "message": "connection refused"}
    assert "Evidence retrieval failed" in r["error"] and "No answer was generated" in r["error"]
    assert last_log(env)["errors"][0]["code"] == "VECTOR_SEARCH_FAILED"


def test_routing_failure_is_structured(env, monkeypatch):
    monkeypatch.setattr(q, "classify", lambda *a: (_ for _ in ()).throw(RuntimeError("CUDA OOM")))
    r = q.ask(Q)
    assert r["error_detail"]["stage"] == "routing" and "CUDA OOM" in r["error"]


def test_fast_path_streams_validates_caches_and_hits(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: iter(["ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ", "ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1]."]))

    r = q.ask(Q)
    assert r["route"] == "simple" and not r["cache_hit"] and r["evidence"][0]["evidence_id"] == "E1"
    text, state = consume(r)
    assert text == GOOD and state["error"] is None and state["grounding"]["grounded"]

    log = last_log(env)
    assert log["route"] == "simple" and log["routing"]["reason"] == "test" and log["query_id"]
    assert "total" in log["latency_ms"] and log["validation"]["grounded"] is True

    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: pytest.fail("cache should answer"))
    hit = q.ask(Q)
    assert hit["cache_hit"] and hit["answer"] == GOOD and last_log(env)["cache_hit"] is True


def test_cache_not_served_after_pipeline_fingerprint_changes(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: iter([GOOD]))
    consume(q.ask(Q))

    monkeypatch.setattr(q, "fingerprint", lambda route: "fp2")  # e.g. model or prompt version changed
    r = q.ask(Q)
    assert not r["cache_hit"]


def test_ungrounded_or_degraded_answers_are_not_cached(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: iter(["ರಾಜಪುರೋಹಿತರು 1999 ರಲ್ಲಿ ಜನಿಸಿದರು [E1]."]))
    text, state = consume(q.ask(Q))
    assert not state["grounding"]["grounded"] and state["warning"]
    assert not q.ask(Q)["cache_hit"]

    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage(degraded=True))
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: iter([GOOD]))
    consume(q.ask(Q))
    assert not q.ask(Q)["cache_hit"]


def test_stream_llm_error_is_reported_not_cached(env, monkeypatch):
    def failing(prompt):
        yield "ರಾಜ"
        raise RuntimeError("402 payment required")

    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    monkeypatch.setattr(q, "generate_answer_stream", failing)
    _, state = consume(q.ask(Q))
    assert "LLM_FAILED" in state["error"] and last_log(env)["errors"][0]["code"] == "LLM_FAILED"


def test_marker_is_stripped_from_stream_across_token_boundaries(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    pieces = [INSUFFICIENT_MARKER[:6], INSUFFICIENT_MARKER[6:] + " ಆಧಾರ", " ಸಾಕಾಗುವುದಿಲ್ಲ"]
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: iter(pieces))
    text, state = consume(q.ask(Q))
    assert text == "ಆಧಾರ ಸಾಕಾಗುವುದಿಲ್ಲ" and state["grounding"]["abstained"]


def test_empty_retrieval_answers_without_calling_llm(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: RetrievalStage(bundle=build_evidence([])))
    monkeypatch.setattr(q, "generate_answer_stream", lambda p: pytest.fail("LLM must not be called"))
    r = q.ask(Q)
    assert r["success"] and r["answer"] and r["grounding"]["abstained"] and r["error"] is None


def test_complex_route_uses_agent_and_caches_only_grounded(env, monkeypatch):
    env.set_route("complex")
    calls = []

    def fake_agent(question, **kw):
        calls.append(kw)
        return {"question": question, "answer": GOOD, "retrieved": [], "reranked": [], "evidence": [],
                "grounding": {"grounded": True, "score": 1.0}, "iterations": [{}], "iteration_count": 1,
                "degraded": False, "error": None, "prompt": ""}

    monkeypatch.setattr(q, "run_agentic_answer", fake_agent)
    r = q.ask(Q, where={"book_id": "A"})
    assert r["answer"] == GOOD and calls[0]["where"] == {"book_id": "A"} and calls[0]["normalized_query"] == Q
    assert q.ask(Q, where={"book_id": "A"})["cache_hit"]
    assert not q.ask(Q)["cache_hit"]  # different book filter -> different key


def test_agent_stage_error_becomes_structured_result(env, monkeypatch):
    env.set_route("complex")
    monkeypatch.setattr(q, "run_agentic_answer",
                        lambda *a, **k: (_ for _ in ()).throw(StageError("generation", "LLM_FAILED", "no credits")))
    r = q.ask(Q)
    assert r["error_detail"]["code"] == "LLM_FAILED" and not r["success"]


def test_answer_question_compat_shape(env, monkeypatch):
    monkeypatch.setattr(q, "retrieve_evidence", lambda *a, **k: stage())
    monkeypatch.setattr(q, "default_generate", lambda prompt, trace=None: GOOD)
    r = q.answer_question(Q, top_k=10, top_n=5)
    assert r["answer"] == GOOD and r["error"] is None and r["reranked"][0]["text"] == TEXT_A1
    assert r["grounding"]["grounded"] and r["reranked"][0]["chunk_id"] == "A_1"


def test_logger_never_raises_and_redacts_secrets(env, tmp_path, monkeypatch):
    logger.log_query({"question": "x", "openrouter_api_key": "sk-secret", "nested": {"Authorization": "Bearer z"}})
    line = env.log.read_text()
    assert "sk-secret" not in line and "Bearer z" not in line and "[redacted]" in line
    monkeypatch.setattr(logger, "LOG_PATH", tmp_path)  # a directory: open() fails
    logger.log_query({"question": "x"})  # must not raise
