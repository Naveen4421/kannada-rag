"""Shared evidence pipeline used by every route:

    processed query -> hybrid retrieval (dense + sparse, RRF) -> rerank
    -> evidence engine -> [context builder -> LLM -> grounding] (callers)

`retrieve_evidence` is the single implementation of the retrieval half; the
fast path, the agent and answer_question() all call it.
"""
from dataclasses import dataclass, field

from src import config as cfg
from src.evidence.engine import EvidenceBundle, build_evidence
from src.retrieval.reranker import rerank_with_info
from src.retrieval.search import hybrid_search


@dataclass
class RetrievalStage:
    retrieved: list = field(default_factory=list)
    reranked: list = field(default_factory=list)
    bundle: EvidenceBundle = field(default_factory=EvidenceBundle)
    warnings: list = field(default_factory=list)
    reranker_fallback: bool = False
    # True when a stage fell back (reranker failed / one retrieval channel
    # failed): the answer is usable but must not be cached.
    degraded: bool = False


def retrieve_evidence(query_text, top_k, top_n, where=None, question_embedding=None, trace=None,
                      scorer=None, search_fn=hybrid_search):
    """Raises StageError('retrieval', ...) if retrieval itself failed. An empty
    result is returned normally (bundle.sufficiency.reason == NO_EVIDENCE)."""
    stage = RetrievalStage()

    def timed(name):
        return trace.stage(name) if trace else _null()

    with timed("retrieval"):
        out = search_fn(query_text, top_k=top_k, where=where, question_embedding=question_embedding)
    stage.retrieved = out.results
    stage.warnings += out.warnings
    stage.degraded = bool(out.warnings)

    with timed("rerank"):
        reranked, info = rerank_with_info(query_text, out.results, top_n=top_n, scorer=scorer)
    stage.reranked = reranked
    stage.warnings += info.warnings
    stage.reranker_fallback = info.fallback
    stage.degraded = stage.degraded or info.fallback

    with timed("evidence"):
        stage.bundle = build_evidence(reranked)
    stage.warnings += stage.bundle.warnings

    if trace:
        trace.section("retrieval", counts=out.counts, timings_ms={k: round(v, 2) for k, v in out.timings_ms.items()},
                      top_k=top_k, candidate_ids=[c["chunk_id"] for c in out.results])
        trace.section("reranking", top_n=top_n, latency_ms=round(info.latency_ms, 2), truncated=info.truncated,
                      fallback=info.fallback,
                      scores=[{"chunk_id": c["chunk_id"], "rerank_score": c.get("rerank_score")} for c in reranked])
        trace.section("evidence", **stage.bundle.summary())
        for w in stage.warnings:
            trace.warn(w)

    return stage


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
