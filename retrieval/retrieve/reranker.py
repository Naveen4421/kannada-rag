import math
import time
from dataclasses import dataclass, field

from retrieval.pipeline.errors import StageError


@dataclass
class RerankInfo:
    latency_ms: float = 0.0
    candidates: int = 0
    truncated: int = 0          # candidates whose (query, passage) exceeds the model's max_length
    fallback: bool = False      # True: reranker failed, order is the retrieval (RRF) order
    warnings: list = field(default_factory=list)


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def normalize_score(raw):
    """Map a cross-encoder score to 0..1. sentence-transformers' CrossEncoder
    already applies a sigmoid for single-label models (scores in 0..1); if a
    raw logit slips through (outside 0..1) squash it. Ambiguous only for
    logits that happen to lie in 0..1, which is accepted."""
    return raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)


def _default_scorer(pairs):
    from retrieval.retrieve.reranker_model import get_reranker

    return [float(s) for s in get_reranker().predict(pairs, batch_size=32)]


def _count_truncated(pairs, max_length=512):
    try:
        from retrieval.retrieve.reranker_model import get_reranker

        tokenizer = get_reranker().tokenizer
        return sum(1 for q, p in pairs if len(tokenizer(q, p)["input_ids"]) > max_length)
    except Exception:
        return 0


def rerank_with_info(question, candidates, top_n=5, scorer=None, check_truncation=True):
    """Second-stage relevance scoring: retrieval asked 'could this passage be
    relevant?'; the cross-encoder scores each (question, passage) pair
    jointly to answer 'is it relevant to exactly this question?'.

    Does not mutate the input candidates. Sorting is deterministic (rerank
    score desc, RRF score desc, chunk_id). If the model fails, falls back to
    retrieval order with `fallback=True` and no rerank scores -- downstream
    evidence scoring then drops the reranker component instead of inventing
    one.
    """
    info = RerankInfo(candidates=len(candidates))
    if not candidates:
        return [], info

    started = time.perf_counter()
    pairs = [(question, c["text"]) for c in candidates]

    try:
        scores = (scorer or _default_scorer)(pairs)
        if len(scores) != len(candidates):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(candidates)} candidates")
    except Exception as exc:
        info.fallback = True
        info.warnings.append(f"reranker failed, using retrieval order: {exc}")
        info.latency_ms = (time.perf_counter() - started) * 1000
        ordered = [dict(c, rerank_score=None, rerank_score_norm=None) for c in candidates]
        return ordered[:top_n], info

    scored = []
    for candidate, raw in zip(candidates, scores):
        scored.append(dict(candidate, rerank_score=float(raw), rerank_score_norm=normalize_score(float(raw))))

    scored.sort(key=lambda c: (-c["rerank_score"], -c.get("rrf_score", 0.0), c["chunk_id"]))

    if scorer is None and check_truncation:
        info.truncated = _count_truncated(pairs)
        if info.truncated:
            info.warnings.append(
                f"{info.truncated}/{len(pairs)} passages exceed the reranker's 512-token limit; "
                "the tail of those passages was not scored")

    info.latency_ms = (time.perf_counter() - started) * 1000
    return scored[:top_n], info


def rerank(question, candidates, top_n=5, scorer=None):
    """Backward-compatible wrapper returning only the ranked list."""
    return rerank_with_info(question, candidates, top_n=top_n, scorer=scorer)[0]
