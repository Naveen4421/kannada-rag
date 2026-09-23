from src.reranker.model import get_reranker


def rerank(question, candidates, top_n=5):
    if not candidates:
        return []

    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = get_reranker().predict(pairs, batch_size=32)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)

    return ranked[:top_n]
