def rerank(question, candidates, top_n=5):
    """
    Placeholder for the reranking stage.

    Candidates are assumed already ranked best-first by vector search, so
    this just truncates to top_n. Swap in a real cross-encoder (e.g.
    BAAI/bge-reranker-v2-m3) later without changing this function's
    signature, since pipeline/query.py depends on it staying
    (question, candidates, top_n) -> list[dict].
    """
    return candidates[:top_n]
