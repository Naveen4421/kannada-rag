import argparse
import time
from functools import lru_cache

from qdrant_client.http import models

from retrieval import config as cfg
from retrieval.pipeline.errors import StageError
from retrieval.retrieve.types import RetrievalOutput, RetrievalResult


# Resolved lazily (see _defaults) so this module imports without the ML stack.
def _defaults():
    from common.index import COLLECTION_NAME, get_client

    return COLLECTION_NAME, get_client


@lru_cache(maxsize=1)
def _default_client():
    return _defaults()[1]()


def build_filter(where):
    """Equality filter from {field: value}. A list/tuple value matches any of
    its members. Returns None when there is nothing to filter on."""
    if not where:
        return None

    conditions = []
    for key, value in where.items():
        if isinstance(value, (list, tuple, set)):
            match = models.MatchAny(any=list(value))
        else:
            match = models.MatchValue(value=value)
        conditions.append(models.FieldCondition(key=key, match=match))

    return models.Filter(must=conditions)


def _hit_to_chunk(point):
    payload = dict(point.payload or {})
    return {
        "chunk_id": payload.get("chunk_id", str(point.id)),
        "score": float(point.score),
        "payload": payload,
    }


def _run_channel(client, collection_name, query, using, limit, query_filter):
    response = client.query_points(
        collection_name=collection_name,
        query=query,
        using=using,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return [_hit_to_chunk(point) for point in response.points]


def rrf_fuse(dense_hits, sparse_hits, k=60):
    """Reciprocal Rank Fusion over the two channel rankings.

    rrf(d) = sum over channels of 1 / (k + rank_channel(d)), ranks 1-based.
    Chunks are de-duplicated by chunk_id (a chunk found by both channels
    appears once, with both ranks and scores). Order is deterministic: RRF
    score desc, then best channel rank, then chunk_id.
    """
    fused = {}

    def add(hits, channel):
        seen = set()
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:  # duplicate within one channel: keep best rank
                continue
            seen.add(chunk_id)

            payload = hit["payload"]
            entry = fused.get(chunk_id)
            if entry is None:
                entry = fused[chunk_id] = RetrievalResult(
                    chunk_id=chunk_id,
                    book_id=payload.get("book_id"),
                    page=payload.get("page_start"),
                    text=payload.get("text", ""),
                    payload={key: value for key, value in payload.items()
                             if key not in ("text", "chunk_id", "book_id")},
                )
            setattr(entry, f"{channel}_score", hit["score"])
            setattr(entry, f"{channel}_rank", rank)
            entry.rrf_score += 1.0 / (k + rank)

    add(dense_hits, "dense")
    add(sparse_hits, "sparse")

    def sort_key(r):
        ranks = [x for x in (r.dense_rank, r.sparse_rank) if x is not None]
        return (-r.rrf_score, min(ranks), r.chunk_id)

    return sorted(fused.values(), key=sort_key)


def hybrid_search(query, top_k=10, where=None, question_embedding=None, sparse_vector=None,
                  client=None, collection_name=None, config=None):
    """Dense + sparse retrieval fused with client-side RRF.

    The two channels are queried separately (not via Qdrant's server-side
    fusion) so per-channel scores, ranks, counts and latency survive. The
    metadata filter is applied inside each channel query.

    A single failed channel degrades to the other one and is reported in
    `warnings`; both failing raises StageError. Zero hits from healthy
    channels is a normal (empty) result, not an error.
    """
    config = config or cfg.RETRIEVAL
    collection_name = collection_name or _defaults()[0]
    client = client or _default_client()

    out = RetrievalOutput()
    query_filter = build_filter(where)
    dense_limit, sparse_limit = config.limits(top_k)
    dense_hits, sparse_hits = [], []
    failures = {}

    # -- dense -------------------------------------------------------------
    try:
        t = time.perf_counter()
        if question_embedding is None:
            from common.embedding_model import embed_texts

            question_embedding = embed_texts([query])[0]
        vec = question_embedding.tolist() if hasattr(question_embedding, "tolist") else list(question_embedding)
        out.timings_ms["dense_embed"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        dense_hits = _run_channel(client, collection_name, vec, "dense", dense_limit, query_filter)
        out.timings_ms["dense"] = (time.perf_counter() - t) * 1000
    except Exception as exc:
        failures["dense"] = str(exc)

    # -- sparse ------------------------------------------------------------
    try:
        t = time.perf_counter()
        if sparse_vector is None:
            from common.embedding_model import embed_sparse_query

            sparse_vector = embed_sparse_query(query)
        sparse_q = models.SparseVector(
            indices=[int(i) for i in sparse_vector.indices],
            values=[float(v) for v in sparse_vector.values],
        )
        out.timings_ms["sparse_embed"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        sparse_hits = _run_channel(client, collection_name, sparse_q, "sparse", sparse_limit, query_filter)
        out.timings_ms["sparse"] = (time.perf_counter() - t) * 1000
    except Exception as exc:
        failures["sparse"] = str(exc)

    if len(failures) == 2:
        raise StageError("retrieval", "VECTOR_SEARCH_FAILED",
                         "Both retrieval channels failed: " + "; ".join(f"{k}: {v}" for k, v in failures.items()))

    for channel, message in failures.items():
        out.warnings.append(f"{channel} channel failed, using the other channel only: {message}")

    t = time.perf_counter()
    fused = rrf_fuse(dense_hits, sparse_hits, k=config.rrf_k)
    out.timings_ms["rrf"] = (time.perf_counter() - t) * 1000

    out.counts = {
        "dense": len(dense_hits),
        "sparse": len(sparse_hits),
        "fused_unique": len(fused),
        "returned": min(len(fused), top_k),
    }
    out.results = [r.to_dict() for r in fused[:top_k]]
    out.timings_ms["total"] = sum(v for k, v in out.timings_ms.items() if k != "total")

    return out


def search(question, top_k=10, persist_dir=None, collection_name=None, where=None, question_embedding=None):
    """Backward-compatible wrapper returning only the candidate list
    (evaluation scripts and older callers). `persist_dir` is ignored (kept
    for signature compatibility)."""
    return hybrid_search(question, top_k=top_k, where=where, question_embedding=question_embedding,
                         collection_name=collection_name).results


def main():
    parser = argparse.ArgumentParser(description="Search the Kannada chunk index for a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    out = hybrid_search(args.question, top_k=args.top_k)

    print(f"Question: {args.question}")
    print(f"Counts: {out.counts}  Timings(ms): { {k: round(v, 1) for k, v in out.timings_ms.items()} }")
    for warning in out.warnings:
        print(f"WARNING: {warning}")
    print()

    for rank, result in enumerate(out.results, start=1):
        preview = result["text"][:120].replace("\n", " ")

        print(f"--- Rank {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"RRF:      {result['rrf_score']:.4f}  dense #{result['dense_rank']} ({result['dense_score']})  "
              f"sparse #{result['sparse_rank']} ({result['sparse_score']})")
        print(f"Pages:    {result.get('page_start')} - {result.get('page_end')}")
        print(f"Text:     {preview}...")
        print()


if __name__ == "__main__":
    main()
