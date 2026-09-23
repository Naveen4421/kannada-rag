import argparse

from qdrant_client.http import models

from src.embeddings.embed import COLLECTION_NAME, INDEX_DIR, get_client
from src.embeddings.model import embed_texts, embed_sparse_query


def search(question, top_k=10, persist_dir=INDEX_DIR, collection_name=COLLECTION_NAME, where=None, question_embedding=None):
    client = get_client()

    dense_vec = question_embedding if question_embedding is not None else embed_texts([question])[0]
    sparse_vec = embed_sparse_query(question)

    query_filter = None
    if where:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in where.items()
            ]
        )

    prefetch_limit = max(top_k * 4, 50)

    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec),
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=[int(i) for i in sparse_vec.indices],
                    values=[float(v) for v in sparse_vec.values],
                ),
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    ranked = []

    for result in response.points:
        payload = result.payload or {}
        ranked.append({
            "chunk_id": payload.get("chunk_id", str(result.id)),
            "score": result.score,
            "text": payload.get("text", ""),
            **{key: value for key, value in payload.items() if key != "text"},
        })

    return ranked


def main():
    parser = argparse.ArgumentParser(description="Search the Kannada chunk index for a question.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    results = search(args.question, top_k=args.top_k)

    print(f"Question: {args.question}\n")

    for rank, result in enumerate(results, start=1):
        preview = result["text"][:120].replace("\n", " ")

        print(f"--- Rank {rank} ---")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score:    {result['score']:.4f}")
        print(f"Pages:    {result['page_start']} - {result['page_end']}")
        print(f"Text:     {preview}...")
        print()


if __name__ == "__main__":
    main()
