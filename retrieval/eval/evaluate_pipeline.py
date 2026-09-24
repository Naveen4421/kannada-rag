from retrieval.eval.evaluate import TEST_CASES, reciprocal_rank
from retrieval.retrieve.search import search
from retrieval.retrieve.reranker import rerank


def evaluate_stage(label, get_ranked_ids, top_k=5):
    print("\n" + "=" * 80)
    print(f"STAGE: {label}")
    print("=" * 80)

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks = []

    for test in TEST_CASES:
        relevant = set(test["relevant_chunks"])
        ranked_ids = get_ranked_ids(test["question"])

        first_relevant_rank = None
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            if chunk_id in relevant:
                first_relevant_rank = rank
                break

        if first_relevant_rank == 1:
            top1_hits += 1
        if first_relevant_rank is not None and first_relevant_rank <= 3:
            top3_hits += 1
        if first_relevant_rank is not None and first_relevant_rank <= top_k:
            top5_hits += 1

        reciprocal_ranks.append(reciprocal_rank(first_relevant_rank))

        marker = f"rank {first_relevant_rank}" if first_relevant_rank else "NOT FOUND"
        print(f"{test['id']}: {marker}")

    total = len(TEST_CASES)
    print(f"\nTop-1: {top1_hits}/{total} ({top1_hits/total*100:.1f}%)")
    print(f"Top-3: {top3_hits}/{total} ({top3_hits/total*100:.1f}%)")
    print(f"Top-{top_k}: {top5_hits}/{total} ({top5_hits/total*100:.1f}%)")
    print(f"MRR: {sum(reciprocal_ranks)/total:.4f}")


def main():
    def retrieved_ids(question):
        return [c["chunk_id"] for c in search(question, top_k=10, where={"book_id": "TK003"})]

    def reranked_ids(question):
        candidates = search(question, top_k=10, where={"book_id": "TK003"})
        reranked = rerank(question, candidates, top_n=5)
        return [c["chunk_id"] for c in reranked]

    evaluate_stage("Retrieved (search only, top-10)", retrieved_ids, top_k=5)
    evaluate_stage("Reranked (search top-10 -> rerank top-5)", reranked_ids, top_k=5)


if __name__ == "__main__":
    main()
