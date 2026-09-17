import argparse

from src.embeddings.embed import INDEX_DIR, COLLECTION_NAME, get_collection
from src.embeddings.model import embed_texts


def search(question, top_k=10, persist_dir=INDEX_DIR, collection_name=COLLECTION_NAME, where=None):
    collection = get_collection(persist_dir=persist_dir, name=collection_name)

    question_embedding = embed_texts([question])[0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ranked = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        ranked.append({
            "chunk_id": chunk_id,
            "score": 1 - distance,
            "text": text,
            **metadata,
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
