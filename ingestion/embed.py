import argparse
import json
import uuid
from pathlib import Path

from qdrant_client.http import models

from common.embedding_model import embed_texts, embed_sparse
from common.index import COLLECTION_NAME, INDEX_DIR, QDRANT_HOST, QDRANT_PORT, get_client


BATCH_SIZE = 64


def ensure_collection(client, name, vector_size):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                ),
            },
        )


def load_chunks(jsonl_path):
    chunks = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def _make_point(chunk, dense_vector, sparse_vector):
    payload = dict(chunk)

    return models.PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])),
        vector={
            "dense": dense_vector.tolist(),
            "sparse": models.SparseVector(
                indices=[int(i) for i in sparse_vector.indices],
                values=[float(v) for v in sparse_vector.values],
            ),
        },
        payload=payload,
    )


def embed_and_index(jsonl_paths, persist_dir=INDEX_DIR, batch_size=32, collection_name=COLLECTION_NAME):
    client = get_client()

    total = 0

    for jsonl_path in jsonl_paths:
        chunks = load_chunks(jsonl_path)

        if not chunks:
            continue

        print(f"\n{jsonl_path}: {len(chunks)} chunks")

        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start:start + BATCH_SIZE]
            texts = [chunk["text"] for chunk in batch]

            dense_embeddings = embed_texts(texts, batch_size=batch_size)
            sparse_embeddings = embed_sparse(texts)

            ensure_collection(client, collection_name, len(dense_embeddings[0]))

            points = [
                _make_point(chunk, dense_vector, sparse_vector)
                for chunk, dense_vector, sparse_vector in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            # Deterministic point IDs (uuid5 of chunk_id) make this upsert safe to
            # resume: a crash mid-migration just re-writes already-done batches
            # identically, no duplication.
            client.upsert(collection_name=collection_name, points=points)

            total += len(batch)

    return total


def main():
    parser = argparse.ArgumentParser(description="Embed chunk JSONL files and upsert into the Qdrant index.")
    parser.add_argument("jsonl_files", type=Path, nargs="*", help="Defaults to all of data/chunks/*.jsonl")
    parser.add_argument("--persist-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    jsonl_paths = args.jsonl_files or sorted(Path("data/chunks").glob("*.jsonl"))

    if not jsonl_paths:
        print("No chunk files found.")
        return

    total = embed_and_index(
        jsonl_paths,
        persist_dir=args.persist_dir,
        batch_size=args.batch_size,
        collection_name=args.collection,
    )

    print(f"\nIndexed {total} chunks into Qdrant at {QDRANT_HOST}:{QDRANT_PORT} (collection: {args.collection})")


if __name__ == "__main__":
    main()
