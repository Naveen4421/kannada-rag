import argparse
import json
from pathlib import Path

import chromadb

from src.embeddings.model import embed_texts


INDEX_DIR = Path("data/index/chroma")
COLLECTION_NAME = "kannada_chunks"


def get_collection(persist_dir=INDEX_DIR, name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=str(persist_dir))

    return client.get_or_create_collection(
        name,
        metadata={"hnsw:space": "cosine"},
    )


def load_chunks(jsonl_path):
    chunks = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def embed_and_index(jsonl_paths, persist_dir=INDEX_DIR, batch_size=32):
    collection = get_collection(persist_dir=persist_dir)

    total = 0

    for jsonl_path in jsonl_paths:
        chunks = load_chunks(jsonl_path)

        if not chunks:
            continue

        print(f"\n{jsonl_path}: {len(chunks)} chunks")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_texts(texts, batch_size=batch_size)

        ids = [chunk["chunk_id"] for chunk in chunks]
        metadatas = [
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total += len(chunks)

    return total


def main():
    parser = argparse.ArgumentParser(description="Embed chunk JSONL files and upsert into the Chroma index.")
    parser.add_argument("jsonl_files", type=Path, nargs="*", help="Defaults to all of data/chunks/*.jsonl")
    parser.add_argument("--persist-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    jsonl_paths = args.jsonl_files or sorted(Path("data/chunks").glob("*.jsonl"))

    if not jsonl_paths:
        print("No chunk files found.")
        return

    total = embed_and_index(jsonl_paths, persist_dir=args.persist_dir, batch_size=args.batch_size)

    print(f"\nIndexed {total} chunks into: {args.persist_dir} (collection: {COLLECTION_NAME})")


if __name__ == "__main__":
    main()
