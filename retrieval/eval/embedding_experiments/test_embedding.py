import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
CHUNKS_FILE = Path("data/chunks/TK003.jsonl")
NUM_CHUNKS = 10

QUESTION = "ರಾಜಪುರೋಹಿತರ ತಂದೆಯವರ ಮರಣದ ನಂತರ ಅವರ ಬಗ್ಗೆ ಜನರ ಅಭಿಪ್ರಾಯ ಹೇಗಿತ್ತು?"


def load_chunks(path, limit):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

            if len(chunks) >= limit:
                break

    return chunks


def main():
    print("=" * 70)
    print("Kannada Embedding Experiment")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Model device: {model.device}")

    chunks = load_chunks(CHUNKS_FILE, NUM_CHUNKS)

    print(f"\nChunks loaded: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating chunk embeddings...")

    chunk_embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"\nEmbedding tensor shape: {chunk_embeddings.shape}")
    print(f"Vector dimension: {chunk_embeddings.shape[1]}")
    print(f"Embedding device: {chunk_embeddings.device}")

    print("\nGenerating question embedding...")

    question_embedding = model.encode(
        QUESTION,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    print(f"Question vector shape: {question_embedding.shape}")
    print(f"Question device: {question_embedding.device}")

    similarities = torch.matmul(
        chunk_embeddings,
        question_embedding
    )

    ranked = torch.argsort(
        similarities,
        descending=True
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(QUESTION)

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for rank, index in enumerate(ranked, start=1):
        chunk = chunks[index]
        score = similarities[index].item()

        print(f"\n--- Rank {rank} ---")
        print(f"Score:       {score:.4f}")
        print(f"Chunk ID:    {chunk['chunk_id']}")
        print(f"Pages:       {chunk['page_start']} - {chunk['page_end']}")
        print(
            f"Paragraphs:  "
            f"{chunk['paragraph_start']} - {chunk['paragraph_end']}"
        )
        print(f"Characters:  {chunk['char_count']}")
        print("\nText:")
        print(chunk["text"])


if __name__ == "__main__":
    main()
