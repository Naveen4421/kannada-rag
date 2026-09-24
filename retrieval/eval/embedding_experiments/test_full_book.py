import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
CHUNKS_FILE = Path("data/chunks/TK003.jsonl")

QUESTION = "ರಾಜಪುರೋಹಿತರ ತಂದೆಯವರ ಮರಣದ ನಂತರ ಅವರ ಬಗ್ಗೆ ಜನರ ಅಭಿಪ್ರಾಯ ಹೇಗಿತ್ತು?"


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    print("=" * 70)
    print("Full Book Kannada Retrieval Test")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    chunks = load_chunks(CHUNKS_FILE)

    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("\nGenerating embeddings for the entire book...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding device: {embeddings.device}")

    print("\nEmbedding question...")

    question_embedding = model.encode(
        QUESTION,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )

    similarities = torch.matmul(
        embeddings,
        question_embedding
    )

    top_k = min(10, len(chunks))

    scores, indices = torch.topk(
        similarities,
        k=top_k
    )

    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(QUESTION)

    print("\n" + "=" * 70)
    print(f"TOP {top_k} RETRIEVED CHUNKS")
    print("=" * 70)

    for rank, (score, index) in enumerate(
        zip(scores, indices),
        start=1
    ):
        chunk = chunks[index]

        print(f"\n--- Rank {rank} ---")
        print(f"Score:       {score.item():.4f}")
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
