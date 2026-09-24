import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
CHUNKS_FILE = Path("data/chunks/TK003.jsonl")

TEST_CASES = [
    {
        "question": "ರಾಜಪುರೋಹಿತರ ತಂದೆಯವರ ಮರಣದ ನಂತರ ಜನರಲ್ಲಿ ಅವರ ಬಗ್ಗೆ ಯಾವ ಭಾವನೆ ಇತ್ತು?",
        "expected_chunk": "TK003_0005",
    },
    {
        "question": "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಮತ್ತು ಯಾವ ದಿನಾಂಕದಲ್ಲಿ ಜನಿಸಿದರು?",
        "expected_chunk": "TK003_0007",
    },
    {
        "question": "ರಾಜಪುರೋಹಿತರು ಶಿಕ್ಷಕರಾಗಿ ಹೇಗೆ ತಮ್ಮ ವೃತ್ತಿಯನ್ನು ಆರಂಭಿಸಿದರು?",
        "expected_chunk": "TK003_0008",
    },
    {
        "question": "ರಾಜಪುರೋಹಿತರು ಕರ್ನಾಟಕ ಮತ್ತು ಮಹಾರಾಷ್ಟ್ರದ ಇತಿಹಾಸದ ಬಗ್ಗೆ ಯಾವ ರೀತಿಯ ಲೇಖನಗಳನ್ನು ಬರೆದರು?",
        "expected_chunk": "TK003_0009",
    },
    {
        "question": "ರಾಜಪುರೋಹಿತರ ಸ್ವಭಾವ ಹೇಗಿತ್ತು?",
        "expected_chunk": "TK003_0045",
    },
    {
        "question": "ರಾಜಪುರೋಹಿತರು ಇತಿಹಾಸ ಸಂಶೋಧನೆಗಾಗಿ ಮಾಡಿದ ಪ್ರಮುಖ ಕೆಲಸವೇನು?",
        "expected_chunk": "TK003_0084",
    },
]


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main():
    print("=" * 70)
    print("Kannada RAG Retrieval Evaluation")
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

    print("\nEmbedding all chunks...")

    chunk_embeddings = model.encode(
        texts,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print(f"Embedding shape: {chunk_embeddings.shape}")
    print(f"Embedding device: {chunk_embeddings.device}")

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for number, test in enumerate(TEST_CASES, start=1):

        question = test["question"]
        expected = test["expected_chunk"]

        question_embedding = model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarities = torch.matmul(
            chunk_embeddings,
            question_embedding
        )

        top_k = min(5, len(chunks))

        scores, indices = torch.topk(
            similarities,
            k=top_k
        )

        retrieved_ids = [
            chunks[index]["chunk_id"]
            for index in indices
        ]

        rank = (
            retrieved_ids.index(expected) + 1
            if expected in retrieved_ids
            else None
        )

        if rank == 1:
            top1_correct += 1

        if rank is not None and rank <= 3:
            top3_correct += 1

        if rank is not None and rank <= 5:
            top5_correct += 1

        print(f"\n{'-' * 70}")
        print(f"Question {number}")
        print(f"Question: {question}")
        print(f"Expected: {expected}")

        if rank:
            print(f"Found at rank: {rank}")
        else:
            print("Expected chunk: NOT FOUND in Top 5")

        print("\nTop 5:")

        for rank_number, (score, index) in enumerate(
            zip(scores, indices),
            start=1
        ):
            chunk = chunks[index]

            marker = "  <-- EXPECTED" if chunk["chunk_id"] == expected else ""

            print(
                f"{rank_number}. "
                f"{chunk['chunk_id']} "
                f"score={score.item():.4f}"
                f"{marker}"
            )

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Top-1: {top1_correct}/{total} "
        f"({top1_correct / total * 100:.1f}%)"
    )

    print(
        f"Top-3: {top3_correct}/{total} "
        f"({top3_correct / total * 100:.1f}%)"
    )

    print(
        f"Top-5: {top5_correct}/{total} "
        f"({top5_correct / total * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
