import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-m3"
CHUNKS_FILE = Path("data/chunks/TK003.jsonl")

# Each question can have one or more relevant chunks.
# This is important because an answer can span multiple chunks.
TEST_CASES = [
    {
        "id": "Q1",
        "question": "ರಾಜಪುರೋಹಿತರ ತಂದೆಯವರ ಮರಣದ ನಂತರ ಜನರಲ್ಲಿ ಅವರ ಬಗ್ಗೆ ಯಾವ ಭಾವನೆ ಇತ್ತು?",
        "relevant_chunks": ["TK003_0005"],
    },
    {
        "id": "Q2",
        "question": "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಮತ್ತು ಯಾವ ದಿನಾಂಕದಲ್ಲಿ ಜನಿಸಿದರು?",
        "relevant_chunks": ["TK003_0007"],
    },
    {
        "id": "Q3",
        "question": "ರಾಜಪುರೋಹಿತರು ಶಿಕ್ಷಕರಾಗಿ ಹೇಗೆ ತಮ್ಮ ವೃತ್ತಿಯನ್ನು ಆರಂಭಿಸಿದರು?",
        "relevant_chunks": ["TK003_0007", "TK003_0008"],
    },
    {
        "id": "Q4",
        "question": "ರಾಜಪುರೋಹಿತರು ಕರ್ನಾಟಕ ಮತ್ತು ಮಹಾರಾಷ್ಟ್ರದ ಇತಿಹಾಸದ ಬಗ್ಗೆ ಯಾವ ರೀತಿಯ ಲೇಖನಗಳನ್ನು ಬರೆದರು?",
        "relevant_chunks": ["TK003_0009"],
    },
    {
        "id": "Q5",
        "question": "ರಾಜಪುರೋಹಿತರ ಸ್ವಭಾವ ಹೇಗಿತ್ತು?",
        "relevant_chunks": ["TK003_0045", "TK003_0046"],
    },
    {
        "id": "Q6",
        "question": "ರಾಜಪುರೋಹಿತರು ಇತಿಹಾಸ ಸಂಶೋಧನೆಗಾಗಿ ಮಾಡಿದ ಪ್ರಮುಖ ಕೆಲಸವೇನು?",
        "relevant_chunks": ["TK003_0084"],
    },
    {
        "id": "Q7",
        "question": "ರಾಜಪುರೋಹಿತರಿಗೆ ಕನ್ನಡದ ಮೇಲೆ ಪ್ರೀತಿ ಮೂಡಲು ಕಾರಣರಾದವರು ಯಾರು?",
        "relevant_chunks": ["TK003_0007"],
    },
    {
        "id": "Q8",
        "question": "ರಾಜಪುರೋಹಿತರು 1908ರಲ್ಲಿ ಸರ್ಕಾರಿ ನೌಕರಿಗೆ ರಾಜೀನಾಮೆ ನೀಡಿದ ನಂತರ ಏನು ಮಾಡಿದರು?",
        "relevant_chunks": ["TK003_0008"],
    },
    {
        "id": "Q9",
        "question": "ರಾಜಪುರೋಹಿತರ ಮೊದಲ ಪುಸ್ತಕ ಯಾವುದು ಮತ್ತು ಅದು ಯಾವ ವರ್ಷ ಪ್ರಕಟವಾಯಿತು?",
        "relevant_chunks": ["TK003_0008"],
    },
    {
        "id": "Q10",
        "question": "ರಾಜಪುರೋಹಿತರು ಇತಿಹಾಸ ಸಂಶೋಧನೆ ಮಾಡುವಾಗ ಯಾವ ವಿಧಾನಗಳನ್ನು ಅನುಸರಿಸುತ್ತಿದ್ದರು?",
        "relevant_chunks": ["TK003_0070", "TK003_0084"],
    },
    {
        "id": "Q11",
        "question": "ರಾಜಪುರೋಹಿತರು ಕರ್ನಾಟಕದ ಇತಿಹಾಸ ಸಂಶೋಧನಾ ಮಂಡಳಿಯನ್ನು ಯಾರ ಸಹಕಾರದಿಂದ ಸ್ಥಾಪಿಸಿದರು?",
        "relevant_chunks": ["TK003_0084"],
    },
    {
        "id": "Q12",
        "question": "ರಾಜಪುರೋಹಿತರು ಯಾವ ರೀತಿಯ ಗ್ರಂಥಗಳನ್ನು ರಚಿಸಿದ್ದಾರೆ?",
        "relevant_chunks": ["TK003_0084"],
    },
    {
        "id": "Q13",
        "question": "ರಾಜಪುರೋಹಿತರು ಯಾವ ಶಾಲೆಗಳಲ್ಲಿ ಶಿಕ್ಷಕರಾಗಿ ಕೆಲಸ ಮಾಡಿದರು?",
        "relevant_chunks": ["TK003_0008", "TK003_0064"],
    },
    {
        "id": "Q14",
        "question": "ರಾಜಪುರೋಹಿತರು ಶಿಲಾಲೇಖಗಳ ಸಂಶೋಧನೆಗೆ ಹೇಗೆ ಕೊಡುಗೆ ನೀಡಿದರು?",
        "relevant_chunks": ["TK003_0070", "TK003_0084"],
    },
    {
        "id": "Q15",
        "question": "ರಾಜಪುರೋಹಿತರ ಸಂಶೋಧನಾ ಕಾರ್ಯದ ವಿಶೇಷತೆ ಏನು?",
        "relevant_chunks": ["TK003_0070", "TK003_0084"],
    },
]


def load_chunks(path):
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def reciprocal_rank(rank):
    if rank is None:
        return 0.0

    return 1.0 / rank


def main():
    print("=" * 80)
    print("KANNADA RAG RETRIEVAL EVALUATION")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDevice: {device}")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
    )

    chunks = load_chunks(CHUNKS_FILE)

    print(f"Total chunks: {len(chunks)}")
    print(f"Test questions: {len(TEST_CASES)}")

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

    top1_hits = 0
    top3_hits = 0
    top5_hits = 0

    reciprocal_ranks = []

    print("\n" + "=" * 80)
    print("QUESTION RESULTS")
    print("=" * 80)

    for test in TEST_CASES:

        question_id = test["id"]
        question = test["question"]
        relevant_chunks = set(test["relevant_chunks"])

        question_embedding = model.encode(
            question,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        similarities = torch.matmul(
            chunk_embeddings,
            question_embedding,
        )

        top_k = min(5, len(chunks))

        scores, indices = torch.topk(
            similarities,
            k=top_k,
        )

        retrieved = []

        for score, index in zip(scores, indices):
            chunk_id = chunks[index]["chunk_id"]

            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "score": score.item(),
                    "relevant": chunk_id in relevant_chunks,
                }
            )

        # Find the first relevant result anywhere in the ranking.
        all_scores, all_indices = torch.sort(
            similarities,
            descending=True,
        )

        first_relevant_rank = None

        for rank, index in enumerate(all_indices, start=1):
            chunk_id = chunks[index]["chunk_id"]

            if chunk_id in relevant_chunks:
                first_relevant_rank = rank
                break

        if first_relevant_rank == 1:
            top1_hits += 1

        if first_relevant_rank is not None and first_relevant_rank <= 3:
            top3_hits += 1

        if first_relevant_rank is not None and first_relevant_rank <= 5:
            top5_hits += 1

        reciprocal_ranks.append(
            reciprocal_rank(first_relevant_rank)
        )

        print("\n" + "-" * 80)
        print(f"{question_id}")
        print(f"Question: {question}")
        print(
            "Relevant chunks: "
            + ", ".join(test["relevant_chunks"])
        )

        if first_relevant_rank is not None:
            print(
                f"First relevant chunk rank: "
                f"{first_relevant_rank}"
            )
        else:
            print("First relevant chunk rank: NOT FOUND")

        print("\nTop 5 retrieved:")

        for rank, result in enumerate(retrieved, start=1):

            marker = " <-- RELEVANT" if result["relevant"] else ""

            print(
                f"{rank}. "
                f"{result['chunk_id']} "
                f"score={result['score']:.4f}"
                f"{marker}"
            )

    total = len(TEST_CASES)

    top1 = top1_hits / total
    top3 = top3_hits / total
    top5 = top5_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)

    print(
        f"Top-1 Recall: {top1_hits}/{total} "
        f"({top1 * 100:.1f}%)"
    )

    print(
        f"Top-3 Recall: {top3_hits}/{total} "
        f"({top3 * 100:.1f}%)"
    )

    print(
        f"Top-5 Recall: {top5_hits}/{total} "
        f"({top5 * 100:.1f}%)"
    )

    print(
        f"MRR: {mrr:.4f}"
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print(
        """
Top-1:
  Relevant evidence is the first retrieved chunk.

Top-3:
  Relevant evidence appears within the first 3 chunks.

Top-5:
  Relevant evidence appears within the first 5 chunks.

MRR:
  Measures how high the first relevant result appears.
"""
    )


if __name__ == "__main__":
    main()
