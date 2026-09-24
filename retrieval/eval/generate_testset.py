import json
import random
from pathlib import Path

from common.llm import generate_answer


CHUNKS_DIR = Path("data/chunks")
OUTPUT_PATH = Path("data/eval/generated_testset.json")
PER_BOOK = 18
MIN_CHARS = 400  # skip very short chunks -- not enough content for a good factual question

GENERATION_PROMPT = (
    "ಕೆಳಗಿನ ಕನ್ನಡ ಪಠ್ಯವನ್ನು ಓದಿ. ಈ ಪಠ್ಯದಲ್ಲಿ ಇರುವ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಮಾತ್ರ "
    "ಉತ್ತರಿಸಬಹುದಾದ ಒಂದು ನಿರ್ದಿಷ್ಟ ಸತ್ಯಾಧಾರಿತ ಪ್ರಶ್ನೆಯನ್ನು ಮತ್ತು ಅದರ ಉತ್ತರವನ್ನು "
    "ಬರೆಯಿರಿ. ಸಾಮಾನ್ಯ ಜ್ಞಾನದಿಂದ ಉತ್ತರಿಸಬಹುದಾದ ಪ್ರಶ್ನೆಯನ್ನು ಬರೆಯಬೇಡಿ -- ಈ "
    "ನಿರ್ದಿಷ್ಟ ಪಠ್ಯದಲ್ಲಿ ಇರುವ ವಿವರವನ್ನೇ (ಹೆಸರು, ದಿನಾಂಕ, ಸ್ಥಳ, ಘಟನೆ) ಆಧರಿಸಿರಲಿ.\n\n"
    "ಔಟ್‌ಪುಟ್ ಸ್ವರೂಪ (ಇದನ್ನೇ ನಿಖರವಾಗಿ ಅನುಸರಿಸಿ):\n"
    "ಪ್ರಶ್ನೆ: <ಪ್ರಶ್ನೆ>\n"
    "ಉತ್ತರ: <ಉತ್ತರ>\n\n"
    "ಪಠ್ಯ:\n{text}"
)


def load_chunks(book_id):
    path = CHUNKS_DIR / f"{book_id}.jsonl"
    chunks = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def sample_chunks(book_id, n, seed=42):
    chunks = [c for c in load_chunks(book_id) if c["char_count"] >= MIN_CHARS]
    rng = random.Random(seed)
    rng.shuffle(chunks)

    return chunks[:n]


def _parse_question_answer(raw):
    question = None
    answer = None

    for line in raw.strip().split("\n"):
        line = line.strip()

        if line.startswith("ಪ್ರಶ್ನೆ:"):
            question = line[len("ಪ್ರಶ್ನೆ:"):].strip()
        elif line.startswith("ಉತ್ತರ:"):
            answer = line[len("ಉತ್ತರ:"):].strip()

    return question, answer


def generate_case(chunk):
    prompt = GENERATION_PROMPT.format(text=chunk["text"])
    raw = generate_answer(prompt)
    question, answer = _parse_question_answer(raw)

    if not question or not answer:
        return None

    return {
        "question": question,
        "expected_answer": answer,
        "relevant_chunks": [chunk["chunk_id"]],
        "book_id": chunk["book_id"],
    }


def main():
    book_ids = sorted(p.stem for p in CHUNKS_DIR.glob("*.jsonl"))
    cases = []

    for book_id in book_ids:
        sampled = sample_chunks(book_id, PER_BOOK)
        print(f"{book_id}: generating {len(sampled)} questions...")

        for chunk in sampled:
            case = generate_case(chunk)
            if case:
                cases.append(case)
            else:
                print(f"  skipped {chunk['chunk_id']} (unparseable output)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"\nGenerated {len(cases)} test cases -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
