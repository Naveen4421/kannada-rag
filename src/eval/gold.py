"""Gold test set loader and mechanical verifier.

    python -m src.eval.gold        # re-check every mechanical claim against data/chunks

The verifier only checks what can be checked by string/number comparison
(chunk exists, required strings occur in the chunk, page ranges cover the
page, 'absent' terms occur nowhere, no other chunk contains a 'rare' word).
Semantic correctness of a question/answer pair still needs a human; each
case's `verification` field says which kind of check backs it.
"""
import glob
import json
from pathlib import Path

GOLD_PATH = Path(__file__).parent / "gold" / "kannada_gold.json"
CHUNKS_GLOB = "data/chunks/*.jsonl"


def load_gold(path=GOLD_PATH, categories=None):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if not categories or c["category"] in categories]


def load_chunks(pattern=CHUNKS_GLOB):
    chunks = {}
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunk = json.loads(line)
                    chunks[chunk["chunk_id"]] = chunk
    return chunks


def verify(cases, chunks):
    """-> list of problem strings (empty = all mechanical claims hold)."""
    problems = []

    for case in cases:
        cid = case["id"]

        for chunk_id in case.get("relevant_chunks", []):
            chunk = chunks.get(chunk_id)
            if chunk is None:
                problems.append(f"{cid}: relevant chunk {chunk_id} does not exist")
                continue
            for needle in case.get("must_contain", []):
                if needle not in chunk["text"]:
                    problems.append(f"{cid}: {chunk_id} does not contain {needle!r}")
            if case.get("book_id") and chunk["book_id"] != case["book_id"]:
                problems.append(f"{cid}: {chunk_id} is not in book {case['book_id']}")

        for term in case.get("corpus_absent_terms", []):
            found = [i for i, c in chunks.items() if term in c["text"]]
            if found:
                problems.append(f"{cid}: term {term!r} is present in {found[:3]}")

        if case["category"] in ("rare_word", "exact_word"):
            needle = case["must_contain"][0]
            expected = {i for i, c in chunks.items() if needle in c["text"]}
            if expected != set(case["relevant_chunks"]):
                problems.append(f"{cid}: chunks containing {needle!r} are {sorted(expected)}, "
                                f"labelled {sorted(case['relevant_chunks'])}")

        if case["category"] == "page_specific":
            expected = {i for i, c in chunks.items()
                        if c["book_id"] == case["book_id"] and c["page_start"] <= case["page"] <= c["page_end"]}
            if expected != set(case["relevant_chunks"]):
                problems.append(f"{cid}: page {case['page']} is covered by {sorted(expected)}, "
                                f"labelled {sorted(case['relevant_chunks'])}")

    return problems


def main():
    cases = load_gold()
    chunks = load_chunks()
    problems = verify(cases, chunks)

    by_kind = {}
    for case in cases:
        by_kind.setdefault(case["category"], 0)
        by_kind[case["category"]] += 1

    print(f"{len(cases)} cases: {by_kind}")
    for p in problems:
        print("PROBLEM:", p)
    print("mechanical verification: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
