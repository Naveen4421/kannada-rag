from common.llm import generate_answer
from retrieval.retrieve.search import search


EXPANSION_PROMPT = (
    "ಈ ಕೆಳಗಿನ ಕನ್ನಡ ಪ್ರಶ್ನೆಗೆ {n} ಪರ್ಯಾಯ ರೂಪಗಳನ್ನು ಬರೆಯಿರಿ — ಅದೇ ಅರ್ಥವನ್ನು "
    "ಬೇರೆ ಬೇರೆ ಪದಗಳಲ್ಲಿ ಕೇಳುವ ಪ್ರಶ್ನೆಗಳು. ಪ್ರತಿ ಸಾಲಿನಲ್ಲಿ ಒಂದು ಪ್ರಶ್ನೆ ಮಾತ್ರ ಬರೆಯಿರಿ, "
    "ಸಂಖ್ಯೆ ಅಥವಾ ಬೇರೆ ಯಾವುದೇ ಪಠ್ಯ ಸೇರಿಸಬೇಡಿ.\n\n"
    "ಮೂಲ ಪ್ರಶ್ನೆ: {question}"
)


def _parse_variants(raw, n):
    lines = [line.strip(" -*0123456789.\t") for line in raw.strip().split("\n")]
    lines = [line for line in lines if line]
    return lines[:n]


def generate_query_variants(question, n=3):
    prompt = EXPANSION_PROMPT.format(n=n, question=question)
    raw = generate_answer(prompt)
    variants = _parse_variants(raw, n)

    return [question] + variants


def rrf_merge(result_lists, k=60):
    scores = {}
    chunks_by_id = {}

    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = chunk

    ordered_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

    merged = []
    for chunk_id in ordered_ids:
        chunk = dict(chunks_by_id[chunk_id])
        chunk["score"] = scores[chunk_id]
        merged.append(chunk)

    return merged


def expand_and_retrieve(question, top_k, where=None):
    variants = generate_query_variants(question)
    result_lists = [search(variant, top_k=top_k, where=where) for variant in variants]

    return rrf_merge(result_lists)[:top_k]
