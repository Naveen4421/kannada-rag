from src.embeddings.model import embed_texts
from src.retrieval.search import search
from src.retrieval.reranker import rerank
from src.retrieval.query_expansion import generate_query_variants
from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer


MAX_ITERATIONS = 3

CRITIQUE_PROMPT = (
    "ಕೆಳಗಿನ ಪ್ರಶ್ನೆ, ಸಂದರ್ಭ ಮತ್ತು ಉತ್ತರವನ್ನು ಪರಿಶೀಲಿಸಿ. ಉತ್ತರವು ಸಂಪೂರ್ಣವಾಗಿ "
    "ಸಂದರ್ಭದ ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ ಇದೆಯೇ ಮತ್ತು ಪ್ರಶ್ನೆಗೆ ನಿಜವಾಗಿಯೂ ಉತ್ತರಿಸುತ್ತದೆಯೇ?\n"
    "ಮೊದಲ ಸಾಲಿನಲ್ಲಿ ಕೇವಲ 'ಹೌದು' ಅಥವಾ 'ಇಲ್ಲ' ಎಂದು ಬರೆಯಿರಿ, ನಂತರದ ಸಾಲಿನಲ್ಲಿ ಒಂದು "
    "ಚಿಕ್ಕ ಕಾರಣ ಬರೆಯಿರಿ.\n\n"
    "ಪ್ರಶ್ನೆ: {question}\n\n"
    "ಸಂದರ್ಭ:\n{context}\n\n"
    "ಉತ್ತರ: {answer}"
)


def critique(question, answer, context_chunks):
    context = "\n\n".join(chunk["text"] for chunk in context_chunks)
    prompt = CRITIQUE_PROMPT.format(question=question, context=context, answer=answer)

    try:
        raw = generate_answer(prompt)
    except Exception:
        # A flaky judge should not turn into an infinite retry loop -- treat
        # an unusable verdict as "grounded" so the loop still terminates.
        return {"grounded": True, "reason": "critique call failed"}

    first_line = raw.strip().split("\n", 1)[0].strip()
    grounded = first_line.startswith("ಹೌದು")
    reason = raw.strip().split("\n", 1)[1].strip() if "\n" in raw.strip() else ""

    return {"grounded": grounded, "reason": reason}


def run_agentic_answer(question, top_k, top_n, where=None):
    iterations = []

    current_where = where
    current_top_k = top_k
    current_top_n = top_n
    current_question = question

    answer = None
    reranked = []
    prompt_text = ""

    for attempt in range(MAX_ITERATIONS):
        question_embedding = embed_texts([current_question])[0]
        retrieved = search(current_question, top_k=current_top_k, where=current_where, question_embedding=question_embedding)
        reranked = rerank(current_question, retrieved, top_n=current_top_n)

        # Always answer the ORIGINAL question, even if this iteration used a
        # rewritten query to fetch different context.
        prompt_text = build_prompt(question, reranked)
        answer = generate_answer(prompt_text)

        verdict = critique(question, answer, reranked)

        iterations.append({
            "iteration": attempt + 1,
            "query_used": current_question,
            "where": current_where,
            "top_k": current_top_k,
            "top_n": current_top_n,
            "retrieved_ids": [c["chunk_id"] for c in retrieved],
            "reranked_ids": [c["chunk_id"] for c in reranked],
            "grounded": verdict["grounded"],
            "reason": verdict["reason"],
        })

        if verdict["grounded"]:
            break

        if attempt == 0:
            # Broaden: search the whole library, not just one book, and
            # widen both the candidate pool and what reaches the LLM --
            # doubling top_k alone can still leave the reranker picking the
            # same top_n chunks it picked before.
            current_where = None
            current_top_k = current_top_k * 2
            current_top_n = current_top_n * 2
        elif attempt == 1:
            variants = generate_query_variants(question, n=1)
            if len(variants) > 1:
                current_question = variants[1]

    return {
        "question": question,
        "answer": answer,
        "reranked": reranked,
        "prompt": prompt_text,
        "iterations": iterations,
        "iteration_count": len(iterations),
        "error": None,
    }
