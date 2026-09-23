import argparse
import time

from src.embeddings.model import embed_texts
from src.retrieval.search import search
from src.retrieval.reranker import rerank
from src.generation.prompt import build_prompt
from src.generation.llm import generate_answer, generate_answer_stream
from src.routing.router import classify
from src.pipeline.agent import run_agentic_answer
from src.cache import store
from src.observability.logger import log_query


def answer_question(question, top_k=10, top_n=5, where=None, question_embedding=None):
    retrieved = search(question, top_k=top_k, where=where, question_embedding=question_embedding)
    reranked = rerank(question, retrieved, top_n=top_n)
    prompt_text = build_prompt(question, reranked)

    answer = None
    error = None

    try:
        answer = generate_answer(prompt_text)
    except NotImplementedError as exc:
        error = str(exc)
    except Exception as exc:
        error = f"LLM call failed: {exc}"

    return {
        "question": question,
        "retrieved": retrieved,
        "reranked": reranked,
        "prompt": prompt_text,
        "answer": answer,
        "error": error,
    }


def _finalize_stream(stream_gen, cache_key, question, route, base_payload, error_holder):
    collected = []

    try:
        for piece in stream_gen:
            collected.append(piece)
            yield piece
    except Exception as exc:
        error_holder["error"] = f"LLM call failed: {exc}"

    error = error_holder["error"]
    latency_ms = (time.perf_counter() - base_payload["_started"]) * 1000
    answer = "".join(collected) if error is None else None

    payload = {**base_payload, "answer": answer, "error": error}
    payload.pop("_started", None)

    if error is None:
        store.put(cache_key, question, route, payload)

    log_query({
        "question": question,
        "route": route,
        "reranked": [{"chunk_id": c["chunk_id"]} for c in base_payload["reranked"]],
        "latency_ms": {"total": latency_ms},
        "cache_hit": False,
        "error": error,
    })


def _error_result(question, route, message):
    return {
        "question": question,
        "retrieved": [],
        "reranked": [],
        "prompt": "",
        "route": route,
        "answer": None,
        "answer_stream": None,
        "cache_hit": False,
        "error": message,
    }


def ask(question, where=None):
    """
    Dispatcher: route -> cache -> (agentic loop | fast streaming path).
    Computes the question embedding once and reuses it for both routing and
    the fast-path search (the agentic loop re-embeds per iteration since its
    query text can change).

    Every stage (embedding, routing, retrieval, reranking -- not just the
    final LLM call) is guarded: a failure anywhere (e.g. GPU OOM on a shared
    machine) degrades to a clean error result instead of an unhandled
    exception reaching the UI.
    """
    started = time.perf_counter()

    try:
        question_embedding = embed_texts([question])[0]
        route_decision = classify(question, question_embedding)
    except Exception as exc:
        return _error_result(question, "unknown", f"Routing failed: {exc}")

    book_id = where.get("book_id") if where else None
    cache_key = store.make_key(question, book_id, route_decision.top_k, route_decision.top_n, route_decision.route)

    cached = store.get(cache_key)
    if cached is not None:
        log_query({
            "question": question,
            "route": route_decision.route,
            "reranked": [{"chunk_id": c["chunk_id"]} for c in cached.get("reranked", [])],
            "latency_ms": {"total": (time.perf_counter() - started) * 1000},
            "cache_hit": True,
            "error": cached.get("error"),
        })
        return {**cached, "cache_hit": True, "answer_stream": None}

    if route_decision.use_self_critique:
        try:
            result = run_agentic_answer(question, top_k=route_decision.top_k, top_n=route_decision.top_n, where=where)
            error = result["error"]
        except Exception as exc:
            result = {"question": question, "answer": None, "reranked": [], "prompt": "", "iterations": [], "iteration_count": 0}
            error = f"LLM call failed: {exc}"

        payload = {**result, "error": error, "route": route_decision.route}

        if error is None:
            store.put(cache_key, question, route_decision.route, payload)

        log_query({
            "question": question,
            "route": route_decision.route,
            "iteration_count": payload.get("iteration_count"),
            "reranked": [{"chunk_id": c["chunk_id"]} for c in payload.get("reranked", [])],
            "latency_ms": {"total": (time.perf_counter() - started) * 1000},
            "cache_hit": False,
            "error": error,
        })

        return {**payload, "cache_hit": False, "answer_stream": None}

    # Fast path: no critique, so the final generation call can stream live.
    try:
        retrieved = search(question, top_k=route_decision.top_k, where=where, question_embedding=question_embedding)
        reranked = rerank(question, retrieved, top_n=route_decision.top_n)
        prompt_text = build_prompt(question, reranked)
    except Exception as exc:
        error = f"Retrieval failed: {exc}"
        log_query({
            "question": question,
            "route": route_decision.route,
            "latency_ms": {"total": (time.perf_counter() - started) * 1000},
            "cache_hit": False,
            "error": error,
        })
        return _error_result(question, route_decision.route, error)

    base_payload = {
        "question": question,
        "retrieved": retrieved,
        "reranked": reranked,
        "prompt": prompt_text,
        "route": route_decision.route,
        "_started": started,
    }

    # The stream is lazy -- any LLM error (e.g. no credits) only surfaces once
    # app.py actually consumes it. error_holder is a mutable box both this
    # dict and the generator share, so the caller can check it *after*
    # consuming answer_stream (e.g. via st.write_stream) to display the error.
    error_holder = {"error": None}
    stream = _finalize_stream(generate_answer_stream(prompt_text), cache_key, question, route_decision.route, base_payload, error_holder)

    return {
        "question": question,
        "retrieved": retrieved,
        "reranked": reranked,
        "prompt": prompt_text,
        "route": route_decision.route,
        "answer": None,
        "answer_stream": stream,
        "stream_error": error_holder,
        "cache_hit": False,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="Ask a Kannada question against the indexed book collection.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    result = answer_question(args.question, top_k=args.top_k, top_n=args.top_n)

    print("=" * 70)
    print(f"Retrieved (top {args.top_k}):")
    print("=" * 70)
    for rank, chunk in enumerate(result["retrieved"], start=1):
        print(f"{rank}. {chunk['chunk_id']}  score={chunk['score']:.4f}")

    print()
    print("=" * 70)
    print(f"Reranked (top {args.top_n}):")
    print("=" * 70)
    for rank, chunk in enumerate(result["reranked"], start=1):
        print(f"{rank}. {chunk['chunk_id']}  score={chunk['score']:.4f}")

    print()
    print("=" * 70)
    print("Prompt sent to LLM:")
    print("=" * 70)
    print(result["prompt"])

    print()
    print("=" * 70)
    print("Answer:")
    print("=" * 70)
    if result["error"]:
        print(f"[{result['error']}]")
    else:
        print(result["answer"])


if __name__ == "__main__":
    main()
