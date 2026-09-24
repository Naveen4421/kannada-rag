"""Query entry points.

    ask()             UI dispatcher: process -> route -> cache -> (agent | fast streaming path)
    answer_question() single-shot, non-streaming (evaluation, CLI)

Both run the same evidence pipeline (rag_core.retrieve_evidence ->
context_builder -> LLM -> grounding). Failures become structured results:
`error` (user-facing string, as before) plus `error_detail`
({"stage", "code", "message"}). A retrieval failure is never turned into an
LLM answer without evidence.
"""
import argparse
import time

from retrieval import config as cfg
from retrieval.cache import store
from retrieval.cache.fingerprint import fingerprint
from retrieval.generation.context_builder import build_evidence_prompt, strip_insufficient_marker, INSUFFICIENT_MARKER
from common.llm import generate_answer_stream
from retrieval.observability.trace import QueryTrace
from retrieval.pipeline.agent import NO_EVIDENCE_ANSWER, UNGROUNDED_CAVEAT, default_generate, run_agentic_answer
from retrieval.pipeline.errors import StageError
from retrieval.pipeline.rag_core import retrieve_evidence
from retrieval.query_processing.processor import process_query
from retrieval.routing.router import classify
from retrieval.validation.grounding import validate_grounding
from retrieval.evidence.engine import NO_EVIDENCE


def _error_result(question, route, stage_error, trace=None):
    """`stage_error` is a StageError or a plain message (treated as an
    unclassified failure)."""
    if not isinstance(stage_error, StageError):
        stage_error = StageError("unknown", "UNEXPECTED_ERROR", str(stage_error))

    detail = stage_error.to_dict()
    label = {"retrieval": "Evidence retrieval failed", "routing": "Query routing failed",
             "generation": "Answer generation failed"}.get(detail["stage"], f"{detail['stage']} failed")
    message = f"{label} ({detail['code']}): {detail['message']}"

    if detail["stage"] == "retrieval":
        message += ". No answer was generated because no evidence could be retrieved."

    if trace:
        trace.error(detail)
        trace.set(route=route, cache_hit=False)
        trace.finish()

    return {
        "success": False,
        "question": question,
        "retrieved": [],
        "reranked": [],
        "evidence": [],
        "prompt": "",
        "route": route,
        "answer": None,
        "answer_stream": None,
        "cache_hit": False,
        "error": message,
        "error_detail": detail,
    }


def _no_evidence_result(question, stage, route):
    return {
        "success": True, "question": question, "retrieved": stage.retrieved, "reranked": stage.reranked,
        "evidence": [], "prompt": "", "route": route, "answer": NO_EVIDENCE_ANSWER,
        "grounding": {"grounded": True, "score": 1.0, "abstained": True, "method": "no_evidence",
                      "supported_claims": [], "unsupported_claims": [], "citations_valid": True, "issues": []},
        "answer_stream": None, "degraded": stage.degraded, "error": None,
    }


def answer_question(question, top_k=10, top_n=5, where=None, question_embedding=None):
    """Single-shot evidence RAG (no routing, cache or retries)."""
    trace = QueryTrace(question)
    pq = process_query(question, where)
    trace.set(normalized_query=pq.normalized_query, query_processing=pq.to_dict(), route="single_shot")

    try:
        stage = retrieve_evidence(pq.normalized_query, top_k, top_n, where=where,
                                  question_embedding=question_embedding, trace=trace)
        if stage.bundle.sufficiency["reason"] == NO_EVIDENCE:
            result = _no_evidence_result(question, stage, "single_shot")
        else:
            prompt_text = build_evidence_prompt(question, stage.bundle)
            raw = default_generate(prompt_text, trace)
            grounding = validate_grounding(raw, stage.bundle, use_llm_judge=False)
            clean, _ = strip_insufficient_marker(raw)
            result = {
                "success": True, "question": question, "retrieved": stage.retrieved, "reranked": stage.reranked,
                "evidence": stage.bundle.items, "prompt": prompt_text, "answer": clean,
                "grounding": grounding, "degraded": stage.degraded, "error": None,
            }
        trace.set(cache_hit=False)
        trace.finish()
        return result
    except StageError as exc:
        return _error_result(question, "single_shot", exc, trace)


def _strip_marker_stream(pieces):
    """Drops a leading INSUFFICIENT_MARKER from a token stream (the marker is
    a machine signal, not text for the reader). Yields (piece, abstained_flag_box)."""
    box = {"abstained": False}
    buffer, decided = "", False

    def gen():
        nonlocal buffer, decided
        for piece in pieces:
            if decided:
                yield piece
                continue
            buffer += piece
            if len(buffer.lstrip()) >= len(INSUFFICIENT_MARKER) or not INSUFFICIENT_MARKER.startswith(buffer.lstrip()):
                decided = True
                text, box["abstained"] = strip_insufficient_marker(buffer)
                buffer = ""
                if text:
                    yield text
        if not decided and buffer:
            text, box["abstained"] = strip_insufficient_marker(buffer)
            yield text

    return gen(), box


def _finalize_stream(stream_gen, marker_box, cache_key, fp, question, route, base_payload, state, trace, stage):
    collected = []

    try:
        for piece in stream_gen:
            collected.append(piece)
            yield piece
    except Exception as exc:
        state["error"] = f"Answer generation failed (LLM_FAILED): {exc}"
        trace.error({"stage": "generation", "code": "LLM_FAILED", "message": str(exc)})

    answer = "".join(collected) if state["error"] is None else None
    grounding = None

    if answer is not None:
        # Deterministic checks only: the text is already on screen, and the
        # fast path has no retry, so no extra LLM judge call.
        raw = (INSUFFICIENT_MARKER + " " + answer) if marker_box["abstained"] else answer
        grounding = validate_grounding(raw, stage.bundle, min_score=cfg.AGENT.min_grounding_score, use_llm_judge=False)
        state["grounding"] = grounding
        state["answer"] = answer

        if not grounding["grounded"]:
            state["warning"] = UNGROUNDED_CAVEAT.strip()

    trace.section("validation", grounded=grounding and grounding["grounded"], score=grounding and grounding["score"],
                  unsupported_claims=[c["text"][:120] for c in (grounding or {}).get("unsupported_claims", [])],
                  citations_valid=grounding and grounding["citations_valid"], issues=(grounding or {}).get("issues"),
                  method=grounding and grounding["method"])
    trace.set(cache_hit=False)

    if state["error"] is None and grounding["grounded"] and not stage.degraded:
        store.put(cache_key, question, route, {**base_payload, "answer": answer, "grounding": grounding, "error": None},
                  fingerprint=fp)

    trace.finish()


def ask(question, where=None):
    """Dispatcher: process -> route -> cache -> (agentic loop | fast streaming path).

    The question is embedded once (normalised text) and reused for routing and
    the first retrieval. Every stage is guarded so a failure (e.g. GPU OOM)
    yields a structured error result rather than an exception in the UI.
    """
    trace = QueryTrace(question)
    pq = process_query(question, where)
    trace.set(normalized_query=pq.normalized_query, query_processing=pq.to_dict())

    try:
        from common.embedding_model import embed_texts

        question_embedding = embed_texts([pq.normalized_query])[0]
        route = classify(pq.normalized_query, question_embedding)
    except Exception as exc:
        return _error_result(question, "unknown", StageError("routing", "ROUTING_FAILED", str(exc)), trace)

    trace.set(route=route.route, routing=route.diagnostics())

    cache_key = store.make_key(question, pq.book_filter, route.top_k, route.top_n, route.route)
    fp = fingerprint(route.route)

    cached = store.get(cache_key, fp)
    if cached is not None:
        trace.set(cache_hit=True)
        trace.section("evidence", **{k: v for k, v in cached.get("evidence_summary", {}).items()})
        trace.finish()
        return {**cached, "cache_hit": True, "answer_stream": None}

    if route.use_self_critique:
        try:
            result = run_agentic_answer(question, top_k=route.top_k, top_n=route.top_n, where=where,
                                        normalized_query=pq.normalized_query, trace=trace,
                                        question_embedding=question_embedding)
        except StageError as exc:
            return _error_result(question, route.route, exc, trace)
        except Exception as exc:
            return _error_result(question, route.route, StageError("agent", "UNEXPECTED_ERROR", str(exc)), trace)

        payload = {**result, "success": True, "route": route.route, "routing": route.diagnostics()}

        if result["grounding"]["grounded"] and not result["degraded"]:
            store.put(cache_key, question, route.route, payload, fingerprint=fp)

        trace.set(cache_hit=False)
        trace.finish()

        return {**payload, "cache_hit": False, "answer_stream": None}

    # Fast path: the final generation call streams live.
    try:
        stage = retrieve_evidence(pq.normalized_query, route.top_k, route.top_n, where=where,
                                  question_embedding=question_embedding, trace=trace)
    except StageError as exc:
        return _error_result(question, route.route, exc, trace)
    except Exception as exc:
        return _error_result(question, route.route, StageError("retrieval", "UNEXPECTED_ERROR", str(exc)), trace)

    if stage.bundle.sufficiency["reason"] == NO_EVIDENCE:
        result = {**_no_evidence_result(question, stage, route.route), "cache_hit": False}
        trace.set(cache_hit=False)
        trace.finish()
        return result

    prompt_text = build_evidence_prompt(question, stage.bundle)
    base_payload = {
        "success": True, "question": question, "retrieved": stage.retrieved, "reranked": stage.reranked,
        "evidence": stage.bundle.items, "evidence_summary": stage.bundle.summary(),
        "prompt": prompt_text, "route": route.route, "routing": route.diagnostics(), "degraded": stage.degraded,
    }

    # The stream is lazy: an LLM error only surfaces once the UI consumes it.
    # `state` is a mutable box shared with the generator; the caller reads
    # state["error"] / state["grounding"] AFTER consuming answer_stream.
    # (Exposed as "stream_error" for compatibility with the existing UI.)
    state = {"error": None, "grounding": None, "answer": None, "warning": None}
    stream, marker_box = _strip_marker_stream(generate_answer_stream(prompt_text))
    final = _finalize_stream(stream, marker_box, cache_key, fp, question, route.route, base_payload, state, trace, stage)

    return {
        **base_payload,
        "answer": None,
        "answer_stream": final,
        "stream_error": state,
        "stream_state": state,
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
        print(f"{rank}. {chunk['chunk_id']}  rrf={chunk['rrf_score']:.4f}  "
              f"dense#{chunk['dense_rank']} sparse#{chunk['sparse_rank']}")

    print()
    print("=" * 70)
    print(f"Evidence (top {args.top_n}):")
    print("=" * 70)
    for item in result.get("evidence", []):
        print(f"{item['evidence_id']}. {item['chunk_id']}  {item['book_name']} p.{item['page']}  "
              f"evidence={item['evidence_score']:.3f} {item['components']}")

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
        grounding = result.get("grounding")
        if grounding:
            print(f"\n[grounded={grounding['grounded']} score={grounding['score']} issues={grounding['issues']}]")


if __name__ == "__main__":
    main()
