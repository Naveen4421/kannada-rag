"""Evidence-driven agentic loop for the complex route.

    retrieve -> rerank -> evidence -> sufficiency check -> generate
             -> grounding validation -> (retry only if there is a reason)

A retry always has a logged reason and a concrete change; the loop is bounded
by AGENT.max_iterations:

  retry reason                          action on the next iteration
  NO_EVIDENCE / WEAK_EVIDENCE /         1st: broaden (drop book filter, double pool)
  LOW_RERANK_CONFIDENCE                 2nd: retrieve with an LLM query variant
  UNSUPPORTED_CLAIMS / INVALID_          1st: regenerate on the SAME evidence with the
  CITATIONS / UNSUPPORTED_NUMBERS /     problems listed as feedback; then as above
  OVERSTATED

Conflicting evidence is not a retry reason: more retrieval does not resolve
disagreement between books, so it is passed to the generator (which must
report both versions) and surfaced in the result.

If no evidence exists after the last retrieval the LLM is not called.
"""
import time

from src import config as cfg
from src.evidence.engine import NO_EVIDENCE
from src.generation.context_builder import build_evidence_prompt, strip_insufficient_marker
from src.pipeline.errors import StageError
from src.pipeline.rag_core import retrieve_evidence
from src.validation.grounding import feedback_lines, validate_grounding

MAX_TOP_K, MAX_TOP_N = 60, 12

NO_EVIDENCE_ANSWER = "ಈ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ಲಭ್ಯವಿರುವ ಪುಸ್ತಕಗಳಲ್ಲಿ ಸಂಬಂಧಿತ ಆಧಾರ ಕಂಡುಬಂದಿಲ್ಲ."
# "Note: some parts of this answer are not fully supported by the evidence provided."
UNGROUNDED_CAVEAT = "\n\nಗಮನಿಸಿ: ಈ ಉತ್ತರದ ಕೆಲವು ಅಂಶಗಳನ್ನು ಒದಗಿಸಿದ ಆಧಾರಗಳು ಸಂಪೂರ್ಣವಾಗಿ ಬೆಂಬಲಿಸುವುದಿಲ್ಲ."

_ISSUE_TO_REASON = [("INVALID_CITATIONS", "INVALID_CITATIONS"), ("UNSUPPORTED_NUMBERS", "UNSUPPORTED_NUMBERS"),
                    ("OVERSTATED", "OVERSTATED"), ("UNSUPPORTED_CLAIMS", "UNSUPPORTED_CLAIMS")]


def default_generate(prompt, trace=None):
    from src.generation.llm import MODEL_NAME, generate_answer_with_usage

    started = time.perf_counter()
    try:
        result = generate_answer_with_usage(prompt)
    except Exception as exc:
        raise StageError("generation", "LLM_FAILED", f"LLM call failed: {exc}") from exc

    if trace:
        gen = trace.data.setdefault("generation", {"model": MODEL_NAME, "latency_ms": 0.0, "usage": []})
        gen["latency_ms"] += (time.perf_counter() - started) * 1000
        gen["usage"].append(result.get("usage", {}))

    return result["text"]


def _default_variants(question):
    from src.retrieval.query_expansion import generate_query_variants

    return generate_query_variants(question, n=1)


def grounding_reason(result):
    for issue, reason in _ISSUE_TO_REASON:
        if issue in result["issues"]:
            return reason
    return "NOT_GROUNDED"


def run_agentic_answer(question, top_k, top_n, where=None, normalized_query=None, trace=None, config=None,
                       generate_fn=None, judge=None, variants_fn=None, retrieve_fn=retrieve_evidence,
                       question_embedding=None):
    config = config or cfg.AGENT
    generate_fn = generate_fn or (lambda prompt: default_generate(prompt, trace))
    variants_fn = variants_fn or _default_variants
    query_text = normalized_query or question

    state = {"where": where, "top_k": top_k, "top_n": top_n, "query": query_text, "embedding": question_embedding}
    broaden_step = 0
    iterations = []
    stage = answer = grounding = prompt_text = None
    retry_feedback = None
    reuse_stage = False
    final_answer = None

    def next_retrieval_plan():
        """Advance the retrieval strategy; False when nothing new is left."""
        nonlocal broaden_step
        while broaden_step < 2:
            step, broaden_step = broaden_step, broaden_step + 1
            if step == 0:
                new_k, new_n = min(state["top_k"] * 2, MAX_TOP_K), min(state["top_n"] * 2, MAX_TOP_N)
                changed = state["where"] is not None or (new_k, new_n) != (state["top_k"], state["top_n"])
                if changed:
                    state.update(where=None, top_k=new_k, top_n=new_n)
                    return "broaden"
            else:
                try:
                    variants = variants_fn(question)
                except Exception:
                    return False
                if len(variants) > 1 and variants[1].strip() and variants[1] != state["query"]:
                    state.update(query=variants[1], embedding=None)
                    return "query_variant"
        return False

    for attempt in range(1, config.max_iterations + 1):
        is_last = attempt == config.max_iterations

        if not reuse_stage:
            stage = retrieve_fn(state["query"], top_k=state["top_k"], top_n=state["top_n"], where=state["where"],
                                question_embedding=state["embedding"], trace=trace)
        reuse_stage = False

        record = {
            "iteration": attempt, "query_used": state["query"], "where": state["where"],
            "top_k": state["top_k"], "top_n": state["top_n"],
            "retrieved_ids": [c["chunk_id"] for c in stage.retrieved],
            "reranked_ids": [c["chunk_id"] for c in stage.reranked],
            "sufficiency": stage.bundle.sufficiency, "generated": False,
            "grounded": None, "grounding_score": None, "retry_reason": None, "retry_action": None,
        }
        iterations.append(record)
        sufficiency = stage.bundle.sufficiency

        if not sufficiency["sufficient"]:
            action = None if is_last else next_retrieval_plan()
            if action:
                record.update(retry_reason=sufficiency["reason"], retry_action=action)
                continue
            if sufficiency["reason"] == NO_EVIDENCE:
                final_answer = NO_EVIDENCE_ANSWER
                grounding = {"grounded": True, "score": 1.0, "abstained": True, "supported_claims": [],
                             "unsupported_claims": [], "citations_valid": True, "issues": [],
                             "method": "no_evidence", "citations": {}, "unsupported_numbers": [],
                             "overstated": False, "judge_error": None}
                break
            # Weak evidence and nothing new to try: answer anyway; the prompt
            # tells the model to abstain if the evidence does not suffice.

        prompt_text = build_evidence_prompt(question, stage.bundle, feedback=retry_feedback)
        raw = generate_fn(prompt_text)
        retry_feedback = None
        record["generated"] = True

        grounding = validate_grounding(raw, stage.bundle, min_score=config.min_grounding_score,
                                       use_llm_judge=config.use_llm_judge, judge=judge)
        answer = raw
        record.update(grounded=grounding["grounded"], grounding_score=grounding["score"])

        if grounding["grounded"]:
            break

        reason = grounding_reason(grounding)
        record["failure_reason"] = reason
        if is_last:
            break

        if not any(i.get("retry_action") == "regenerate" for i in iterations):
            retry_feedback = feedback_lines(grounding)
            record.update(retry_reason=reason, retry_action="regenerate")
            reuse_stage = True
            continue

        action = next_retrieval_plan()
        if not action:
            record["retry_action"] = "none_available"
            break
        record.update(retry_reason=reason, retry_action=action)

    if final_answer is None:
        clean, abstained = strip_insufficient_marker(answer)
        final_answer = clean if grounding["grounded"] else clean + UNGROUNDED_CAVEAT
        grounding["abstained"] = abstained or grounding.get("abstained", False)

    if trace:
        trace.data["agent"]["iterations"] = iterations
        trace.section("validation", grounded=grounding["grounded"], score=grounding["score"],
                      unsupported_claims=[c["text"][:120] for c in grounding["unsupported_claims"]],
                      citations_valid=grounding["citations_valid"], issues=grounding["issues"],
                      method=grounding["method"])

    return {
        "question": question,
        "answer": final_answer,
        "retrieved": stage.retrieved,
        "reranked": stage.reranked,
        "evidence": stage.bundle.items,
        "evidence_summary": stage.bundle.summary(),
        "grounding": grounding,
        "prompt": prompt_text or "",
        "iterations": iterations,
        "iteration_count": len(iterations),
        "warnings": stage.warnings,
        "degraded": stage.degraded,
        "error": None,
    }
