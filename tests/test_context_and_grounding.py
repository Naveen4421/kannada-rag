import json

from retrieval.evidence.engine import build_evidence
from retrieval.generation.context_builder import (INSUFFICIENT_MARKER, build_context, build_evidence_prompt,
                                             strip_insufficient_marker)
from retrieval.validation.grounding import extract_claims, feedback_lines, validate_grounding
from tests.conftest import make_chunk
from tests.test_evidence import BORN, BORN_B, BORN_WRONG, chunk

Q = "ರಾಜಪುರೋಹಿತರು ಎಲ್ಲಿ ಜನಿಸಿದರು?"


def bundle(*chunks):
    return build_evidence(list(chunks))


def test_context_is_structured_and_keeps_question():
    b = bundle(chunk("A_1", "A", BORN, page=42), chunk("B_1", "B", BORN_B, page=7))
    ctx = build_context(Q, b)
    assert ctx.startswith(f"QUESTION\n{Q}")
    for expected in ("EVIDENCE 1 [E1]", "Book: Title A", "Page: 42", "EVIDENCE 2 [E2]", f"Passage:\n{BORN}"):
        assert expected in ctx


def test_prompt_rules_and_conflict_note_and_feedback():
    b = bundle(chunk("A_1", "A", BORN), chunk("B_1", "B", BORN_WRONG))
    prompt = build_evidence_prompt(Q, b, feedback=["Unsupported by the evidence: x"])
    assert "Do not invent book names" in prompt and INSUFFICIENT_MARKER in prompt
    assert "NOTE: E1 and E2" in prompt and "CORRECTION" in prompt and prompt.rstrip().endswith("(in Kannada):")


def test_marker_stripping():
    assert strip_insufficient_marker(f"{INSUFFICIENT_MARKER} ಮಾಹಿತಿ ಇಲ್ಲ") == ("ಮಾಹಿತಿ ಇಲ್ಲ", True)
    assert strip_insufficient_marker("ಉತ್ತರ") == ("ಉತ್ತರ", False)


def test_claim_extraction_skips_citation_only_fragments():
    claims = extract_claims("ರಾಜಪುರೋಹಿತರು ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1].\n[E1]")
    assert len(claims) == 1


GOOD = "ರಾಜಪುರೋಹಿತರು 1880 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1]."


def test_supported_answer_is_grounded():
    r = validate_grounding(GOOD, bundle(chunk("A_1", "A", BORN, page=42)))
    assert r["grounded"] and r["score"] == 1.0 and r["citations_valid"] and r["method"] == "lexical"


def test_fabricated_citation_and_page_fail():
    b = bundle(chunk("A_1", "A", BORN, page=42))
    r = validate_grounding(GOOD.replace("[E1]", "[E5]") + " page 99", b)
    assert not r["grounded"] and "INVALID_CITATIONS" in r["issues"]
    assert r["citations"]["invalid_ids"] == ["E5"] and r["citations"]["invalid_pages"] == [99]


def test_fabricated_number_fails_even_if_text_overlaps():
    r = validate_grounding("ರಾಜಪುರೋಹಿತರು 1999 ರಲ್ಲಿ ಧಾರವಾಡದಲ್ಲಿ ಜನಿಸಿದರು [E1].", bundle(chunk("A_1", "A", BORN)))
    assert r["unsupported_numbers"] == ["1999"] and not r["grounded"]


def test_unsupported_claim_detected_lexically():
    ans = GOOD + " ಅವರು ಅಮೆರಿಕಾದ ವಿಶ್ವವಿದ್ಯಾಲಯದಲ್ಲಿ ಪ್ರಾಧ್ಯಾಪಕರಾಗಿ ಕೆಲಸ ಮಾಡಿ ನೊಬೆಲ್ ಬಹುಮಾನ ಪಡೆದರು."
    r = validate_grounding(ans, bundle(chunk("A_1", "A", BORN)), min_score=0.9)
    assert len(r["unsupported_claims"]) == 1 and not r["grounded"] and "UNSUPPORTED_CLAIMS" in r["issues"]
    assert any("Unsupported" in line for line in feedback_lines(r))


def test_abstention_is_grounded_but_flagged():
    r = validate_grounding(f"{INSUFFICIENT_MARKER} ಆಧಾರ ಸಾಕಾಗುವುದಿಲ್ಲ", bundle(chunk("A_1", "A", BORN)))
    assert r["grounded"] and r["abstained"]


def test_empty_answer_is_not_grounded():
    r = validate_grounding("  ", bundle(chunk("A_1", "A", BORN)))
    assert not r["grounded"] and r["issues"] == ["NO_CLAIMS"]


def judge_reply(supported, overstated=False):
    return lambda prompt: json.dumps({"claims": [{"id": 1, "supported": supported, "evidence_ids": ["E1"]}],
                                      "overstated": overstated})


def test_llm_judge_overrides_lexical_and_can_flag_overstatement():
    b = bundle(chunk("A_1", "A", BORN))
    r = validate_grounding(GOOD, b, use_llm_judge=True, judge=judge_reply(False))
    assert r["method"] == "llm+lexical" and not r["grounded"] and r["unsupported_claims"][0]["basis"] == "llm"
    r = validate_grounding(GOOD, b, use_llm_judge=True, judge=judge_reply(True, overstated=True))
    assert not r["grounded"] and "OVERSTATED" in r["issues"]


def test_judge_failure_falls_back_to_lexical():
    def bad(prompt):
        return "not json"

    r = validate_grounding(GOOD, bundle(chunk("A_1", "A", BORN)), use_llm_judge=True, judge=bad)
    assert r["method"] == "lexical" and r["judge_error"] and r["grounded"]
