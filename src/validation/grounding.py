"""Grounding validation of a generated answer against the evidence it was given.

Two layers:
1. Deterministic, always on: citation validity (every [E#] and every page
   number must exist in the evidence), fabricated numbers (numbers in the
   answer that occur nowhere in the evidence), and per-claim lexical support
   (char 4-gram containment of the claim in the cited/all evidence).
2. Optional LLM claim judge: per-claim supported/unsupported and an
   'overstated' flag (answer stronger than the evidence allows). If the judge
   fails or is disabled, layer 1 alone decides and `method` says so.

The lexical check is a coarse proxy (it rewards copying and can miss
paraphrase); the LLM judge is the better signal but costs a call and is
itself an LLM. Neither is ground truth.
"""
import json
import re

from src.evidence.agreement import _ngrams
from src.evidence.citations import evidence_pages, to_ascii_digits, validate_citations, extract_page_mentions
from src.generation.context_builder import strip_insufficient_marker

LEXICAL_SUPPORT_THRESHOLD = 0.4
MIN_CLAIM_LETTERS = 12

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_CITATION_TOKEN = re.compile(r"\[[^\[\]]*\]")
_NUMBER = re.compile(r"(?<!\d)\d{2,}(?!\d)")

JUDGE_PROMPT = (
    "You are a strict fact-checking judge. Given EVIDENCE passages and the CLAIMS extracted from an "
    "answer, decide for each claim whether the evidence explicitly supports it. Also decide whether "
    "the answer as a whole states things more strongly or more generally than the evidence allows.\n"
    'Reply with JSON only: {{"claims": [{{"id": 1, "supported": true, "evidence_ids": ["E1"]}}], '
    '"overstated": false}}\n\n'
    "EVIDENCE:\n{evidence}\n\nCLAIMS:\n{claims}"
)


def extract_claims(answer):
    claims = []
    for sentence in _SENTENCE_SPLIT.split(answer or ""):
        text = sentence.strip()
        letters = sum(1 for ch in _CITATION_TOKEN.sub("", text) if ch.isalpha())
        if letters >= MIN_CLAIM_LETTERS:
            claims.append(text)
    return claims


def _cited_in(sentence):
    ids = []
    for group in re.findall(r"\[([^\[\]]*)\]", sentence):
        ids.extend(f"E{n}" for n in re.findall(r"\bE(\d+)\b", group))
    return ids


def _lexical_support(claim, items):
    claim_text = _CITATION_TOKEN.sub(" ", claim)
    grams = _ngrams(claim_text)
    if not grams:
        return 1.0, []

    cited = set(_cited_in(claim))
    pool = [i for i in items if i["evidence_id"] in cited] or items
    evidence_grams = set().union(*(_ngrams(i["text"]) for i in pool)) if pool else set()
    return len(grams & evidence_grams) / len(grams), [i["evidence_id"] for i in pool]


def _unsupported_numbers(answer, items):
    evidence_digits = to_ascii_digits(" ".join(i["text"] for i in items))
    evidence_numbers = set(_NUMBER.findall(evidence_digits))
    # Ignore what the citation validator already covers (page numbers, [E#] ids).
    body = _CITATION_TOKEN.sub(" ", to_ascii_digits(answer))
    page_numbers = {str(p) for p in extract_page_mentions(answer)}
    return sorted({n for n in _NUMBER.findall(body) if n not in evidence_numbers and n not in page_numbers})


def _parse_judge(raw, n_claims):
    match = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge reply")
    data = json.loads(match.group(0))
    verdicts = {int(c["id"]): c for c in data.get("claims", []) if "id" in c}
    if not verdicts:
        raise ValueError("judge returned no claim verdicts")
    return verdicts, bool(data.get("overstated", False))


def _default_judge(prompt):
    from src.generation.llm import generate_answer

    return generate_answer(prompt)


def validate_grounding(answer, bundle, min_score=0.6, use_llm_judge=False, judge=None):
    """-> dict; see module docstring. `answer` is the raw model output
    (the insufficiency marker, if present, is handled here)."""
    items = bundle.items
    clean, abstained = strip_insufficient_marker(answer)

    if abstained:
        return {"grounded": True, "score": 1.0, "abstained": True, "supported_claims": [],
                "unsupported_claims": [], "citations_valid": True, "citations": {}, "unsupported_numbers": [],
                "overstated": False, "method": "abstention", "issues": [], "judge_error": None}

    claims = extract_claims(clean)
    citations = validate_citations(clean, items)
    bad_numbers = _unsupported_numbers(clean, items)
    issues = []

    if not claims:
        return {"grounded": False, "score": 0.0, "abstained": False, "supported_claims": [],
                "unsupported_claims": [], "citations_valid": citations["valid"], "citations": citations,
                "unsupported_numbers": bad_numbers, "overstated": False, "method": "none",
                "issues": ["NO_CLAIMS"], "judge_error": None}

    verdicts, overstated, judge_error, method = None, False, None, "lexical"
    if use_llm_judge and items:
        try:
            prompt = JUDGE_PROMPT.format(
                evidence="\n\n".join(f"[{i['evidence_id']}] {i['text']}" for i in items),
                claims="\n".join(f"{n}. {c}" for n, c in enumerate(claims, start=1)),
            )
            verdicts, overstated = _parse_judge((judge or _default_judge)(prompt), len(claims))
            method = "llm+lexical"
        except Exception as exc:
            judge_error = str(exc)

    supported, unsupported = [], []
    for number, claim in enumerate(claims, start=1):
        lex, pool_ids = _lexical_support(claim, items)
        if verdicts is not None and number in verdicts:
            ok = bool(verdicts[number].get("supported"))
            ids = verdicts[number].get("evidence_ids") or pool_ids
            basis = "llm"
        else:
            ok, ids, basis = lex >= LEXICAL_SUPPORT_THRESHOLD, pool_ids, "lexical"

        entry = {"text": claim, "evidence_ids": ids, "basis": basis, "lexical_overlap": round(lex, 3)}
        (supported if ok else unsupported).append(entry)

    score = len(supported) / len(claims)

    if not citations["valid"]:
        issues.append("INVALID_CITATIONS")
    if bad_numbers:
        issues.append("UNSUPPORTED_NUMBERS")
    if unsupported:
        issues.append("UNSUPPORTED_CLAIMS")
    if overstated:
        issues.append("OVERSTATED")
    if not citations["has_citations"]:
        issues.append("NO_CITATIONS")  # soft: reported, does not fail grounding by itself

    hard = {"INVALID_CITATIONS", "UNSUPPORTED_NUMBERS", "OVERSTATED"}
    grounded = score >= min_score and not (hard & set(issues))

    return {"grounded": grounded, "score": round(score, 3), "abstained": False,
            "supported_claims": supported, "unsupported_claims": unsupported,
            "citations_valid": citations["valid"], "citations": citations,
            "unsupported_numbers": bad_numbers, "overstated": overstated, "method": method,
            "issues": issues, "judge_error": judge_error}


def feedback_lines(result):
    """Human-readable problems for the agent's corrective re-generation."""
    lines = []
    for claim in result.get("unsupported_claims", [])[:5]:
        lines.append(f"Unsupported by the evidence: {claim['text'][:200]}")
    if result.get("unsupported_numbers"):
        lines.append("These numbers do not appear in the evidence: " + ", ".join(result["unsupported_numbers"]))
    cites = result.get("citations", {})
    if cites.get("invalid_ids"):
        lines.append("Cited evidence IDs that do not exist: " + ", ".join(cites["invalid_ids"]))
    if cites.get("invalid_pages"):
        lines.append("Page numbers not present in the evidence: " + ", ".join(map(str, cites["invalid_pages"])))
    if result.get("overstated"):
        lines.append("The answer claims more than the evidence supports; soften or remove it.")
    return lines
