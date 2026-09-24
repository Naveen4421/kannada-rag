"""Cross-source agreement / conflict analysis. Heuristic and lexical.

There is no NLI model here, so "supports the same information" is
approximated and every rule is deliberately simple and inspectable:

- Independence is by book_id. Several passages of one book are reported as
  `same_book_passages`, never as agreement.
- Two passages from DIFFERENT books that overlap topically (char 4-gram
  Jaccard >= topical_overlap) corroborate each other.
- Near-identical text (>= near_duplicate) in the same book is a duplicate and
  supports nothing; across books it still counts but is flagged
  `near_duplicate_of` (the books may share a source).
- Possible conflict: topically overlapping passages from different books that
  each contain 3-4 digit numbers (mostly years) and share none of them.

The thresholds are uncalibrated. A stronger judge (NLI/LLM) can replace
`_relation` without touching callers.
"""
import re
import unicodedata

from src import config as cfg
from src.evidence.citations import to_ascii_digits

_NUMBER = re.compile(r"(?<!\d)\d{3,4}(?!\d)")


def _ngrams(text, n=4):
    kept = "".join(
        ch for ch in unicodedata.normalize("NFC", to_ascii_digits(text)).lower()
        if unicodedata.category(ch)[0] in ("L", "M", "N")
    )
    return {kept[i:i + n] for i in range(max(0, len(kept) - n + 1))}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _numbers(text):
    return set(_NUMBER.findall(to_ascii_digits(text)))


def analyze(items, config=None):
    config = config or cfg.EVIDENCE
    grams = [_ngrams(i["text"]) for i in items]
    nums = [_numbers(i["text"]) for i in items]

    per_item = {
        i["evidence_id"]: {
            "independent_sources": [], "same_book_passages": 0,
            "near_duplicate_of": [], "conflicts_with": [],
        }
        for i in items
    }
    conflicts = []

    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ia, ib = items[a], items[b]
            ea, eb = ia["evidence_id"], ib["evidence_id"]
            sim = _jaccard(grams[a], grams[b])
            same_book = ia["book_id"] == ib["book_id"]

            if same_book:
                if sim >= config.near_duplicate:
                    per_item[ea]["near_duplicate_of"].append(eb)
                    per_item[eb]["near_duplicate_of"].append(ea)
                else:
                    per_item[ea]["same_book_passages"] += 1
                    per_item[eb]["same_book_passages"] += 1
                continue

            if sim < config.topical_overlap:
                continue

            if sim >= config.near_duplicate:
                per_item[ea]["near_duplicate_of"].append(eb)
                per_item[eb]["near_duplicate_of"].append(ea)

            if nums[a] and nums[b] and not (nums[a] & nums[b]):
                per_item[ea]["conflicts_with"].append(eb)
                per_item[eb]["conflicts_with"].append(ea)
                conflicts.append({"evidence_ids": [ea, eb], "books": [ia["book_id"], ib["book_id"]],
                                  "kind": "numeric", "values": [sorted(nums[a]), sorted(nums[b])]})
            else:
                for src, dst in ((ea, ib), (eb, ia)):
                    if dst["book_id"] not in per_item[src]["independent_sources"]:
                        per_item[src]["independent_sources"].append(dst["book_id"])

    if not items:
        status = "none"
    elif conflicts:
        status = "conflicting"
    elif any(v["independent_sources"] for v in per_item.values()):
        status = "corroborated"
    else:
        status = "single_source"

    return {
        "status": status,
        "per_item": per_item,
        "conflicts": conflicts,
        "books": sorted({i["book_id"] for i in items}),
    }


def agreement_component(info):
    """0..1 agreement component for one item. 0.5 = no corroboration
    observed (neutral, not bad); grows with the number of independent
    supporting books; a conflict pulls it below neutral."""
    if info["conflicts_with"]:
        return 0.2
    n = len(info["independent_sources"])
    return 0.5 + 0.5 * n / (n + 1)
