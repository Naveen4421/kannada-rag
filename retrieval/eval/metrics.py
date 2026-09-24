"""Pure metric functions (no models, no network) so they are unit-tested."""
import math
from collections import defaultdict


def recall_at_k(ranked, relevant, k):
    relevant = set(relevant)
    return len(relevant & set(ranked[:k])) / len(relevant) if relevant else None


def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / k if k else None


def reciprocal_rank(ranked, relevant):
    relevant = set(relevant)
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked, relevant, k):
    """Binary-relevance nDCG@k."""
    relevant = set(relevant)
    if not relevant:
        return None
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def score_ranking(ranked, relevant, ks=(1, 3, 5, 10)):
    row = {"mrr": reciprocal_rank(ranked, relevant)}
    for k in ks:
        row[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        row[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        row[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant, k)
    return row


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(rows_by_category):
    """rows_by_category: {category: [score_ranking dict, ...]} -> macro means
    per category plus 'ALL'. Cases without relevant chunks must not be passed in."""
    out = {}
    everything = []
    for category, rows in rows_by_category.items():
        out[category] = {"n": len(rows), **{m: _mean(r[m] for r in rows) for m in rows[0]}} if rows else {"n": 0}
        everything.extend(rows)
    if everything:
        out["ALL"] = {"n": len(everything), **{m: _mean(r[m] for r in everything) for m in everything[0]}}
    return out


def grounding_summary(groundings):
    """Aggregate grounding results over answered (non-abstained) questions.

    supported/unsupported claim rate are micro-averaged over all claims;
    citation_valid_rate = answers whose cited IDs and pages all exist;
    citation_present_rate = answers that cite at least one evidence ID.
    """
    answered = [g for g in groundings if g and not g.get("abstained")]
    supported = sum(len(g["supported_claims"]) for g in answered)
    unsupported = sum(len(g["unsupported_claims"]) for g in answered)
    claims = supported + unsupported
    n = len(answered)

    return {
        "answered": n,
        "abstained": sum(1 for g in groundings if g and g.get("abstained")),
        "claims": claims,
        "supported_claim_rate": supported / claims if claims else None,
        "unsupported_claim_rate": unsupported / claims if claims else None,
        "grounded_answer_rate": sum(1 for g in answered if g["grounded"]) / n if n else None,
        "citation_valid_rate": sum(1 for g in answered if g["citations_valid"]) / n if n else None,
        "citation_present_rate": sum(1 for g in answered if g.get("citations", {}).get("has_citations")) / n if n else None,
    }


def abstention_summary(cases_and_abstained):
    """[(expected_behavior, abstained_bool)] -> how well the system abstains
    when it should and answers when it should."""
    by = defaultdict(list)
    for expected, abstained in cases_and_abstained:
        by[expected].append(abstained)

    should = by.get("abstain", []) + by.get("clarify_or_abstain", [])
    answerable = by.get("answer", [])
    return {
        "abstain_when_expected": sum(should) / len(should) if should else None,
        "false_abstain_rate": sum(answerable) / len(answerable) if answerable else None,
        "n_should_abstain": len(should),
        "n_answerable": len(answerable),
    }


def latency_summary(samples_ms):
    """{stage: [ms, ...]} -> {stage: {n, mean, p50, p95}}"""
    out = {}
    for stage, values in samples_ms.items():
        values = sorted(values)
        if not values:
            continue
        pick = lambda p: values[min(len(values) - 1, int(round(p * (len(values) - 1))))]
        out[stage] = {"n": len(values), "mean": sum(values) / len(values), "p50": pick(0.5), "p95": pick(0.95)}
    return out
