"""Evidence Engine: reranked chunks -> structured, scored, cited evidence.

The LLM never sees raw chunks and never supplies metadata: book, page, chunk
and IDs are copied from the retrieved payload here.
"""
from dataclasses import dataclass, field

from retrieval import config as cfg
from retrieval.evidence import agreement as agreement_mod
from retrieval.evidence import scorer
from retrieval.evidence.citations import make_evidence_id, page_label

# Sufficiency reason codes (consumed by the agent to decide whether to retry).
NO_EVIDENCE = "NO_EVIDENCE"
LOW_RERANK_CONFIDENCE = "LOW_RERANK_CONFIDENCE"
WEAK_EVIDENCE = "WEAK_EVIDENCE"


@dataclass
class EvidenceBundle:
    items: list = field(default_factory=list)
    agreement: dict = field(default_factory=dict)
    sufficiency: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def summary(self):
        """Compact, log-safe view (no passage text)."""
        return {
            "evidence_ids": [i["evidence_id"] for i in self.items],
            "book_ids": [i["book_id"] for i in self.items],
            "pages": [page_label(i) for i in self.items],
            "evidence_scores": [i["evidence_score"] for i in self.items],
            "agreement_status": self.agreement.get("status"),
            "conflicts": self.agreement.get("conflicts", []),
            "sufficiency": self.sufficiency,
        }


def assess_sufficiency(items, config=None):
    config = config or cfg.EVIDENCE

    if not items:
        return {"sufficient": False, "reason": NO_EVIDENCE, "detail": "no passages retrieved"}

    reranker_scores = [i["components"]["reranker"] for i in items if i["components"]["reranker"] is not None]
    strong = sum(1 for i in items if i["evidence_score"] >= config.strong_score)

    if reranker_scores and max(reranker_scores) < config.min_top_reranker:
        return {"sufficient": False, "reason": LOW_RERANK_CONFIDENCE,
                "detail": f"best reranker score {max(reranker_scores):.3f} < {config.min_top_reranker}"}

    if strong < config.min_strong_items:
        return {"sufficient": False, "reason": WEAK_EVIDENCE,
                "detail": f"{strong} item(s) with evidence_score >= {config.strong_score}, need {config.min_strong_items}"}

    return {"sufficient": True, "reason": None, "detail": f"{strong} strong item(s)"}


def build_evidence(reranked, config=None, rrf_k=None):
    config = config or cfg.EVIDENCE

    items = []
    for position, chunk in enumerate(reranked, start=1):
        items.append({
            "evidence_id": make_evidence_id(position),
            "book_id": chunk.get("book_id"),
            "book_name": chunk.get("title") or chunk.get("book_id"),
            "author": chunk.get("author"),
            "page": page_label(chunk),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "paragraph_start": chunk.get("paragraph_start"),
            "paragraph_end": chunk.get("paragraph_end"),
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "language": chunk.get("language", "kn"),
            "ocr_quality": chunk.get("ocr_quality"),
            "retrieval_score": chunk.get("rrf_score", chunk.get("score")),
            "rrf_score": chunk.get("rrf_score", chunk.get("score", 0.0)),
            "dense_rank": chunk.get("dense_rank"),
            "sparse_rank": chunk.get("sparse_rank"),
            "reranker_score": chunk.get("rerank_score"),
            "rerank_score_norm": chunk.get("rerank_score_norm"),
        })

    report = agreement_mod.analyze(items, config)
    warnings = []

    for item in items:
        info = report["per_item"][item["evidence_id"]]
        item["agreement"] = info
        item.update(scorer.score_item(item, info, config, rrf_k))

    if items and all(i["reranker_score"] is None for i in items):
        warnings.append("reranker scores unavailable; evidence scores use retrieval, source and agreement only")

    return EvidenceBundle(items=items, agreement=report, sufficiency=assess_sufficiency(items, config), warnings=warnings)
