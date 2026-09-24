"""Transparent evidence score.

evidence_score = sum(w_i * c_i) / sum(w_i)   over the components that are
available for the item; missing components (e.g. the reranker failed) are
dropped and the remaining weights renormalised, never imputed.

Components (each 0..1, all visible in the output):
  retrieval       RRF score as a fraction of the best attainable (rank 1 in both channels)
  reranker        cross-encoder relevance of (question, passage), sigmoid-normalised
  source_quality  0.5 * metadata completeness (real title, real author, page) +
                  0.5 * text quality (payload `ocr_quality` if the ingestion
                  provides one, else the share of Kannada letters for kn text)
  agreement       see agreement.agreement_component

The result is a heuristic ranking/strength signal. It is NOT a calibrated
probability; weights and thresholds live in retrieval/config.py.
"""
import unicodedata

from retrieval import config as cfg
from retrieval.evidence.agreement import agreement_component

FORMULA = "sum(w_i * c_i) / sum(w_i) over available components"
NOTE = "heuristic evidence strength, not a calibrated probability"


def retrieval_component(item, rrf_k=None, channels=2):
    rrf_k = rrf_k or cfg.RETRIEVAL.rrf_k
    best = channels / (rrf_k + 1)
    return max(0.0, min(1.0, item.get("rrf_score", 0.0) / best))


def _kannada_ratio(text):
    letters = [ch for ch in text if unicodedata.category(ch)[0] in ("L", "M")]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if "ಀ" <= ch <= "೿") / len(letters)


def text_quality(item):
    ocr = item.get("ocr_quality")
    if isinstance(ocr, (int, float)) and 0.0 <= ocr <= 1.0:
        return float(ocr)
    if item.get("language", "kn") == "kn":
        return min(1.0, _kannada_ratio(item.get("text", "")) / 0.8)
    return 1.0


def metadata_completeness(item):
    title = (item.get("title") or "").strip()
    author = (item.get("author") or "").strip()
    checks = [
        bool(title) and title != item.get("book_id"),
        bool(author) and author.lower() != "unknown",
        item.get("page_start") is not None,
    ]
    return sum(checks) / len(checks)


def source_quality_component(item):
    return 0.5 * metadata_completeness(item) + 0.5 * text_quality(item)


def score_item(item, agreement_info, config=None, rrf_k=None):
    config = config or cfg.EVIDENCE

    components = {
        "retrieval": retrieval_component(item, rrf_k),
        "reranker": item.get("rerank_score_norm"),
        "source_quality": source_quality_component(item),
        "agreement": agreement_component(agreement_info),
    }
    weights = {
        "retrieval": config.w_retrieval,
        "reranker": config.w_reranker,
        "source_quality": config.w_source_quality,
        "agreement": config.w_agreement,
    }

    used = {k: w for k, w in weights.items() if components[k] is not None and w > 0}
    total = sum(used.values())
    score = sum(components[k] * w for k, w in used.items()) / total if total else 0.0

    return {
        "evidence_score": round(score, 4),
        "components": {k: (None if v is None else round(v, 4)) for k, v in components.items()},
        "weights_used": {k: round(w / total, 4) for k, w in used.items()} if total else {},
    }
