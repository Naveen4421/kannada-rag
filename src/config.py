"""Central, env-overridable settings for the query-time RAG core.

Every value here is a tuning knob, not a calibrated constant: thresholds and
weights are starting points to be tuned against the gold test set
(src/eval/gold/). Change them via environment variables, not code.
"""
import os
from dataclasses import dataclass, field


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


def _b(name, default):
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes")


# Bump when SYSTEM/answer instructions or the context layout change; part of
# the cache fingerprint so old cached answers are not served.
PROMPT_VERSION = "evidence-v1"


@dataclass(frozen=True)
class RetrievalConfig:
    # Per-channel candidate counts. 0 = auto (max(4 * top_k, 50), the
    # pre-existing behaviour).
    dense_limit: int = field(default_factory=lambda: _i("RAG_DENSE_LIMIT", 0))
    sparse_limit: int = field(default_factory=lambda: _i("RAG_SPARSE_LIMIT", 0))
    rrf_k: int = field(default_factory=lambda: _i("RAG_RRF_K", 60))

    def limits(self, top_k):
        auto = max(top_k * 4, 50)
        return (self.dense_limit or auto, self.sparse_limit or auto)


@dataclass(frozen=True)
class EvidenceConfig:
    # Weights of the transparent evidence score. Missing components (e.g. no
    # reranker score) are dropped and the remaining weights renormalised.
    w_retrieval: float = field(default_factory=lambda: _f("EVID_W_RETRIEVAL", 0.15))
    w_reranker: float = field(default_factory=lambda: _f("EVID_W_RERANKER", 0.45))
    w_source_quality: float = field(default_factory=lambda: _f("EVID_W_SOURCE", 0.15))
    w_agreement: float = field(default_factory=lambda: _f("EVID_W_AGREEMENT", 0.25))

    # Sufficiency: the best item's reranker score (0..1) must reach
    # min_top_reranker, and at least min_strong_items must reach strong_score.
    min_top_reranker: float = field(default_factory=lambda: _f("EVID_MIN_TOP_RERANKER", 0.10))
    strong_score: float = field(default_factory=lambda: _f("EVID_STRONG_SCORE", 0.50))
    min_strong_items: int = field(default_factory=lambda: _i("EVID_MIN_STRONG_ITEMS", 1))

    # Char-4-gram Jaccard between passages of different books above which
    # they are treated as being about the same thing. Uncalibrated.
    topical_overlap: float = field(default_factory=lambda: _f("EVID_TOPICAL_OVERLAP", 0.12))
    near_duplicate: float = field(default_factory=lambda: _f("EVID_NEAR_DUPLICATE", 0.90))


@dataclass(frozen=True)
class AgentConfig:
    max_iterations: int = field(default_factory=lambda: _i("RAG_MAX_ITERATIONS", 3))
    min_grounding_score: float = field(default_factory=lambda: _f("RAG_MIN_GROUNDING", 0.6))
    # LLM claim-level judge in the complex route (one extra LLM call per
    # iteration). Deterministic checks always run.
    use_llm_judge: bool = field(default_factory=lambda: _b("RAG_LLM_JUDGE", True))


@dataclass(frozen=True)
class CacheConfig:
    max_age_days: int = field(default_factory=lambda: _i("RAG_CACHE_MAX_AGE_DAYS", 7))


RETRIEVAL = RetrievalConfig()
EVIDENCE = EvidenceConfig()
AGENT = AgentConfig()
CACHE = CacheConfig()
