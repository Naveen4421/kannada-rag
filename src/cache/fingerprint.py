"""Cache fingerprint: everything that changes what a cached answer would be.

Strategy (see also store.py): the cache KEY identifies the question
(normalised text, book filter, route, top_k, top_n). The FINGERPRINT identifies
the pipeline that produced the answer. An entry whose fingerprint differs from
the current one is treated as a miss and overwritten -- so a model, prompt or
retrieval-config change invalidates exactly the entries it affects, and
nothing else. Only settings that influence a route's answer are included
(agent settings only for the complex route).

Also included: the index size. Ingesting a book changes what retrieval can
find, so answers cached before it must not be served. Re-ingesting a book
without changing the total chunk count is NOT detected; clear the cache then.
"""
import hashlib
import json
import time
from dataclasses import asdict

from src import config as cfg

_INDEX_TTL_S = 30
_index_memo = {"at": 0.0, "value": None}


def index_size():
    now = time.monotonic()
    if _index_memo["value"] is not None and now - _index_memo["at"] < _INDEX_TTL_S:
        return _index_memo["value"]

    try:
        from src.retrieval.search import _default_client, _defaults

        value = _default_client().count(collection_name=_defaults()[0], exact=True).count
    except Exception:
        return "unknown"

    _index_memo.update(at=now, value=value)
    return value


def components(route, index_size_value=None):
    from src.embeddings.model import MODEL_NAME as EMBED_MODEL, SPARSE_MODEL_NAME
    from src.generation.llm import MODEL_NAME as LLM_MODEL
    from src.reranker.model import RERANKER_MODEL

    parts = {
        "llm": LLM_MODEL,
        "embedding": EMBED_MODEL,
        "sparse": SPARSE_MODEL_NAME,
        "reranker": RERANKER_MODEL,
        "prompt_version": cfg.PROMPT_VERSION,
        "retrieval": asdict(cfg.RETRIEVAL),
        "evidence": asdict(cfg.EVIDENCE),
        "index_size": index_size() if index_size_value is None else index_size_value,
    }
    if route == "complex":
        parts["agent"] = asdict(cfg.AGENT)
    return parts


def fingerprint(route, index_size_value=None):
    raw = json.dumps(components(route, index_size_value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
