"""One trace per query: a query_id plus per-stage timings and facts, emitted
as a single JSONL line so a query can be followed through the whole pipeline.

Only ids, scores, timings and counts are recorded for retrieved material --
never passage text, prompts, or credentials.
"""
import time
import uuid
from contextlib import contextmanager

from retrieval.observability.logger import log_query


class QueryTrace:
    def __init__(self, original_query="", clock=time.perf_counter):
        self.query_id = uuid.uuid4().hex[:12]
        self._clock = clock
        self._started = clock()
        self.data = {"query_id": self.query_id, "original_query": original_query,
                     "latency_ms": {}, "agent": {"iterations": []}, "errors": [], "warnings": []}

    def set(self, **fields):
        self.data.update(fields)

    def section(self, name, **fields):
        self.data.setdefault(name, {}).update(fields)

    @contextmanager
    def stage(self, name):
        """Times a block into latency_ms[name] (accumulates across iterations)."""
        t = self._clock()
        try:
            yield
        finally:
            lat = self.data["latency_ms"]
            lat[name] = lat.get(name, 0.0) + (self._clock() - t) * 1000

    def warn(self, message):
        self.data["warnings"].append(message)

    def error(self, err):
        self.data["errors"].append(err)

    def add_iteration(self, **fields):
        self.data["agent"]["iterations"].append(fields)

    def finish(self, emit=True):
        self.data["latency_ms"]["total"] = (self._clock() - self._started) * 1000
        if emit:
            log_query(self.data)
        return self.data
