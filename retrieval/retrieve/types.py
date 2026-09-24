from dataclasses import dataclass, field, asdict


@dataclass
class RetrievalResult:
    """One fused candidate. Per-channel scores/ranks are kept (None when the
    chunk was not returned by that channel) so retrieval can be debugged and
    the evidence layer can see how a chunk was found."""

    chunk_id: str
    book_id: str = None
    page: object = None
    text: str = ""
    dense_score: float = None
    sparse_score: float = None
    dense_rank: int = None
    sparse_rank: int = None
    rrf_score: float = 0.0
    # Remaining payload fields (title, author, page_start, ...), unchanged.
    payload: dict = field(default_factory=dict)

    def to_dict(self):
        """Flat dict for the rest of the pipeline. `score` is the RRF score,
        kept for backward compatibility with the old search() output."""
        flat = dict(self.payload)
        flat.update({k: v for k, v in asdict(self).items() if k != "payload"})
        flat["score"] = self.rrf_score
        return flat


@dataclass
class RetrievalOutput:
    results: list = field(default_factory=list)  # list[dict] (RetrievalResult.to_dict)
    timings_ms: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
