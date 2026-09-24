# RAG core (query time)

```
question -> Query Processor -> Router -> Cache
         -> Hybrid retrieval: dense (bge-m3) + sparse (BM25), client-side RRF
         -> Reranker (bge-reranker-v2-m3)
         -> Evidence Engine (ids, source/page, scores, agreement/conflict, sufficiency)
         -> Context Builder -> LLM -> Grounding Validator -> answer + citations
```

Complex route wraps the same pipeline in `src/pipeline/agent.py` (bounded retries with logged reasons).
Simple route streams; grounding is checked after the stream (deterministic checks only, no retry).

## Modules
| Stage | File |
|---|---|
| Query processing | `src/query_processing/processor.py` |
| Retrieval, RRF | `src/retrieval/search.py`, `types.py` |
| Rerank | `src/retrieval/reranker.py` |
| Evidence | `src/evidence/{engine,scorer,agreement,citations}.py` |
| Context/prompt | `src/generation/context_builder.py` |
| Grounding | `src/validation/grounding.py` |
| Shared retrieval half | `src/pipeline/rag_core.py` |
| Agent | `src/pipeline/agent.py` |
| Entry points | `src/pipeline/query.py` (`ask`, `answer_question`) |
| Trace/log | `src/observability/trace.py`, `logger.py` (one JSONL line per query) |
| Config | `src/config.py` (env-overridable) |

## Evidence score
`evidence_score = sum(w_i * c_i) / sum(w_i)` over available components (missing ones are dropped, never imputed):
`retrieval` (RRF as a fraction of the best attainable), `reranker` (sigmoid-normalised cross-encoder score),
`source_quality` (0.5 metadata completeness + 0.5 text quality; uses `ocr_quality` from the payload if ingestion
provides it, else the Kannada-letter share), `agreement` (0.5 neutral; up with independent supporting books; 0.2 on conflict).
Default weights 0.15 / 0.45 / 0.15 / 0.25 (`EVID_W_*`). It is a heuristic strength signal, NOT a calibrated probability.
All thresholds (`EVID_*`) are uncalibrated starting points: tune them on `src/eval/gold`.

## Agreement / conflict (heuristic, lexical)
Independence is by `book_id`. Same-book passages are counted as `same_book_passages`, never as agreement.
Different-book passages with char-4-gram Jaccard >= `EVID_TOPICAL_OVERLAP` corroborate; if both contain 3-4 digit
numbers (mostly years) and share none, they are flagged as a possible numeric conflict. No NLI model is involved.

## Retry policy (complex route, max `RAG_MAX_ITERATIONS`=3)
| Reason | Next action |
|---|---|
| NO_EVIDENCE / WEAK_EVIDENCE / LOW_RERANK_CONFIDENCE (checked before generating) | broaden (drop book filter, double pool), then LLM query variant |
| UNSUPPORTED_CLAIMS / INVALID_CITATIONS / UNSUPPORTED_NUMBERS / OVERSTATED | regenerate once on the same evidence with feedback, then as above |
No evidence after the last retrieval: the LLM is not called. Conflicting evidence is reported, not retried.

## Cache
Key = normalised question + book filter + route + top_k + top_n. Each row also stores a **fingerprint** of what
produced it: LLM model, embedding/sparse/reranker models, `PROMPT_VERSION`, retrieval and evidence config, agent config
(complex route only) and the collection's chunk count. A fingerprint mismatch or age > `RAG_CACHE_MAX_AGE_DAYS` is a miss,
so a change invalidates only the entries it affects. Bump `PROMPT_VERSION` in `src/config.py` when prompts change.
Ungrounded or degraded (reranker/channel fallback) answers are not cached. Rows written before this change have no
fingerprint and are never served. Limit: re-ingesting a book without changing the total chunk count is not detected.

## Failures
`error` stays a user-facing string; `error_detail` = `{stage, code, message}`; `success` is False.
A retrieval failure produces no answer and says so; it never falls through to an LLM answer without evidence.
One failed retrieval channel degrades to the other and is reported in `warnings`; a failed reranker falls back to RRF order.

## Tests and evaluation
```
pip install -r requirements-dev.txt && pytest          # no GPU/Qdrant needed (in-memory Qdrant)
python -m src.eval.gold                                # re-verify the mechanical gold-set claims
python -m src.eval.run_eval retrieval                  # needs Qdrant + models
python -m src.eval.run_eval rag --limit 10             # needs Qdrant + models + OPENROUTER_API_KEY
```
