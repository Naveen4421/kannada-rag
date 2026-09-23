# Kannada RAG — Stack & Workflow

**Technical report** · 2026-09-23
What the system is built from, and how a question travels from a Kannada query to a grounded, cited answer — now a modular RAG pipeline, up from the original naive single-pass design.

| Books/sources indexed | Chunks indexed | Target scale | Generation |
|---|---|---|---|
| 7 (6 real + 1 test) | 1,964 | ~30,000 books | Connected (OpenRouter) |

---

## 1. Overview

The project is a Retrieval-Augmented Generation (RAG) system for searching and answering questions over a collection of Kannada-language books. A user asks a question in Kannada; the system retrieves the most relevant passages from the indexed books and passes them to a large language model, which answers using only that retrieved text and cites its source (book, author, page, chunk).

The system is now split into two clearly separated pipelines, each with its own Streamlit UI page:

- **Dataset pipeline** — turns a `.docx` or `.txt` book file into indexed vectors. Title/author are read automatically from the book's own opening pages (LLM-assisted extraction), not typed in by hand.
- **Query pipeline** — takes a Kannada question, routes it by complexity, retrieves via hybrid dense+keyword search, reranks with a real cross-encoder, and either streams a fast answer or runs a bounded self-critique loop for harder questions.

OCR is still treated as an external, upstream concern for `.docx` sources, but plain-text ingestion (added this pass) accepts arbitrary line-based text exports (e.g. OCR dumps with no paragraph structure) directly. Currently indexed: **7 sources / 1,964 chunks** across 5 historical/biographical books, 1 science-magazine OCR dump, and 1 small test file.

---

## 2. Technology stack

Everything runs locally in a single Python 3.12 virtual environment, with Qdrant running as a separate Docker container — no other external services except the OpenRouter API call at generation time.

| Stage | Technology | Version | Role | Status |
|---|---|---|---|---|
| Ingestion (DOCX) | python-docx | 1.2.0 | Parses paragraphs + Word page-break markers into canonical page/paragraph JSON | ✅ live |
| Ingestion (plain text) | Custom (`text_loader.py`) | — | Splits on blank lines; falls back to one-paragraph-per-line for line-based OCR exports with no blank-line breaks | ✅ live |
| Ingestion (PDF) | PyMuPDF (`pymupdf`) | 1.28.2 | Installed and runnable standalone; not wired into the pipeline — schema mismatch with the chunker | ⏸ deferred |
| Metadata extraction | OpenRouter (`openai/gpt-4o-mini`) | — | Reads the book's opening ~1,500 chars, extracts title/author; falls back to filename/"Unknown" on failure — ingestion never blocks on this | ✅ live |
| Chunking | Custom paragraph-aware chunker | — | Greedy grouping, 1000–1500 char target, paragraph boundaries preserved, no overlap | ✅ live |
| Dense embedding | sentence-transformers · BAAI/bge-m3 | 6.0.1 | Multilingual sentence embeddings; confirmed to handle Kannada well | ✅ live |
| Sparse embedding | fastembed · Qdrant/bm25 | 0.8.1 | Local BM25-style sparse vectors for lexical/exact-term matching | ✅ live |
| Embedding runtime | PyTorch | 2.11.0+cu128 | CUDA if available, otherwise CPU — GPU is shared with other processes on this host and can OOM under contention | ✅ live |
| Vector / search DB | Qdrant | 1.19.1 | Docker-hosted at `localhost:6333`; **hybrid** collection (named `dense` + `sparse` vectors, RRF fusion); one collection across all books, filterable by `book_id`; accessed via a stable alias (`kannada_chunks → kannada_chunks_v2`) so schema migrations don't touch calling code | ✅ live |
| Retrieval | Custom (`retrieval/search.py`) | — | Embeds the question (dense + sparse), queries Qdrant with native RRF fusion, returns ranked chunks with metadata | ✅ live |
| Reranking | Cross-encoder — BAAI/bge-reranker-v2-m3 | — | Real reranking (no longer a stub): scores every `(question, chunk)` pair, re-sorts | ✅ live |
| Routing | Custom (`routing/router.py`) | — | Heuristic (Kannada question-word markers) + embedding-cosine-similarity classifier, zero extra LLM/embedding calls; picks `top_k`/`top_n` and whether query expansion / self-critique run | ✅ live |
| Query expansion | OpenRouter (`openai/gpt-4o-mini`) | — | For "complex"-routed questions only: generates paraphrase variants, merges results via manual RRF | ✅ live |
| Generation | OpenRouter API — `openai/gpt-4o-mini` | — | Chat completion over HTTPS via `requests`, `max_tokens` explicitly capped (1024) to avoid balance-reservation 402s, streamed for the fast path | ✅ live |
| Self-critique agent | Custom (`pipeline/agent.py`) | — | Bounded 3-iteration loop: generate → critique groundedness → broaden retrieval → rewrite query → retry | ✅ live |
| Caching | SQLite (`cache/store.py`) | stdlib | Keyed by normalized question + book filter + top_k/top_n + route; skips the whole pipeline on a hit | ✅ live |
| Observability | JSONL (`observability/logger.py`) | — | Per-query log: route, retrieved/reranked chunk IDs, latency breakdown, cache hit, errors | ✅ live |
| Offline eval | RAGAS | 0.4.3 | Faithfulness / answer relevancy / context precision (LLM-judged) + ID-based precision/recall (free, exact) against a synthetic + hand-labeled test set | ✅ live, needs LLM credits for judged metrics |
| HTTP client | requests | 2.34.2 | Used for all OpenRouter calls | ✅ live |
| Secrets | python-dotenv | 1.2.3 | Loads `OPENROUTER_API_KEY` from a local, git-ignored `.env` | ✅ live |
| User interface | Streamlit | 1.64.0 | **Multipage app**: landing page + separate Ingest page + separate Ask page (see §5) | ✅ live |
| Containerization | Docker / docker-compose | — | `Dockerfile` + `docker-compose.yml` defining `qdrant`, `indexer`, and `app` services | ✅ present, not the primary run path used day-to-day |

---

## 3. Architecture & workflow

Two independent pipelines, each with its own UI page, sharing one Qdrant collection.

```text
┌──────────────────────────────────────────────────────────────────────┐
│  PIPELINE 1 — DATASET (📥 Ingest Data page / pipeline/ingest.py)      │
│                                                                        │
│  .docx / .txt      Loader + Chunker        Metadata            Embed  │
│  file        raw   docx_loader.py /  chunks extract      title/ dense+sparse   Qdrant
│  (uploaded  ──────→ text_loader.py  ───────→ (LLM reads   author ─────────────→ hybrid
│   or CLI)          chunker.py       (JSONL)   opening text)      bge-m3 + BM25   collection
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│  PIPELINE 2 — QUERY (🔎 Ask Questions page / pipeline/query.py:ask()) │
│                                                                        │
│  Question ──embed──→ Router ──cache key──→ Cache? ──hit──→ cached answer
│  (Kannada)          (heuristic+                │miss
│                      cosine sim,                ▼
│                      1 embed call)     ┌────────┴─────────┐
│                                   [simple]            [complex]
│                                        ▼                    ▼
│                          Hybrid search           Agentic loop (≤3 iter):
│                          (dense+sparse RRF)       search → rerank → generate
│                          → rerank (cross-encoder) → critique groundedness
│                          → prompt → stream answer   → broaden / rewrite → retry
└──────────────────────────────────────────────────────────────────────┘
```

Pipeline 1 fills the shared Qdrant collection offline (or on-demand via the Ingest page). Pipeline 2 queries that same collection at ask-time. The router is the fork point: simple factoid questions (`ಎಲ್ಲಿ/ಯಾವಾಗ/ಯಾರು/ಎಷ್ಟು`-type markers, or embedding similarity to example simple questions) take the fast streaming path; open-ended/analytical questions (`ಹೇಗೆ/ಏಕೆ/ವಿವರಿಸಿ`-type, or embedding similarity to example complex questions) get the full agentic treatment. Every request is cached and logged regardless of which path it took.

---

## 4. Pipeline stages

| # | Stage | What it does | File |
|---|---|---|---|
| 01 | Ingest | Reads `.docx` (page-break-aware) or `.txt` (blank-line paragraphs, falling back to one-per-line for line-based OCR dumps). Writes canonical JSON. | `src/ingestion/docx_loader.py`, `src/ingestion/text_loader.py` |
| 02 | Extract metadata | Best-effort title/author from the book's own opening text via the LLM; never blocks ingestion if it fails. | `src/ingestion/metadata_extraction.py` |
| 03 | Chunk | Greedily groups paragraphs into 1000–1500 char chunks, paragraph boundaries preserved, no overlap. | `src/ingestion/chunker.py` |
| 04 | Embed & index | Encodes every chunk with both bge-m3 (dense) and Qdrant/bm25 (sparse), upserts in batches of 64 (crash-safe/idempotent — deterministic point IDs). | `src/embeddings/model.py`, `src/embeddings/embed.py` |
| 05 | Route | Classifies question complexity from Kannada question-word markers + cosine similarity to reference questions — no extra model call. | `src/routing/router.py` |
| 06 | Retrieve | Hybrid dense+sparse search via Qdrant's native RRF fusion, optionally scoped to one `book_id`. | `src/retrieval/search.py` |
| 07 | Rerank | Real cross-encoder (BAAI/bge-reranker-v2-m3) scores and re-sorts every candidate. | `src/retrieval/reranker.py`, `src/reranker/model.py` |
| 08 | Expand (complex only) | Generates paraphrase/HyDE-style query variants, merges multi-query retrieval via RRF. | `src/retrieval/query_expansion.py` |
| 09 | Prompt | Kannada system instructions (answer only from context; if a specific asked aspect is missing, say so explicitly rather than substituting an adjacent fact) + labeled source blocks + question. | `src/generation/prompt.py` |
| 10 | Generate | OpenRouter chat completion, `max_tokens` capped, streaming or plain, with token-usage tracking. | `src/generation/llm.py` |
| 11 | Self-critique (complex only) | Up to 3 iterations: generate → LLM checks groundedness → broaden retrieval (drop book filter, double top_k/top_n) → rewrite query → retry. Always returns its best attempt at the cap. | `src/pipeline/agent.py` |
| 12 | Cache | SQLite, keyed by normalized question + filters + route; skips the whole pipeline on a hit. | `src/cache/store.py` |
| 13 | Log | JSONL, one line per query: route, chunk IDs, latency breakdown, cache hit, errors. | `src/observability/logger.py` |
| 14 | Serve | Two-page Streamlit app (see §5). | `app.py`, `pages/*.py` |

---

## 5. Two-pipeline UI

Streamlit multipage app — one running process (one shared, cached embedding model in memory), two distinct pages in the sidebar:

- **📥 Ingest Data** — file uploader (`.docx`/`.txt` only), no manual fields. Saves the upload, runs the full dataset pipeline, shows a success summary with the *auto-extracted* title/author, or a clear error. Lists currently indexed books.
- **🔎 Ask Questions** — book filter, question box. Shows a route badge (`simple`/`complex`, cache hit, agent iteration count), streams the answer live on the fast path, shows cited sources in expandable panels.
- **app.py** — a thin landing page listing both pipelines and the current book catalog.

---

## 6. Indexed data

All sources share one hybrid Qdrant collection (`kannada_chunks`, currently an alias to `kannada_chunks_v2`), so a question searches across everything unless scoped to one book.

| Source | Type | Chunks | Notes |
|---|---|---:|---|
| TK001 | .docx | 637 | Filename-default metadata |
| TK002 | .docx | 490 | ಚೆನ್ನಬಸವನಾಯಕ — real title/author found on-page |
| TK003 | .docx | 86 | Full metadata (title, author) |
| TK004 | .docx | 235 | ಖಗೇಂದ್ರಮಣಿದರ್ಪಣಂ by ಮಂಗರಸ — real title/author found on-page |
| TK005 | .docx | 380 | ರಾಮಚರಿತ-ಮಾನಸ — real title/author found on-page |
| FTS | .txt | 135 | ವಿಜ್ಞಾನ ಕರ್ಣಾಟಕ science magazine, line-based OCR dump (no blank-line breaks — exercised the text-loader fallback) |
| TK006 | .txt | 1 | Small hand-written test file, not real content |
| **Total** | | **1,964** | |

---

## 7. Retrieval quality

Measured against the 15-question hand-labeled Kannada eval set (`src/retrieval/evaluate.py`, `TK003`-scoped), comparing the current hybrid+rerank pipeline against the original naive baseline:

| Metric | Naive (dense-only, no rerank) | Current (hybrid dense+BM25 + cross-encoder rerank) |
|---|---|---|
| Top-1 recall | 66.7% | 73.3% |
| Top-3 recall | 80.0% | 93.3% |
| Top-5 recall | 80.0% | **100.0%** |
| MRR | 0.724 | 0.839 |

Notably, two questions (Q8, Q9) that dense-only search couldn't find **at all** were recovered by BM25 catching exact terms the embedding model missed.

A larger, synthetic 89-question set (`src/eval/generate_testset.py`, ~18 questions per book, LLM-generated from sampled chunks with the source chunk as ground truth) plus the 15 hand-labeled ones were run through RAGAS (`src/eval/ragas_eval.py`, 104 total questions):

| Metric | Result | Note |
|---|---|---|
| ID-based context recall (free, exact) | **76.4%** | Correct chunk retrieved within top-5, across all 7 books |
| ID-based context precision (free, exact) | 16.2% | Expected to be low — most questions have exactly 1 ground-truth chunk among 5 returned, so the ceiling is ~20%; 0.162/0.20 ≈ 81% of that ceiling |
| Faithfulness, Answer Relevancy, LLM Context Precision | Not yet validly measured | The OpenRouter account ran out of credits mid-run; ~400+ judge calls failed with HTTP 402. Wiring is confirmed correct on a 3-case dry run before that — needs a re-run with credits available |

---

## 8. Fixes made this pass

Real bugs found and fixed while hardening the pipeline, worth recording since they're not obvious from the code alone:

- **Silent streaming failures**: the query dispatcher (`ask()`) hardcoded `"error": None` on its fast/streaming return path, before the stream was ever consumed. An LLM failure mid-stream (e.g. no credits) was logged to file but never shown in the UI — just sources, no answer, no error. Fixed with a shared mutable `error_holder` the UI checks after consuming the stream.
- **Unhandled crashes on retrieval failures**: only the final LLM call was wrapped in error handling; a failure during embedding/routing/search/rerank (e.g. a GPU OOM from other processes on this shared host) crashed `ask()` entirely. Now every stage is guarded and degrades to a clean `error` message.
- **OpenRouter 402s despite having balance**: requests with no explicit `max_tokens` get OpenRouter's default ceiling (16k+), which reserves more budget than a small remaining balance can cover, even though the actual answer only needs a few hundred tokens. Fixed by capping `max_tokens=1024` on every generation call.
- **`.txt` ingestion collapsing to one giant chunk**: the text loader only split paragraphs on blank lines; a real OCR export with zero blank lines (711 single-newline-separated lines) collapsed into one unsplittable ~390KB "paragraph." Fixed with a fallback to one-paragraph-per-line when no blank-line breaks are found.
- **Streamed Kannada text rendering as mojibake**: `requests`' automatic encoding detection failed on the SSE stream, decoding UTF-8 bytes as Latin-1. Fixed by decoding each line explicitly as UTF-8 instead of relying on `iter_lines(decode_unicode=True)`.

---

## 9. Limitations & roadmap

- **PDF ingestion still deferred** — `ingestion/pdf_parser.py` exists but produces a schema the chunker can't consume.
- **RAGAS LLM-judged metrics need a credits top-up** to get a real reading — currently only the free ID-based metrics are valid.
- **GPU is shared with other processes on this host** — reranking and embedding can transiently OOM under contention; this now degrades gracefully instead of crashing, but isn't something the app itself controls.
- **Four of the original five `.docx` books still carry filename-default metadata** (only TK002/004/005 have confirmed real titles/authors) — the auto-extraction feature added this pass will fix this on the next re-ingest once credits are available.
- **Docker path exists but isn't the primary day-to-day run path** — local `.venv` + Docker-hosted Qdrant is what's actually being used and tested.

---

## 10. Quick reference

```bash
# Add a new book (auto-extracts title/author, embeds, indexes)
python -m src.pipeline.ingest data/input/NEWBOOK.docx
python -m src.pipeline.ingest data/input/newfile.txt

# Ask a question from the CLI (fast path, no routing)
python -m src.pipeline.query "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಇಲ್ಲಿ"

# Launch the two-page UI
streamlit run app.py

# Re-run the eval baseline
python -m src.retrieval.evaluate_pipeline

# Generate a fresh synthetic test set / run RAGAS
python -m src.eval.generate_testset
python -m src.eval.ragas_eval
```

---

*Kannada RAG — internal technical report, generated 2026-09-23.*
