# Kannada RAG — Stack & Workflow

**Technical report** · 2026-09-16
What the system is built from, and how a question travels from a Kannada query to a grounded, cited answer, as of the current build.

| Books indexed | Chunks indexed | Target scale | Generation |
|---|---|---|---|
| 5 | 1,828 | ~30,000 books | Connected (OpenRouter) |

---

## 1. Overview

The project is a Retrieval-Augmented Generation (RAG) system for searching and answering questions over a collection of Kannada-language books. A user asks a question in Kannada; the system retrieves the most relevant passages from the indexed books and passes them to a large language model, which answers using only that retrieved text and cites its source (book, author, page, chunk).

OCR is treated as an external, upstream concern — the pipeline starts from already-extracted `.docx` text. The long-term goal is roughly 30,000 books; the current build has ingested and indexed **5** books end to end and exposes them through a Streamlit UI with a live LLM connection.

---

## 2. Technology stack

Everything runs locally in a single Python 3.12 virtual environment — no external services except the embedding model download and the OpenRouter API call at generation time.

| Stage | Technology | Version | Role | Status |
|---|---|---|---|---|
| Ingestion (DOCX) | python-docx | 1.2.0 | Parses paragraphs + Word page-break markers into canonical page/paragraph JSON | ✅ live |
| Ingestion (PDF) | PyMuPDF (`pymupdf`) | 1.28.2 | Installed and runnable standalone; not wired into the pipeline — schema mismatch with the chunker | ⏸ deferred |
| Ingestion (plain text) | — | — | No `.txt` loader yet; dispatch seam documented in `pipeline/ingest.py` | ⏸ planned |
| Chunking | Custom paragraph-aware chunker | — | Greedy grouping, 1000–1500 char target, paragraph boundaries preserved, no overlap | ✅ live |
| Embedding model | sentence-transformers · BAAI/bge-m3 | 6.0.1 | Multilingual sentence embeddings; confirmed to handle Kannada well | ✅ live |
| Embedding runtime | PyTorch | 2.11.0+cu128 | CUDA if available, otherwise CPU | ✅ live |
| Vector / search DB | ChromaDB (PersistentClient) | 1.5.9 | Local, file-persisted at `data/index/chroma/`; cosine similarity; one collection across all books, filterable by `book_id` | ✅ live |
| Retrieval | Custom (`retrieval/search.py`) | — | Embeds the question, queries Chroma, returns ranked chunks with metadata | ✅ live |
| Reranking | Pass-through stub (`retrieval/reranker.py`) | — | Truncates vector-ranked results to top-n; no cross-encoder scoring yet | 🟡 stub |
| Generation | OpenRouter API — `openai/gpt-4o-mini` | — | Chat completion over HTTPS via `requests`, grounded strictly on retrieved chunks | ✅ live |
| HTTP client | requests | 2.34.2 | Used for the OpenRouter call | ✅ live |
| Secrets | python-dotenv | 1.2.3 | Loads `OPENROUTER_API_KEY` from a local, git-ignored `.env` | ✅ live |
| User interface | Streamlit | 1.64.0 | Question box, per-book filter, top-k/top-n controls, answer + expandable sources | ✅ live |

---

## 3. Architecture & workflow

Two flows meet at the vector database. Books are ingested once, offline, into a shared index; each question then queries that same index at ask-time.

```text
                    INGESTION (offline, per book)

   DOCX files          Loader + Chunker         Embed (bge-m3)        Chroma vector DB
  ┌───────────┐  raw   ┌────────────────┐ chunks ┌──────────────┐ vectors+ ┌────────────────┐
  │ data/input│ text  →│ docx_loader.py │ (JSONL)│embeddings/    │metadata →│ data/index/    │
  │ /*.docx   │────────│ chunker.py     │───────→│ model.py      │─────────→│ chroma/        │
  └───────────┘        └────────────────┘        └──────────────┘          └────────┬───────┘
                                                                                      │
                                                                          nearest     │
                                                                          chunks      │
                                                                       (query-time)   │
                    QUERY (per question)                                             │
                                                                                      ▼
  ┌───────────────┐  question   ┌─────────────┐  top-10/20  ┌───────────────────┐
  │ User question │────────────→│  Retrieval   │────────────→│ Reranker (stub)   │
  │ (Kannada)     │             │ search.py    │             │ pass-through      │
  └───────────────┘             └─────────────┘             └─────────┬─────────┘
                                                                        │ top-3/5
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │ Prompt builder   │
                                                              │ generation/      │
                                                              │ prompt.py        │
                                                              └────────┬─────────┘
                                                                       │ prompt text
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │ LLM call         │
                                                              │ OpenRouter       │
                                                              │ gpt-4o-mini      │
                                                              └────────┬─────────┘
                                                                       │ answer
                                                                       ▼
                                                              ┌──────────────────┐
                                                              │ Answer           │
                                                              │ + citations      │
                                                              └──────────────────┘
```

Ingestion (top) fills one shared Chroma collection offline. At query time, a question and the same collection both feed **Retrieval**; results pass through today's pass-through reranker stub before the prompt is built and sent to the LLM.

---

## 4. Pipeline stages

| # | Stage | What it does | File |
|---|---|---|---|
| 01 | Ingest | Reads a `.docx`, walks paragraphs, detects Word page-break XML markers to group text into pages. Writes canonical JSON to `data/processed/<book_id>.json`. | `src/ingestion/docx_loader.py` |
| 02 | Chunk | Greedily groups consecutive paragraphs into 1000–1500 character chunks, preferring paragraph boundaries, no overlap. Writes `data/chunks/<book_id>.jsonl`. | `src/ingestion/chunker.py` |
| 03 | Embed & index | Encodes every chunk with bge-m3 and upserts into Chroma, keyed by `chunk_id` — safe to re-run without duplicating vectors. | `src/embeddings/embed.py` |
| 04 | Retrieve | Embeds the question with the same model, queries Chroma for the top-k nearest chunks (cosine), optionally scoped to one `book_id`. | `src/retrieval/search.py` |
| 05 | Rerank 🟡 stub | Currently just truncates the vector-ranked list to top-n. Signature is stable so a real cross-encoder can drop in later without touching callers. | `src/retrieval/reranker.py` |
| 06 | Prompt | Builds a Kannada system instruction plus labeled source blocks (title, author, page range, chunk id) and the question. | `src/generation/prompt.py` |
| 07 | Generate | Calls OpenRouter's chat completions endpoint with `openai/gpt-4o-mini`, grounded strictly on the passed chunks. | `src/generation/llm.py` |
| 08 | Serve | Streamlit app wraps the whole chain: question box, book filter, top-k/top-n sliders, answer, and expandable cited sources. | `app.py` |

---

## 5. Indexed data

All five books share one Chroma collection (`kannada_chunks`), so a question searches across all of them unless scoped to one book in the UI.

| Book ID | Title | Author | Pages | Chunks | Metadata |
|---|---|---|---:|---:|---|
| TK001 | TK001 *(filename default)* | Unknown | 384 | 637 | 🟡 defaulted |
| TK002 | TK002 *(filename default)* | Unknown | 367 | 490 | 🟡 defaulted |
| TK003 | ಕರ್ತವ್ಯಾನಂದ ಶ್ರೀ ರಾಜ ಪುರೋಹಿತರು | ಶ್ರೀನಿವಾಸ ಹಾವನೂರ | 84 | 86 | ✅ complete |
| TK004 | TK004 *(filename default)* | Unknown | 210 | 235 | 🟡 defaulted |
| TK005 | TK005 *(filename default)* | Unknown | 240 | 380 | 🟡 defaulted |
| **Total** | | | **1,285** | **1,828** | |

Four books were ingested with filename-derived titles and no author, since real metadata wasn't supplied at ingest time. Re-running `pipeline.ingest` with `--title`/`--author` overwrites this cleanly.

---

## 6. Retrieval quality

Measured with the existing 15-question Kannada eval set (`src/retrieval/evaluate.py`), run against `TK003` only:

| Top-1 recall | Top-3 recall | Top-5 recall | MRR |
|---|---|---|---|
| 66.7% | 80.0% | 80.0% | 0.724 |

These numbers come from a brute-force, in-memory baseline independent of Chroma. Cross-checked directly: the Chroma-backed `pipeline/query.py` returns identical ranks and scores for the same questions, confirming the vector DB isn't degrading retrieval quality relative to that baseline.

> **Known gap:** the same evaluation surfaced a case where the correct passage was retrieved but the LLM answered a nearby-but-wrong aspect of it (asked *where* someone was born, answered *when*). The book itself never states the place explicitly. Fixed by tightening the prompt's system instructions to require the model to name the specific missing aspect rather than substitute an adjacent fact — verified against both the failing and a working case.

---

## 7. Limitations & roadmap

- **Reranking is a stub.** No cross-encoder is scoring candidates yet — result quality rests entirely on vector search. `BAAI/bge-reranker-v2-m3` was evaluated and deferred by choice.
- **PDF and plain-text ingestion aren't wired in.** `ingestion/pdf_parser.py` exists but produces a schema the chunker can't consume; a `.txt` loader has a documented seam but no implementation.
- **Four of five books carry placeholder metadata** (title = filename, author = "Unknown") — cosmetic only, doesn't affect retrieval.
- **No automated test suite yet** — correctness has been checked via regression diffs and the eval script above, not `pytest`.
- **Single global collection by design** — supports cross-book search now and per-book filtering already works via metadata; this is the intended shape for scaling toward 30,000 books, not a shortcut.

---

## 8. Quick reference

```bash
# Add a new book (loads → chunks → embeds → indexes)
python -m src.pipeline.ingest data/input/NEWBOOK.docx \
    --book-id NEWBOOK --title "..." --author "..."

# Ask a question from the CLI
python -m src.pipeline.query "ನಿಮ್ಮ ಪ್ರಶ್ನೆ ಇಲ್ಲಿ"

# Launch the UI
streamlit run app.py
```

---

*Kannada RAG — internal technical report, generated 2026-09-16.*
