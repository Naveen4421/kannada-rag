# Kannada RAG

A Retrieval-Augmented Generation (RAG) system designed for searching and answering questions from a large collection of Kannada books.

The system starts from extracted Kannada text. OCR is treated as an upstream process and is not part of this RAG project.

---

## 1. Project Goal

The goal is to build a scalable Kannada knowledge-retrieval system that can work with a large collection of books.

The initial development uses one book:

```text
TK003.docx
```

The long-term target is approximately:

```text
30,000 Kannada books
```

Users will be able to ask questions in Kannada and receive answers based on the contents of the available books.

The system should also provide source information such as:

* Book
* Author
* Page
* Chunk

---

## 2. What is RAG?

RAG stands for **Retrieval-Augmented Generation**.

Instead of asking an LLM to answer a question only from its training knowledge, the system first searches the project's document collection for relevant information.

The retrieved text is then provided to the LLM so that it can generate a grounded answer.

```text
User Question
      ↓
Retrieve relevant documents
      ↓
Relevant Kannada text
      ↓
LLM
      ↓
Grounded answer
```

RAG is an architecture, not a single model.

---

## 3. Overall Architecture

```text
                    BOOK COLLECTION
                           │
                           ▼
                 OCR / TEXT EXTRACTION
                    (external system)
                           │
                           ▼
                  Extracted Kannada Text
                           │
                           ▼
                  ┌─────────────────┐
                  │    Ingestion    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Cleaning    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Chunking    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Kannada         │
                  │ Embeddings      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Vector Index /  │
                  │ Database        │
                  └────────┬────────┘
                           │
                           │
                  USER QUESTION
                           │
                           ▼
                  ┌─────────────────┐
                  │ Query Embedding │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Retrieval     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   Reranking     │
                  │   (optional)    │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      LLM        │
                  └────────┬────────┘
                           │
                           ▼
                  Kannada Answer
                  + Source Metadata
```

---

## 4. OCR Boundary

OCR is **not part of this repository's RAG pipeline**.

The expected flow is:

```text
Scanned Book
     ↓
External OCR System
     ↓
Extracted Kannada Text
     ↓
Kannada RAG
```

This separation allows different OCR systems to be used without changing the RAG pipeline.

For example, an OCR system may produce:

```text
book_000001.txt
```

The RAG system can then ingest that text.

---

## 5. Current Pipeline

The current implementation is:

```text
DOCX
 ↓
DOCX Loader
 ↓
Processed JSON
 ↓
Paragraph-aware Chunker
 ↓
JSONL Chunks
```

For `TK003`:

```text
TK003.docx
     ↓
472 non-empty paragraphs
     ↓
84 detected document page sections
     ↓
Paragraph-aware chunks
```

---

## 6. Chunking Strategy

The first chunking strategy is paragraph-aware.

Current configuration:

```text
Target size: approximately 1000 characters
Maximum: approximately 1500 characters
Overlap: none
```

The chunker prefers paragraph boundaries instead of cutting Kannada text in the middle of a paragraph.

Example:

```text
Paragraph 1
Paragraph 2
Paragraph 3
      ↓
    Chunk 1

Paragraph 4
Paragraph 5
Paragraph 6
      ↓
    Chunk 2
```

There is currently **no overlap** between chunks.

This makes the first retrieval implementation easier to evaluate and avoids unnecessary duplication.

Chunk sizes are not required to be exactly 1000 characters.

Semantic and paragraph boundaries are more important than an exact character count.

---

## 7. Chunk Metadata

Each chunk stores metadata such as:

```json
{
  "chunk_id": "TK003_0001",
  "book_id": "TK003",
  "title": "ಕರ್ತವ್ಯಾನಂದ ಶ್ರೀ ರಾಜ ಪುರೋಹಿತರು",
  "author": "ಶ್ರೀನಿವಾಸ ಹಾವನೂರ",
  "language": "kn",
  "page_start": 1,
  "page_end": 3,
  "paragraph_start": 1,
  "paragraph_end": 23,
  "text": "...",
  "char_count": 1237
}
```

This metadata will later be used for source attribution and citations.

---

## 8. Project Structure

```text
kannada-rag/
│
├── data/
│   │
│   ├── input/
│   │   └── original extracted files
│   │
│   ├── processed/
│   │   └── canonical document JSON
│   │
│   ├── chunks/
│   │   └── chunked JSONL files
│   │
│   └── index/
│       └── vector index / database
│
├── ingestion/            PIPELINE 1: dataset preparation and indexing
│   ├── docx_loader.py, text_loader.py, pdf_parser.py
│   ├── metadata_extraction.py
│   ├── chunker.py
│   ├── ingest.py         (load -> metadata -> chunk -> JSONL -> embed)
│   └── embed.py          (dense + sparse embeddings -> Qdrant)
│
├── retrieval/            PIPELINE 2: query time (evidence-grounded RAG)
│   ├── config.py
│   ├── query_processing/, routing/, cache/
│   ├── retrieve/         (search.py hybrid+RRF, reranker.py, query_expansion.py)
│   ├── evidence/, generation/, validation/
│   ├── pipeline/         (query.py, agent.py, rag_core.py, errors.py)
│   ├── observability/
│   └── eval/             (metrics, gold set, run_eval, RAGAS scripts)
│
├── common/               used by both pipelines
│   ├── embedding_model.py  (bge-m3 dense, BM25 sparse)
│   ├── llm.py              (OpenRouter client)
│   ├── index.py            (Qdrant client, collection name)
│   └── catalog.py          (list ingested books)
│
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   └── test_retrieval.py
│
├── requirements.txt
├── README.md
└── .venv/
```

Some directories/files are planned and will be added as development progresses.

---

## 9. Data Flow

### Ingestion

```text
Input File
   ↓
Loader
   ↓
Canonical JSON
   ↓
Cleaner
   ↓
Chunker
   ↓
JSONL chunks
```

### Embedding

```text
JSONL chunks
   ↓
Kannada embedding model
   ↓
Vector for every chunk
   ↓
Vector index
```

### Query

```text
Kannada question
   ↓
Query embedding
   ↓
Similarity search
   ↓
Relevant chunks
   ↓
Optional reranking
   ↓
LLM
   ↓
Answer + sources
```

---

## 10. Embeddings

The project will use a **Kannada-capable embedding model**.

Embeddings convert text into numerical vectors that can be compared for semantic similarity.

For example:

```text
Kannada chunk
     ↓
Embedding model
     ↓
[0.021, -0.183, 0.442, ...]
```

The embedding does not replace the original text.

Both the vector and the original chunk text are retained.

---

## 11. Retrieval

When a user asks a question:

```text
"ರಾಜಪುರೋಹಿತರ ತಂದೆಯವರು ಯಾವಾಗ ನಿಧನರಾದರು?"
```

the system will:

```text
Question
   ↓
Query embedding
   ↓
Vector search
   ↓
Top-K chunks
```

The retrieved chunks contain the original Kannada text and metadata.

The LLM receives those chunks as context.

---

## 12. LLM Generation

The LLM is responsible for generating the final response using retrieved information.

Conceptually:

```text
System Instructions
       +
User Question
       +
Retrieved Kannada Context
       ↓
      LLM
       ↓
Kannada Answer
```

The LLM should not be expected to search the entire book collection itself.

Retrieval is handled by the RAG system.

---

## 13. Source Attribution

The system will preserve source information throughout the pipeline.

A future response can contain information such as:

```text
Source:
Book: ಕರ್ತವ್ಯಾನಂದ ಶ್ರೀ ರಾಜ ಪುರೋಹಿತರು
Author: ಶ್ರೀನಿವಾಸ ಹಾವನೂರ
Page: 5
Chunk: TK003_0005
```

This makes answers easier to verify against the original book.

---

## 14. Scaling to 30,000 Books

The system is designed so that books can be processed independently.

```text
Book 1
   ↓
Load → Clean → Chunk → Embed → Index

Book 2
   ↓
Load → Clean → Chunk → Embed → Index

Book 3
   ↓
Load → Clean → Chunk → Embed → Index

...

Book 30,000
   ↓
Load → Clean → Chunk → Embed → Index
```

The entire collection does not need to be loaded into memory at once.

The ingestion pipeline can process books incrementally.

---

## 15. Future Retrieval Improvements

The first implementation will use vector similarity search.

Later, the system can support hybrid retrieval:

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Keyword Search       Vector Search
             │                     │
             └──────────┬──────────┘
                        ▼
                    Candidates
                        │
                        ▼
                    Reranker
                        │
                        ▼
                  Best Chunks
                        │
                        ▼
                       LLM
```

Hybrid retrieval can be useful when users ask for:

* Exact names
* Historical dates
* Specific book terminology
* Rare Kannada words
* Semantic questions

---

## 16. Development Status

### Completed

* [x] Project structure
* [x] Initial `TK003.docx`
* [x] DOCX extraction
* [x] Canonical processed JSON
* [x] Paragraph metadata
* [x] Page-break preservation
* [x] Paragraph-aware chunking
* [x] No-overlap chunking
* [x] JSONL chunk output

### In Progress

* [ ] Kannada embedding model
* [ ] Embedding generation
* [ ] Vector index
* [ ] Similarity retrieval
* [ ] Kannada retrieval evaluation

### Planned

* [ ] Reranking
* [ ] Hybrid search
* [ ] LLM integration
* [ ] Source citations
* [ ] Query API
* [ ] User interface
* [ ] Large-scale ingestion
* [ ] 30,000-book collection

---

## 17. Development Philosophy

The project is being developed incrementally.

Each stage is validated before moving to the next stage:

```text
Input
  ↓
Validate
  ↓
Ingestion
  ↓
Validate
  ↓
Chunking
  ↓
Validate
  ↓
Embeddings
  ↓
Validate
  ↓
Retrieval
  ↓
Validate
  ↓
LLM
```

The original extracted text should remain unchanged.

Derived representations such as cleaned text, chunks, embeddings, and indexes should be stored separately.

---

## 18. Current Next Step

The current chunking pipeline has been validated using `TK003`.

The next step is:

```text
TK003.jsonl
     ↓
Kannada embedding model
     ↓
Generate embeddings
     ↓
Test semantic similarity
```

Only after confirming that Kannada retrieval works correctly should the project move to the vector database and LLM generation stages.
# kannada-rag
