import argparse
from pathlib import Path

from src.ingestion import docx_loader, chunker
from src.embeddings.embed import embed_and_index


def ingest_file(input_path, book_id=None, title=None, author="Unknown", language="kn", skip_embed=False):
    input_path = Path(input_path)
    book_id = book_id or input_path.stem.upper()
    title = title or input_path.stem

    suffix = input_path.suffix.lower()

    if suffix == ".docx":
        doc = docx_loader.build_processed_document(
            input_path,
            book_id=book_id,
            title=title,
            author=author,
            language=language,
        )
    elif suffix == ".txt":
        raise NotImplementedError(
            "Plain-text ingestion isn't implemented yet. Add "
            "src/ingestion/text_loader.py with a build_processed_document(path, "
            "book_id, title=None, author='Unknown', language='kn') -> dict "
            "matching docx_loader's signature/output shape, then register "
            "'.txt' in this dispatch."
        )
    elif suffix == ".pdf":
        raise NotImplementedError(
            "PDF ingestion is out of scope for now. ingestion/pdf_parser.py "
            "exists but produces page-level text with no paragraph splitting, "
            "which is incompatible with chunker.py's expected schema "
            "({page_number, paragraphs: [...]})."
        )
    else:
        raise NotImplementedError(f"Unsupported input type: {suffix}")

    processed_path = docx_loader.save_processed_document(doc)
    print(f"Processed: {processed_path}")

    chunks = chunker.chunk_book(doc)
    chunks_path = chunker.save_chunks(chunks)
    print(f"Chunked:   {chunks_path} ({len(chunks)} chunks)")

    if not skip_embed:
        embed_and_index([chunks_path])
        print(f"Indexed:   {len(chunks)} chunks")

    paragraph_count = sum(len(page["paragraphs"]) for page in doc["pages"])

    return {
        "book_id": book_id,
        "pages": doc["page_count"],
        "paragraphs": paragraph_count,
        "chunks": len(chunks),
        "chunks_path": chunks_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest a book file: load -> chunk -> embed.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--book-id")
    parser.add_argument("--title")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--language", default="kn")
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()

    print(f"Ingesting: {args.input_file}")

    summary = ingest_file(
        args.input_file,
        book_id=args.book_id,
        title=args.title,
        author=args.author,
        language=args.language,
        skip_embed=args.skip_embed,
    )

    print("\nDone:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
