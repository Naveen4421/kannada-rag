import argparse
from pathlib import Path

from ingestion import docx_loader, text_loader, chunker
from ingestion.metadata_extraction import extract_metadata
from ingestion.embed import embed_and_index


LOADERS = {
    ".docx": docx_loader,
    ".txt": text_loader,
}


def ingest_file(input_path, book_id=None, title=None, author=None, language="kn", skip_embed=False):
    input_path = Path(input_path)
    book_id = book_id or input_path.stem.upper()

    suffix = input_path.suffix.lower()

    if suffix == ".pdf":
        raise NotImplementedError(
            "PDF ingestion is out of scope for now. ingestion/pdf_parser.py "
            "exists but produces page-level text with no paragraph splitting, "
            "which is incompatible with chunker.py's expected schema "
            "({page_number, paragraphs: [...]})."
        )

    loader = LOADERS.get(suffix)
    if loader is None:
        raise NotImplementedError(f"Unsupported input type: {suffix}")

    doc = loader.build_processed_document(
        input_path,
        book_id=book_id,
        title=title,
        author=author or "Unknown",
        language=language,
    )

    # If the caller didn't pin title/author explicitly, try to read them off
    # the book's own opening pages rather than falling back straight to the
    # filename -- best-effort, never blocks ingestion if it fails.
    if title is None or author is None:
        extracted_title, extracted_author = extract_metadata(doc)

        if title is None and extracted_title:
            doc["title"] = extracted_title

        if author is None and extracted_author:
            doc["author"] = extracted_author

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
        "title": doc["title"],
        "author": doc["author"],
        "pages": doc["page_count"],
        "paragraphs": paragraph_count,
        "chunks": len(chunks),
        "chunks_path": chunks_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest a book file: load -> chunk -> embed.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--book-id")
    parser.add_argument("--title", help="Omit to auto-extract from the book's opening pages")
    parser.add_argument("--author", help="Omit to auto-extract from the book's opening pages")
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
