import argparse
from pathlib import Path

from src.ingestion.docx_loader import save_processed_document


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Paragraphs are separated by one or more blank lines -- the plain-text
    # equivalent of docx's paragraph boundaries. Plain text has no page
    # markers, so the whole file is a single page (page_number=1).
    blocks = [block.strip() for block in raw.split("\n\n")]
    blocks = [block for block in blocks if block]

    # Some plain-text exports (e.g. line-by-line OCR dumps) have no blank
    # lines at all, which would otherwise collapse the whole file into one
    # unsplittable "paragraph" -- fall back to one paragraph per line.
    if len(blocks) <= 1:
        blocks = [line.strip() for line in raw.split("\n")]
        blocks = [line for line in blocks if line]

    paragraphs = [
        {"paragraph_id": i, "text": block}
        for i, block in enumerate(blocks, start=1)
    ]

    return [{"page_number": 1, "paragraphs": paragraphs}]


def build_processed_document(path, book_id, title=None, author="Unknown", language="kn"):
    pages = load_text(path)

    return {
        "book_id": book_id,
        "title": title or path.stem,
        "author": author,
        "language": language,
        "source_file": path.name,
        "page_count": len(pages),
        "pages": pages,
    }


def main():
    parser = argparse.ArgumentParser(description="Load a plain-text book into canonical processed JSON.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--title")
    parser.add_argument("--author", default="Unknown")
    parser.add_argument("--language", default="kn")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    print(f"Reading: {args.input_file}")

    doc = build_processed_document(
        args.input_file,
        book_id=args.book_id,
        title=args.title,
        author=args.author,
        language=args.language,
    )

    output_path = save_processed_document(doc, output_dir=args.output_dir)

    paragraph_count = sum(len(page["paragraphs"]) for page in doc["pages"])

    print(f"Paragraphs stored: {paragraph_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
