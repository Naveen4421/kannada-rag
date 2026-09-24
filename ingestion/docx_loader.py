import argparse
import json
from pathlib import Path

from docx import Document


def has_page_break(paragraph):
    xml = paragraph._p.xml
    return (
        'w:type="page"' in xml
        or "<w:lastRenderedPageBreak/>" in xml
    )


def load_docx(path):
    doc = Document(path)

    pages = []
    current_page = []
    page_number = 1

    for paragraph_number, paragraph in enumerate(doc.paragraphs, start=1):
        text = paragraph.text.strip()

        if text:
            current_page.append({
                "paragraph_id": paragraph_number,
                "text": text
            })

        if has_page_break(paragraph):
            pages.append({
                "page_number": page_number,
                "paragraphs": current_page
            })

            page_number += 1
            current_page = []

    # Save remaining content after the final page break
    if current_page:
        pages.append({
            "page_number": page_number,
            "paragraphs": current_page
        })

    return pages


def build_processed_document(path, book_id, title=None, author="Unknown", language="kn"):
    pages = load_docx(path)

    return {
        "book_id": book_id,
        "title": title or path.stem,
        "author": author,
        "language": language,
        "source_file": path.name,
        "page_count": len(pages),
        "pages": pages,
    }


def save_processed_document(doc, output_dir=Path("data/processed")):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{doc["book_id"]}.json'

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Load a DOCX book into canonical processed JSON.")
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

    print(f"Pages created: {doc['page_count']}")
    print(f"Paragraphs stored: {paragraph_count}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
