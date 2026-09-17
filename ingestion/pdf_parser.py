import json
from pathlib import Path

import pymupdf


INPUT_DIR = Path("data/input")
OUTPUT_DIR = Path("data/processed")


def extract_pdf(pdf_path: Path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text
        })

    doc.close()

    return pages


def main():
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF found in: {INPUT_DIR}")
        return

    if len(pdf_files) > 1:
        print("Multiple PDFs found:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        print("\nFor now, keep only one PDF in data/input/")
        return

    input_pdf = pdf_files[0]

    print(f"Processing: {input_pdf.name}")

    pages = extract_pdf(input_pdf)

    output = {
        "book_id": "BOOK_000001",
        "title": input_pdf.stem,
        "language": "kn",
        "source_file": input_pdf.name,
        "total_pages": len(pages),
        "pages": pages
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_json = OUTPUT_DIR / f"{input_pdf.stem}.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Pages extracted: {len(pages)}")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()
