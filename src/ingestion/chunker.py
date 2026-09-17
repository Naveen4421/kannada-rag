import argparse
import json
from pathlib import Path


TARGET_CHARS = 1000
MAX_CHARS = 1500


def load_book(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_paragraphs(book):
    paragraphs = []

    for page in book["pages"]:
        page_number = page["page_number"]

        for paragraph in page["paragraphs"]:
            text = paragraph["text"].strip()

            if text:
                paragraphs.append({
                    "page_number": page_number,
                    "paragraph_id": paragraph["paragraph_id"],
                    "text": text,
                })

    return paragraphs


def chunk_length(items):
    # Newline between paragraphs counts as one character.
    return sum(len(item["text"]) for item in items) + max(0, len(items) - 1)


def create_chunk(book, items, chunk_number):
    return {
        "chunk_id": f'{book["book_id"]}_{chunk_number:04d}',
        "book_id": book["book_id"],
        "title": book["title"],
        "author": book["author"],
        "language": book["language"],
        "page_start": items[0]["page_number"],
        "page_end": items[-1]["page_number"],
        "paragraph_start": items[0]["paragraph_id"],
        "paragraph_end": items[-1]["paragraph_id"],
        "text": "\n".join(item["text"] for item in items),
        "char_count": sum(len(item["text"]) for item in items),
    }


def create_chunks(book, paragraphs, target_chars=TARGET_CHARS, max_chars=MAX_CHARS):
    chunks = []
    current = []

    for paragraph in paragraphs:

        if not current:
            current.append(paragraph)
            continue

        new_length = chunk_length(current + [paragraph])

        # Keep adding paragraphs while we are below the target.
        if new_length <= target_chars:
            current.append(paragraph)
            continue

        # If adding this paragraph crosses the target but
        # remains within the maximum, keep it.
        if new_length <= max_chars:
            current.append(paragraph)

        # Finish the current chunk.
        chunks.append(
            create_chunk(
                book,
                current,
                len(chunks) + 1,
            )
        )

        # IMPORTANT:
        # No overlap.
        # The next chunk starts with the next paragraph.
        if new_length > max_chars:
            current = [paragraph]
        else:
            current = []

    # Save the final chunk.
    if current:
        chunks.append(
            create_chunk(
                book,
                current,
                len(chunks) + 1,
            )
        )

    return chunks


def chunk_book(book, target_chars=TARGET_CHARS, max_chars=MAX_CHARS):
    paragraphs = flatten_paragraphs(book)
    return create_chunks(book, paragraphs, target_chars=target_chars, max_chars=max_chars)


def save_chunks(chunks, output_dir=Path("data/chunks")):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{chunks[0]["book_id"]}.jsonl'

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                ) + "\n"
            )

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Chunk a processed book JSON into paragraph-aware JSONL chunks.")
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--target-chars", type=int, default=TARGET_CHARS)
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS)
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    args = parser.parse_args()

    print(f"Reading: {args.input_file}")

    book = load_book(args.input_file)
    paragraphs = flatten_paragraphs(book)

    print(f"Paragraphs: {len(paragraphs)}")

    chunks = create_chunks(book, paragraphs, target_chars=args.target_chars, max_chars=args.max_chars)
    output_path = save_chunks(chunks, output_dir=args.output_dir)

    sizes = [chunk["char_count"] for chunk in chunks]

    print(f"Chunks created: {len(chunks)}")
    print(f"Saved: {output_path}")
    print()
    print("Chunk size statistics:")
    print(f"Smallest: {min(sizes)}")
    print(f"Largest:  {max(sizes)}")
    print(f"Average:  {sum(sizes) / len(sizes):.1f}")


if __name__ == "__main__":
    main()
