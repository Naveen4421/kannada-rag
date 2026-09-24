import json
from pathlib import Path


def list_books():
    books = []

    for path in sorted(Path("data/processed").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)

        if "pages" in doc and doc["pages"] and "paragraphs" in doc["pages"][0]:
            books.append({"book_id": doc["book_id"], "title": doc["title"], "author": doc.get("author", "Unknown")})

    return books
