import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path


DB_PATH = Path("data/cache/query_cache.sqlite3")


@lru_cache(maxsize=1)
def get_connection(db_path=DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, question TEXT, route TEXT, payload TEXT, created_at TEXT)"
    )
    conn.commit()

    return conn


def make_key(question, book_id, top_k, top_n, route):
    raw = json.dumps(
        {
            "q": question.strip(),
            "book_id": book_id,
            "top_k": top_k,
            "top_n": top_n,
            "route": route,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key):
    conn = get_connection()
    row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()

    if row is None:
        return None

    return json.loads(row[0])


def put(key, question, route, payload):
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, question, route, payload, created_at) "
        "VALUES (?, ?, ?, ?, datetime('now'))",
        (key, question, route, json.dumps(payload, ensure_ascii=False)),
    )
    conn.commit()
