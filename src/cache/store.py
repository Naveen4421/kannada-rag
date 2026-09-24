import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

from src import config as cfg
from src.query_processing.processor import normalize_text


DB_PATH = Path("data/cache/query_cache.sqlite3")


@lru_cache(maxsize=None)
def get_connection(db_path=DB_PATH):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache ("
        "key TEXT PRIMARY KEY, question TEXT, route TEXT, payload TEXT, created_at TEXT, fingerprint TEXT)"
    )

    # Migrate caches created before fingerprints existed. Their rows have a
    # NULL fingerprint and are therefore never served (their config is unknown).
    columns = {row[1] for row in conn.execute("PRAGMA table_info(cache)")}
    if "fingerprint" not in columns:
        conn.execute("ALTER TABLE cache ADD COLUMN fingerprint TEXT")

    conn.commit()

    return conn


def make_key(question, book_id, top_k, top_n, route):
    raw = json.dumps(
        {
            "q": normalize_text(question),
            "book_id": book_id,
            "top_k": top_k,
            "top_n": top_n,
            "route": route,
        },
        sort_keys=True,
        ensure_ascii=False,
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key, fingerprint=None, db_path=None):
    """Cached payload, or None on a miss. A row is a miss when its fingerprint
    differs from `fingerprint` (pipeline changed) or it is older than
    CACHE.max_age_days."""
    conn = get_connection(db_path or DB_PATH)
    row = conn.execute(
        "SELECT payload, fingerprint, "
        "(julianday('now') - julianday(created_at)) AS age_days FROM cache WHERE key = ?",
        (key,),
    ).fetchone()

    if row is None:
        return None

    payload, stored_fingerprint, age_days = row

    if fingerprint is not None and stored_fingerprint != fingerprint:
        return None

    if age_days is not None and age_days > cfg.CACHE.max_age_days:
        return None

    return json.loads(payload)


def put(key, question, route, payload, fingerprint=None, db_path=None):
    conn = get_connection(db_path or DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache (key, question, route, payload, created_at, fingerprint) "
        "VALUES (?, ?, ?, ?, datetime('now'), ?)",
        (key, question, route, json.dumps(payload, ensure_ascii=False, default=str), fingerprint),
    )
    conn.commit()
