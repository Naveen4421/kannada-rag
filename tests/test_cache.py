import sqlite3

from retrieval.cache import fingerprint as fp
from retrieval.cache import store


def test_key_normalises_whitespace_but_separates_filters_and_routes():
    k = store.make_key("ಪ್ರಶ್ನೆ  ಇದು ?", None, 6, 3, "simple")
    assert k == store.make_key(" ಪ್ರಶ್ನೆ ಇದು ?", None, 6, 3, "simple")
    assert k != store.make_key("ಪ್ರಶ್ನೆ ಇದು ?", "TK003", 6, 3, "simple")
    assert k != store.make_key("ಪ್ರಶ್ನೆ ಇದು ?", None, 15, 6, "complex")


def test_fingerprint_mismatch_is_a_miss_and_match_is_a_hit(tmp_path):
    db = tmp_path / "c.sqlite3"
    store.put("k", "q", "simple", {"answer": "a"}, fingerprint="fp1", db_path=db)
    assert store.get("k", "fp1", db_path=db) == {"answer": "a"}
    assert store.get("k", "fp2", db_path=db) is None


def test_old_schema_is_migrated_and_old_rows_not_served(tmp_path):
    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, question TEXT, route TEXT, payload TEXT, created_at TEXT)")
    conn.execute("INSERT INTO cache VALUES ('k','q','simple','{\"answer\":\"old\"}', datetime('now'))")
    conn.commit()
    conn.close()

    assert store.get("k", "fp1", db_path=db) is None
    store.put("k", "q", "simple", {"answer": "new"}, fingerprint="fp1", db_path=db)
    assert store.get("k", "fp1", db_path=db) == {"answer": "new"}


def test_expired_entries_are_misses(tmp_path):
    db = tmp_path / "c.sqlite3"
    store.put("k", "q", "simple", {"answer": "a"}, fingerprint="fp", db_path=db)
    store.get_connection(db).execute("UPDATE cache SET created_at = datetime('now','-30 days')")
    assert store.get("k", "fp", db_path=db) is None


def test_fingerprint_changes_with_model_prompt_index_and_route(monkeypatch):
    base = fp.fingerprint("simple", index_size_value=100)
    assert base == fp.fingerprint("simple", index_size_value=100)
    assert base != fp.fingerprint("simple", index_size_value=101)  # book ingested
    assert base != fp.fingerprint("complex", index_size_value=100)  # agent settings only affect complex
    monkeypatch.setattr("retrieval.config.PROMPT_VERSION", "evidence-v2")
    assert base != fp.fingerprint("simple", index_size_value=100)


def test_agent_setting_does_not_invalidate_simple_route(monkeypatch):
    from retrieval import config
    from dataclasses import replace

    base = fp.fingerprint("simple", index_size_value=1)
    monkeypatch.setattr(config, "AGENT", replace(config.AGENT, max_iterations=9))
    assert fp.fingerprint("simple", index_size_value=1) == base
