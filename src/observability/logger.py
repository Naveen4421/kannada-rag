import json
import re
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path("data/logs/queries.jsonl")

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|authorization|password)", re.IGNORECASE)


def _scrub(value):
    if isinstance(value, dict):
        return {k: ("[redacted]" if _SECRET_KEY.search(str(k)) else _scrub(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def log_query(entry):
    """Append one JSONL record. Never raises: a logging problem must not fail
    a user's query."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **_scrub(entry)}

        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
