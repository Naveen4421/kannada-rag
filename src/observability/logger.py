import json
from datetime import datetime, timezone
from pathlib import Path


LOG_PATH = Path("data/logs/queries.jsonl")


def log_query(entry):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **entry}

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
