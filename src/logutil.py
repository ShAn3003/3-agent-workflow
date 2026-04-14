import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

LOG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(record: Dict[str, Any]) -> None:
    path = LOG_DIR / f"run-{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
