from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TranscriptWriter:
    def __init__(self, workspace_root: Path):
        runs_dir = workspace_root / ".odysseus_harness" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = runs_dir / f"{stamp}-{uuid.uuid4().hex[:8]}.jsonl"

    def write(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "data": data,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
