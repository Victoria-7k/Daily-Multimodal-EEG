from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_manifest(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_manifest(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "events_total": len(rows),
        "complete_wear_events": sum(bool(row.get("is_complete_wear_event")) for row in rows),
        "video_day_events": sum(bool(row.get("has_video")) for row in rows),
        "complete_multimodal_candidates": sum(
            bool(row.get("is_complete_multimodal_candidate")) for row in rows
        ),
    }

