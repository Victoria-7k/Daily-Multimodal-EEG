from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DAY_SECONDS = 86400


def classify_eeg_window_coverage(
    *,
    start_offset_seconds: float,
    end_offset_seconds: float,
    bdf_duration_seconds: float,
) -> dict[str, Any]:
    start = float(start_offset_seconds)
    end = float(end_offset_seconds)
    duration = float(bdf_duration_seconds)
    shifted = _whole_day_shift(start, end, duration)
    if shifted is not None:
        classification = "whole_day_shift_candidate"
    elif 0.0 <= start and end <= duration:
        classification = "in_range"
    elif end <= 0.0:
        classification = "negative_offset"
    elif start >= duration:
        classification = "after_recording_end"
    elif start < duration and end > 0.0:
        classification = "partial_overlap"
    else:
        classification = "out_of_range"
    overlap_seconds = max(0.0, min(end, duration) - max(start, 0.0))
    return {
        "classification": classification,
        "start_offset_seconds": start,
        "end_offset_seconds": end,
        "bdf_duration_seconds": duration,
        "overlap_seconds": overlap_seconds,
        "whole_day_shift_candidate": shifted is not None,
        "suggested_shift_seconds": shifted,
    }


def summarize_eeg_coverage(windows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [record for record in (audit_eeg_window(row) for row in windows) if record is not None]
    counts: dict[str, int] = {}
    affected = set()
    for record in records:
        classification = str(record["classification"])
        counts[classification] = counts.get(classification, 0) + 1
        if classification != "in_range":
            subject = str(record.get("subject_id") or "<missing-subject>")
            session = str(record.get("session_id") or "<missing-session>")
            affected.add(f"{subject}/{session}")
    return {
        "total_windows": len(windows),
        "audited_windows": len(records),
        "in_range_count": counts.get("in_range", 0),
        "negative_offset_count": counts.get("negative_offset", 0),
        "after_recording_end_count": counts.get("after_recording_end", 0),
        "partial_overlap_count": counts.get("partial_overlap", 0),
        "whole_day_shift_candidate_count": counts.get("whole_day_shift_candidate", 0),
        "out_of_range_count": counts.get("out_of_range", 0),
        "missing_duration_count": len(windows) - len(records),
        "affected_subject_sessions": sorted(affected),
        "records": records,
    }


def audit_eeg_window(window: dict[str, Any]) -> dict[str, Any] | None:
    duration = eeg_duration_seconds(window)
    offsets = eeg_window_offsets(window)
    if duration is None or offsets is None:
        return None
    start, end = offsets
    result = classify_eeg_window_coverage(
        start_offset_seconds=start,
        end_offset_seconds=end,
        bdf_duration_seconds=duration,
    )
    result.update(
        {
            "sample_id": window.get("sample_id", ""),
            "event_id": window.get("event_id", ""),
            "subject_id": window.get("subject_id", ""),
            "session_id": window.get("session_id", ""),
            "eeg_bdf_path": window.get("eeg_bdf_path") or window.get("source_path") or "",
        }
    )
    return result


def eeg_window_offsets(row: dict[str, Any]) -> tuple[float, float] | None:
    if (
        row.get("eeg_onset_seconds") is not None
        and row.get("window_start_offset_seconds") is not None
        and row.get("window_end_offset_seconds") is not None
    ):
        onset = float(row["eeg_onset_seconds"])
        return onset + float(row["window_start_offset_seconds"]), onset + float(row["window_end_offset_seconds"])
    if row.get("window_start_offset_seconds") is not None and row.get("window_end_offset_seconds") is not None:
        return float(row["window_start_offset_seconds"]), float(row["window_end_offset_seconds"])
    if row.get("eeg_onset_seconds") is not None:
        onset = float(row["eeg_onset_seconds"])
        start = float(row.get("event_window_start_seconds", row.get("window_start_offset_seconds", -10)))
        end = float(row.get("event_window_end_seconds", row.get("window_end_offset_seconds", 0)))
        return onset + start, onset + end
    return None


def eeg_duration_seconds(row: dict[str, Any]) -> float | None:
    for key in (
        "eeg_recording_duration_seconds",
        "bdf_duration_seconds",
        "recording_duration_seconds",
        "source_duration_seconds",
    ):
        if row.get(key) is not None:
            return float(row[key])
    sidecar = row.get("eeg_json_path")
    if sidecar:
        return _duration_from_sidecar(Path(str(sidecar)))
    return None


def _whole_day_shift(start: float, end: float, duration: float) -> int | None:
    for shift in (-DAY_SECONDS, DAY_SECONDS):
        shifted_start = start + shift
        shifted_end = end + shift
        if 0.0 <= shifted_start and shifted_end <= duration:
            return shift
    return None


def _duration_from_sidecar(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    for key in ("RecordingDuration", "recording_duration", "duration", "Duration"):
        if payload.get(key) is not None:
            value = payload[key]
            if isinstance(value, str) and ":" in value:
                parts = [float(part) for part in value.split(":")]
                total = 0.0
                for part in parts:
                    total = total * 60.0 + part
                return total
            return float(value)
    return None
