from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from daily_multimodal.alignment.time_utils import parse_absolute_time


def build_window_index(
    rows: list[dict[str, Any]],
    *,
    start_seconds: int | float = -10,
    end_seconds: int | float = 0,
    window_size_seconds: int | float = 10,
    stride_seconds: int | float = 5,
    require_all_modalities: bool = False,
) -> list[dict[str, Any]]:
    """Create stable window records from event-level manifest rows."""
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if window_size_seconds > (end_seconds - start_seconds):
        raise ValueError("window_size_seconds cannot exceed the event window range")

    windows: list[dict[str, Any]] = []
    for row in rows:
        if require_all_modalities and not row.get("is_complete_multimodal_candidate"):
            continue
        event_time = parse_absolute_time(row["absolute_onset_time"])
        offset = float(start_seconds)
        window_id = 0
        while offset + float(window_size_seconds) <= float(end_seconds) + 1e-9:
            window_start = event_time + timedelta(seconds=offset)
            window_end = window_start + timedelta(seconds=float(window_size_seconds))
            event_id = row["event_id"]
            video_candidates = row.get("video_candidates", [])
            has_face = bool(video_candidates) if "video_candidates" in row else bool(row.get("has_video"))
            has_audio = (
                any(bool(candidate.get("has_audio_stream")) for candidate in video_candidates)
                if "video_candidates" in row
                else bool(row.get("has_audio"))
            )
            windows.append(
                {
                    "sample_id": f"{event_id}_win-{window_id:04d}",
                    "event_id": event_id,
                    "subject_id": row.get("subject_id", ""),
                    "session_id": row.get("session_id", ""),
                    "segment_id": row.get("segment_id", ""),
                    "window_id": window_id,
                    "absolute_onset_time": row.get("absolute_onset_time", ""),
                    "window_start_time": _format_time(window_start),
                    "window_end_time": _format_time(window_end),
                    "window_start_offset_seconds": _clean_number(offset),
                    "window_end_offset_seconds": _clean_number(offset + float(window_size_seconds)),
                    "window_size_seconds": _clean_number(window_size_seconds),
                    "window_stride_seconds": _clean_number(stride_seconds),
                    "label_columns": row.get("labels", {}),
                    "activity_category": row.get("activity_category", ""),
                    "social_presence": row.get("social_presence", ""),
                    "eeg_recording_start_time": row.get("eeg_recording_start_time", ""),
                    "eeg_onset_seconds": row.get("eeg_onset_seconds"),
                    "eeg_sampling_frequency": row.get("eeg_sampling_frequency"),
                    "eeg_bdf_path": row.get("eeg_bdf_path", ""),
                    "eeg_json_path": row.get("eeg_json_path", ""),
                    "beh_tsv_path": row.get("beh_tsv_path", ""),
                    "wear_ppg_path": row.get("wear_ppg_path", ""),
                    "wear_gsr_path": row.get("wear_gsr_path", ""),
                    "wear_acc_path": row.get("wear_acc_path", ""),
                    "video_day_dir": row.get("video_day_dir", ""),
                    "candidate_mp4_paths": row.get("candidate_mp4_paths", []),
                    "candidate_audio_paths": row.get("candidate_audio_paths", []),
                    "video_candidates": row.get("video_candidates", []),
                    "has_eeg": bool(row.get("has_eeg")),
                    "has_ppg": bool(row.get("has_ppg")),
                    "has_gsr": bool(row.get("has_gsr")),
                    "has_acc": bool(row.get("has_acc")),
                    "has_wear": bool(row.get("has_ppg") or row.get("has_gsr") or row.get("has_acc")),
                    "has_face": has_face,
                    "has_audio": has_audio,
                }
            )
            window_id += 1
            offset += float(stride_seconds)
    return windows


def save_window_index(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def load_window_index(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _format_time(value) -> str:
    if value.microsecond:
        return value.strftime("%Y-%m-%d %H:%M:%S.%f").rstrip("0").rstrip(".")
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _clean_number(value: int | float) -> int | float:
    number = float(value)
    if number.is_integer():
        return int(number)
    return number
