from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from daily_multimodal.alignment.time_utils import parse_absolute_time


_WEAR_TIME_COLUMNS = {
    "ppg": "csv_time_PPG",
    "gsr": "csv_time_GSR",
    "acc": "csv_time_motion",
}


def build_probe_report(
    window: dict[str, Any],
    *,
    eeg_resample_hz: int = 250,
) -> dict[str, Any]:
    """Build a lightweight single-window probe report without training models."""
    window_start = parse_absolute_time(window["window_start_time"])
    window_end = parse_absolute_time(window["window_end_time"])
    window_size = float(window.get("window_size_seconds") or (window_end - window_start).total_seconds())

    return {
        "sample_id": window["sample_id"],
        "event_id": window.get("event_id", ""),
        "subject_id": window.get("subject_id", ""),
        "session_id": window.get("session_id", ""),
        "window_start_time": window["window_start_time"],
        "window_end_time": window["window_end_time"],
        "eeg": _probe_eeg(window, window_start, window_end, window_size, eeg_resample_hz),
        "wear": {
            "ppg": _probe_wear_csv(window.get("wear_ppg_path", ""), "ppg", window_start, window_end),
            "gsr": _probe_wear_csv(window.get("wear_gsr_path", ""), "gsr", window_start, window_end),
            "acc": _probe_wear_csv(window.get("wear_acc_path", ""), "acc", window_start, window_end),
        },
        "video": _probe_media(window, "video"),
        "audio": _probe_media(window, "audio"),
    }


def save_probe_report(report: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def save_shapes_report(report: dict[str, Any], output: Path | str) -> Path:
    lines = [
        f"sample_id={report['sample_id']}",
        f"eeg_expected_resampled_shape={report['eeg']['expected_resampled_shape']}",
        f"ppg_rows_in_window={report['wear']['ppg']['rows_in_window']}",
        f"gsr_rows_in_window={report['wear']['gsr']['rows_in_window']}",
        f"acc_rows_in_window={report['wear']['acc']['rows_in_window']}",
        f"video_candidate_count={report['video']['candidate_count']}",
        f"audio_candidate_count={report['audio']['candidate_count']}",
    ]
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _probe_eeg(
    window: dict[str, Any],
    window_start,
    window_end,
    window_size: float,
    eeg_resample_hz: int,
) -> dict[str, Any]:
    recording_start_text = window.get("eeg_recording_start_time") or ""
    start_offset = None
    end_offset = None
    if recording_start_text:
        recording_start = parse_absolute_time(recording_start_text)
        start_offset = (window_start - recording_start).total_seconds()
        end_offset = (window_end - recording_start).total_seconds()
    path_text = window.get("eeg_bdf_path") or ""
    path = Path(path_text) if path_text else None
    return {
        "path": path_text,
        "exists": bool(path and path.is_file()),
        "window_start_offset_seconds": start_offset,
        "window_end_offset_seconds": end_offset,
        "target_resample_hz": eeg_resample_hz,
        "expected_resampled_shape": ["channels_unknown", int(round(window_size * eeg_resample_hz))],
    }


def _probe_wear_csv(
    path_text: str,
    modality: str,
    window_start,
    window_end,
) -> dict[str, Any]:
    path = Path(path_text) if path_text else None
    if not path:
        return _empty_wear_result("", modality)
    if not path.is_file():
        result = _empty_wear_result(str(path), modality)
        result["exists"] = False
        return result

    time_column = _WEAR_TIME_COLUMNS[modality]
    rows_in_window = 0
    first_time = None
    last_time = None
    scanned_rows = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            scanned_rows += 1
            raw_time = row.get(time_column)
            if not raw_time:
                continue
            try:
                timestamp = parse_absolute_time(raw_time)
            except ValueError:
                continue
            if timestamp < window_start:
                continue
            if timestamp >= window_end:
                break
            rows_in_window += 1
            first_time = first_time or raw_time
            last_time = raw_time

    return {
        "path": path_text,
        "exists": True,
        "time_column": time_column,
        "scanned_rows": scanned_rows,
        "rows_in_window": rows_in_window,
        "first_time_in_window": first_time,
        "last_time_in_window": last_time,
        "non_empty": rows_in_window > 0,
    }


def _empty_wear_result(path: str, modality: str) -> dict[str, Any]:
    return {
        "path": path,
        "exists": bool(path),
        "time_column": _WEAR_TIME_COLUMNS[modality],
        "scanned_rows": 0,
        "rows_in_window": 0,
        "first_time_in_window": None,
        "last_time_in_window": None,
        "non_empty": False,
    }


def _probe_media(window: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "video":
        candidates = window.get("video_candidates") or window.get("candidate_mp4_paths") or []
    else:
        candidates = window.get("candidate_audio_paths") or window.get("candidate_mp4_paths") or []
    existing = [path for path in candidates if Path(path).is_file()]
    return {
        "candidate_count": len(candidates),
        "existing_count": len(existing),
        "first_candidate": candidates[0] if candidates else "",
        "first_candidate_exists": bool(candidates and Path(candidates[0]).is_file()),
    }
