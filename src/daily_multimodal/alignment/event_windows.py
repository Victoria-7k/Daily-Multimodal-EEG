from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from daily_multimodal.alignment.time_utils import parse_absolute_time


DEFAULT_START_SECONDS = -120
DEFAULT_END_SECONDS = 0
DEFAULT_WINDOW_SIZE_SECONDS = 10
DEFAULT_STRIDE_SECONDS = 10


def build_window_index(
    rows: list[dict[str, Any]],
    *,
    start_seconds: int | float = DEFAULT_START_SECONDS,
    end_seconds: int | float = DEFAULT_END_SECONDS,
    window_size_seconds: int | float = DEFAULT_WINDOW_SIZE_SECONDS,
    stride_seconds: int | float = DEFAULT_STRIDE_SECONDS,
    require_all_modalities: bool = False,
) -> list[dict[str, Any]]:
    """Create stable window records from event-level manifest rows."""
    windows, _summary = build_window_index_with_summary(
        rows,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        window_size_seconds=window_size_seconds,
        stride_seconds=stride_seconds,
        require_all_modalities=require_all_modalities,
    )
    return windows


def build_window_index_with_summary(
    rows: list[dict[str, Any]],
    *,
    start_seconds: int | float = DEFAULT_START_SECONDS,
    end_seconds: int | float = DEFAULT_END_SECONDS,
    window_size_seconds: int | float = DEFAULT_WINDOW_SIZE_SECONDS,
    stride_seconds: int | float = DEFAULT_STRIDE_SECONDS,
    require_all_modalities: bool = False,
    min_history_seconds: int | float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create window records and report events skipped before sample expansion."""
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")
    if end_seconds <= start_seconds:
        raise ValueError("end_seconds must be greater than start_seconds")
    if window_size_seconds > (end_seconds - start_seconds):
        raise ValueError("window_size_seconds cannot exceed the event window range")

    required_history_seconds = _required_history_seconds(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        min_history_seconds=min_history_seconds,
    )
    windows: list[dict[str, Any]] = []
    selected_events: set[str] = set()
    skipped_events: list[dict[str, Any]] = []
    for row in rows:
        event_id = str(row.get("event_id", ""))
        if require_all_modalities and not row.get("is_complete_multimodal_candidate"):
            skipped_events.append(
                _skip_record(row, "incomplete_multimodal_candidate", required_history_seconds, None)
            )
            continue
        event_time = parse_absolute_time(row["absolute_onset_time"])
        skip = _skip_reason(
            row,
            event_time=event_time,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            required_history_seconds=required_history_seconds,
        )
        if skip is not None:
            reason, available_history = skip
            skipped_events.append(_skip_record(row, reason, required_history_seconds, available_history))
            continue
        selected_events.add(event_id)
        offset = float(start_seconds)
        window_id = 0
        while offset + float(window_size_seconds) <= float(end_seconds) + 1e-9:
            window_start = event_time + timedelta(seconds=offset)
            window_end = window_start + timedelta(seconds=float(window_size_seconds))
            video_candidates = _video_candidates_for_window(
                row.get("video_candidates", []),
                window_start,
                window_end,
            )
            has_face = (
                bool(video_candidates)
                if "video_candidates" in row
                else bool(row.get("has_video"))
            )
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
                    "event_window_start_seconds": _clean_number(start_seconds),
                    "event_window_end_seconds": _clean_number(end_seconds),
                    "required_history_seconds": _clean_number(required_history_seconds),
                    "pre_event_history_seconds": _clean_number(
                        _available_pre_event_history_seconds(row, event_time)
                    ),
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
                    "video_candidates": video_candidates,
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
    summary = _summary(
        rows=rows,
        windows=windows,
        selected_event_count=len(selected_events),
        skipped_events=skipped_events,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        window_size_seconds=window_size_seconds,
        stride_seconds=stride_seconds,
        required_history_seconds=required_history_seconds,
        require_all_modalities=require_all_modalities,
    )
    return windows, summary


def save_window_index(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def save_window_index_summary(summary: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
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
    if value is None:
        return value
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _required_history_seconds(
    *,
    start_seconds: int | float,
    end_seconds: int | float,
    min_history_seconds: int | float | None,
) -> float:
    if min_history_seconds is not None:
        return float(min_history_seconds)
    if float(start_seconds) < 0 <= float(end_seconds):
        return abs(float(start_seconds))
    return float(end_seconds) - float(start_seconds)


def _skip_reason(
    row: dict[str, Any],
    *,
    event_time,
    start_seconds: int | float,
    end_seconds: int | float,
    required_history_seconds: float,
) -> tuple[str, float | None] | None:
    available_history = _available_pre_event_history_seconds(row, event_time)
    if available_history is not None and available_history + 1e-9 < required_history_seconds:
        return "insufficient_pre_event_history", available_history
    if "video_candidates" in row and not _has_candidate_covering_event_range(
        row.get("video_candidates", []),
        event_time=event_time,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    ):
        return "insufficient_video_coverage", available_history
    return None


def _available_pre_event_history_seconds(row: dict[str, Any], event_time) -> float | None:
    eeg_onset = row.get("eeg_onset_seconds")
    if eeg_onset is not None:
        try:
            return float(eeg_onset)
        except (TypeError, ValueError):
            pass
    recording_start = row.get("eeg_recording_start_time")
    if recording_start:
        try:
            return float((event_time - parse_absolute_time(str(recording_start))).total_seconds())
        except ValueError:
            return None
    return None


def _has_candidate_covering_event_range(
    candidates: Any,
    *,
    event_time,
    start_seconds: int | float,
    end_seconds: int | float,
) -> bool:
    event_start = event_time + timedelta(seconds=float(start_seconds))
    event_end = event_time + timedelta(seconds=float(end_seconds))
    return any(
        _candidate_covers_time_range(candidate, event_start, event_end)
        for candidate in candidates or []
    )


def _has_precise_candidate_covering_window(
    candidates: Any,
    window_start,
    window_end,
    *,
    require_audio: bool = False,
) -> bool:
    return any(
        (not require_audio or bool(candidate.get("has_audio_stream")))
        and _candidate_covers_time_range(candidate, window_start, window_end)
        for candidate in candidates or []
    )


def _candidate_covers_time_range(candidate: dict[str, Any], start_time, end_time) -> bool:
    mp4_start_text = candidate.get("mp4_start_time")
    mp4_end_text = candidate.get("mp4_end_time")
    if mp4_start_text and mp4_end_text:
        try:
            mp4_start = parse_absolute_time(str(mp4_start_text))
            mp4_end = parse_absolute_time(str(mp4_end_text))
        except ValueError:
            return bool(candidate.get("covers_window"))
        return bool(mp4_start <= start_time and mp4_end >= end_time)
    return bool(candidate.get("covers_window"))


def _video_candidates_for_window(
    candidates: Any,
    window_start,
    window_end,
) -> list[dict[str, Any]]:
    rebased: list[dict[str, Any]] = []
    for candidate in candidates or []:
        row = _video_candidate_for_window(candidate, window_start, window_end)
        if row is not None:
            rebased.append(row)
    return rebased


def _video_candidate_for_window(
    candidate: dict[str, Any],
    window_start,
    window_end,
) -> dict[str, Any] | None:
    output = dict(candidate)
    mp4_start_text = candidate.get("mp4_start_time")
    mp4_end_text = candidate.get("mp4_end_time")
    if not mp4_start_text or not mp4_end_text:
        return output if candidate.get("covers_window") or candidate.get("overlap_seconds", 0) else None
    try:
        mp4_start = parse_absolute_time(str(mp4_start_text))
        mp4_end = parse_absolute_time(str(mp4_end_text))
    except ValueError:
        return output if candidate.get("covers_window") or candidate.get("overlap_seconds", 0) else None

    overlap_start = max(window_start, mp4_start)
    overlap_end = min(window_end, mp4_end)
    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
    if overlap_seconds <= 0:
        return None
    duration = candidate.get("duration_seconds")
    try:
        duration_seconds = float(duration) if duration is not None else float((mp4_end - mp4_start).total_seconds())
    except (TypeError, ValueError):
        duration_seconds = float((mp4_end - mp4_start).total_seconds())
    clip_start = max(0.0, float((window_start - mp4_start).total_seconds()))
    clip_end = min(duration_seconds, float((window_end - mp4_start).total_seconds()))
    output.update(
        {
            "clip_start_seconds": _clean_number(round(clip_start, 6)),
            "clip_end_seconds": _clean_number(round(clip_end, 6)),
            "overlap_seconds": _clean_number(round(overlap_seconds, 6)),
            "covers_window": bool(mp4_start <= window_start and mp4_end >= window_end),
        }
    )
    return output


def _skip_record(
    row: dict[str, Any],
    reason: str,
    required_history_seconds: float,
    available_history_seconds: float | None,
) -> dict[str, Any]:
    return {
        "event_id": str(row.get("event_id", "")),
        "subject_id": str(row.get("subject_id", "")),
        "session_id": str(row.get("session_id", "")),
        "absolute_onset_time": str(row.get("absolute_onset_time", "")),
        "reason": reason,
        "available_history_seconds": _clean_number(available_history_seconds),
        "required_history_seconds": _clean_number(required_history_seconds),
    }


def _summary(
    *,
    rows: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    selected_event_count: int,
    skipped_events: list[dict[str, Any]],
    start_seconds: int | float,
    end_seconds: int | float,
    window_size_seconds: int | float,
    stride_seconds: int | float,
    required_history_seconds: float,
    require_all_modalities: bool,
) -> dict[str, Any]:
    return {
        "events_total": len(rows),
        "events_selected": selected_event_count,
        "events_skipped": len(skipped_events),
        "windows_total": len(windows),
        "start_seconds": _clean_number(start_seconds),
        "end_seconds": _clean_number(end_seconds),
        "window_size_seconds": _clean_number(window_size_seconds),
        "stride_seconds": _clean_number(stride_seconds),
        "required_history_seconds": _clean_number(required_history_seconds),
        "require_all_modalities": bool(require_all_modalities),
        "skip_reasons": _count_skip_reasons(skipped_events),
        "skipped_events": skipped_events,
    }


def _count_skip_reasons(skipped_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in skipped_events:
        reason = str(event.get("reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts
