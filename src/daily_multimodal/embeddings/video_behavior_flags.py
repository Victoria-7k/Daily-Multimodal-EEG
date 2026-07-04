from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable


BEHAVIOR_RATIO_NAMES = (
    "face_visible_ratio",
    "low_confidence_ratio",
    "head_down_ratio",
    "side_turn_ratio",
    "hand_near_face_ratio",
    "hand_occlusion_ratio",
    "large_motion_ratio",
    "offscreen_ratio",
    "person_visible_ratio",
    "hand_visible_ratio",
)

_FLAG_TO_RATIO = {
    "face_visible": "face_visible_ratio",
    "low_confidence": "low_confidence_ratio",
    "head_down": "head_down_ratio",
    "side_turn": "side_turn_ratio",
    "hand_near_face": "hand_near_face_ratio",
    "hand_occlusion": "hand_occlusion_ratio",
    "large_motion": "large_motion_ratio",
    "offscreen": "offscreen_ratio",
    "person_visible": "person_visible_ratio",
    "hand_visible": "hand_visible_ratio",
}


def frame_flags_from_openface_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, bool]]:
    """Return first-pass behavior flags from detector/OpenFace-like frame rows."""
    flags: list[dict[str, bool]] = []
    previous_person_bbox: tuple[float, float, float, float] | None = None
    for row in rows:
        face_bbox = _bbox_from_row(row, "face_bbox")
        hand_bbox = _bbox_from_row(row, "hand_bbox")
        person_bbox = _bbox_from_row(row, "person_bbox")

        openface_success = _truthy(row.get("success"))
        mediapipe_face_present = _optional_bool(
            row,
            "mediapipe_face_landmarks_present",
            "mediapipe_face_present",
            "mediapipe_face_tracking_success",
        )
        mediapipe_pose_present = _optional_bool(
            row,
            "mediapipe_pose_present",
            "mediapipe_pose_landmarks_present",
            "mediapipe_pose_tracking_success",
        )

        face_visible = bool(openface_success or face_bbox is not None or mediapipe_face_present is True)
        low_confidence = _is_low_confidence(row, mediapipe_face_present)
        head_down = _float_value(row.get("pose_Rx")) > math.radians(20)
        side_turn = abs(_float_value(row.get("pose_Ry"))) > math.radians(30)
        hand_near_face = _hand_near_face(hand_bbox, face_bbox)
        hand_occlusion = _hand_occlusion(hand_bbox, face_bbox, row.get("hand_landmarks"))
        large_motion = _large_motion(previous_person_bbox, person_bbox)
        offscreen = bool(
            person_bbox is None
            and face_bbox is None
            and mediapipe_pose_present is False
            and mediapipe_face_present is False
        )

        flags.append(
            {
                "face_visible": face_visible,
                "low_confidence": low_confidence,
                "head_down": head_down,
                "side_turn": side_turn,
                "hand_near_face": hand_near_face,
                "hand_occlusion": hand_occlusion,
                "large_motion": large_motion,
                "offscreen": offscreen,
                "person_visible": person_bbox is not None,
                "hand_visible": hand_bbox is not None,
            }
        )
        previous_person_bbox = person_bbox
    return flags


def frame_flags_from_mediapipe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, bool]]:
    """Return behavior flags from MediaPipe Holistic-like per-frame rows."""
    flags: list[dict[str, bool]] = []
    previous_person_bbox: tuple[float, float, float, float] | None = None
    for row in rows:
        face_bbox = _bbox_from_row(row, "face_bbox")
        hand_bbox = _bbox_from_row(row, "hand_bbox")
        person_bbox = _bbox_from_row(row, "person_bbox")
        mediapipe_face_present = _optional_bool(
            row,
            "mediapipe_face_landmarks_present",
            "mediapipe_face_present",
            "mediapipe_face_tracking_success",
        )
        mediapipe_pose_present = _optional_bool(
            row,
            "mediapipe_pose_present",
            "mediapipe_pose_landmarks_present",
            "mediapipe_pose_tracking_success",
        )
        left_hand_present = _optional_bool(
            row,
            "mediapipe_left_hand_landmarks_present",
            "mediapipe_left_hand_present",
        )
        right_hand_present = _optional_bool(
            row,
            "mediapipe_right_hand_landmarks_present",
            "mediapipe_right_hand_present",
        )
        hand_present = _optional_bool(
            row,
            "mediapipe_hand_landmarks_present",
            "mediapipe_hand_present",
            "mediapipe_hands_present",
        )
        if hand_present is None:
            hand_present = bool(left_hand_present or right_hand_present)
        visible_pose_count = int(_float_value(row.get("pose_visible_landmark_count")))
        person_visible = bool(
            mediapipe_pose_present is True
            and (visible_pose_count >= 8 or person_bbox is not None)
        )
        hand_visible = bool(hand_present is True or hand_bbox is not None)
        face_visible = bool(mediapipe_face_present is True or face_bbox is not None)
        offscreen = bool(
            not face_visible
            and not person_visible
            and not hand_visible
            and mediapipe_face_present is False
            and mediapipe_pose_present is False
        )

        flags.append(
            {
                "face_visible": face_visible,
                "low_confidence": mediapipe_face_present is False,
                "head_down": _mediapipe_head_down(row),
                "side_turn": _mediapipe_side_turn(row),
                "hand_near_face": _hand_near_face(hand_bbox, face_bbox),
                "hand_occlusion": _hand_occlusion(hand_bbox, face_bbox, row.get("hand_landmarks")),
                "large_motion": _large_motion(previous_person_bbox, person_bbox),
                "offscreen": offscreen,
                "person_visible": person_visible,
                "hand_visible": hand_visible,
            }
        )
        previous_person_bbox = person_bbox
    return flags


def merge_openface_and_mediapipe_flags(
    openface_flags: Iterable[dict[str, Any]],
    mediapipe_flags: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge OpenFace head/face fields with MediaPipe hand/person/offscreen fields."""
    openface_list = list(openface_flags)
    mediapipe_list = list(mediapipe_flags)
    merged: list[dict[str, Any]] = []
    length = max(len(openface_list), len(mediapipe_list))
    for idx in range(length):
        openface = openface_list[idx] if idx < len(openface_list) else {}
        mediapipe = mediapipe_list[idx] if idx < len(mediapipe_list) else {}
        merged.append(
            {
                "face_visible": bool(openface.get("face_visible", mediapipe.get("face_visible", False))),
                "low_confidence": bool(openface.get("low_confidence", mediapipe.get("low_confidence", False))),
                "head_down": bool(openface.get("head_down", mediapipe.get("head_down", False))),
                "side_turn": bool(openface.get("side_turn", mediapipe.get("side_turn", False))),
                "hand_near_face": bool(mediapipe.get("hand_near_face", openface.get("hand_near_face", False))),
                "hand_occlusion": bool(mediapipe.get("hand_occlusion", openface.get("hand_occlusion", False))),
                "large_motion": bool(mediapipe.get("large_motion", openface.get("large_motion", False))),
                "offscreen": bool(mediapipe.get("offscreen", openface.get("offscreen", False))),
                "person_visible": bool(mediapipe.get("person_visible", openface.get("person_visible", False))),
                "hand_visible": bool(mediapipe.get("hand_visible", openface.get("hand_visible", False))),
            }
        )
    return merged


def aggregate_behavior_window(
    window: dict[str, Any],
    frame_flags: Iterable[dict[str, Any]],
    source: dict[str, Any],
    detectors: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-frame flags into one JSON-serializable behavior row."""
    frames = list(frame_flags)
    sampled_frame_count = len(frames)
    usable_frame_count = sum(1 for frame in frames if frame.get("usable", True))
    usable_frames = [frame for frame in frames if frame.get("usable", True)]
    source_fields = _source_fields(window, source)

    row: dict[str, Any] = {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "source_mp4_path": source_fields["source_mp4_path"],
        "clip_start_seconds": source_fields["clip_start_seconds"],
        "clip_end_seconds": source_fields["clip_end_seconds"],
        "sampled_frame_count": sampled_frame_count,
        "usable_frame_count": usable_frame_count,
    }
    for flag_name, ratio_name in _FLAG_TO_RATIO.items():
        if usable_frame_count:
            row[ratio_name] = (
                sum(1 for frame in usable_frames if bool(frame.get(flag_name))) / usable_frame_count
            )
        else:
            row[ratio_name] = 0.0
    row["detectors"] = dict(detectors or source.get("detectors") or {})
    if "behavior_backend" in row["detectors"]:
        row["behavior_backend"] = row["detectors"]["behavior_backend"]
    return _json_ready(row)


def extract_behavior_flags(
    windows: Iterable[dict[str, Any]],
    *,
    out: Path | str,
    failures_out: Path | str,
    fps: float = 2.0,
    max_windows: int | None = None,
    precomputed_frame_flags: dict[str, list[dict[str, Any]]] | None = None,
    detectors: dict[str, Any] | None = None,
    openface_cache_root: Path | str | None = None,
    openface_encoder_profile: str = "openface_temporal_v1",
    behavior_backend: str | None = None,
    mediapipe_frame_rows_by_sample: dict[str, list[dict[str, Any]]] | None = None,
    max_frames_per_window: int | None = None,
    mediapipe_max_image_size: int = 640,
    progress_out: Path | str | None = None,
    progress_every: int = 10,
) -> dict[str, int]:
    """Write behavior rows from precomputed flags, OpenFace cache rows, or failures."""
    selected = list(windows)
    if max_windows is not None:
        selected = selected[:max_windows]
    out_path = _initialize_jsonl(out)
    write_json([], failures_out)
    progress = _ProgressReporter(progress_out, total=len(selected), every=progress_every)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    mediapipe_backend = None
    if behavior_backend == "mediapipe_holistic_v1" and mediapipe_frame_rows_by_sample is None:
        try:
            mediapipe_backend = _MediaPipeHolisticBackend(
                fps=fps,
                max_frames_per_window=max_frames_per_window,
                max_image_size=mediapipe_max_image_size,
            )
        except RuntimeError as exc:
            mediapipe_backend = exc
    def record_success(row: dict[str, Any]) -> None:
        rows.append(row)
        _append_jsonl_row(row, out_path)

    def record_failure(failure: dict[str, Any]) -> None:
        failures.append(failure)
        write_json(failures, failures_out)

    for index, window in enumerate(selected, start=1):
        sample_id = str(window.get("sample_id", ""))
        progress.start(index, sample_id, written=len(rows), failures=len(failures))
        if precomputed_frame_flags is not None and sample_id in precomputed_frame_flags:
            record_success(
                aggregate_behavior_window(
                window,
                precomputed_frame_flags[sample_id],
                _default_source_from_window(window),
                detectors=detectors,
            )
            )
            progress.done(index, sample_id, status="success", written=len(rows), failures=len(failures))
            continue
        if behavior_backend == "mediapipe_holistic_v1":
            openface_flags: list[dict[str, Any]] = []
            source = _default_source_from_window(window)
            cached_clip_path = None
            if openface_cache_root is not None:
                cached_clip_path = _openface_cache_clip_path(
                    window,
                    cache_root=openface_cache_root,
                    encoder_profile=openface_encoder_profile,
                )
                cache_result = _openface_cache_behavior_source(
                    window,
                    cache_root=openface_cache_root,
                    encoder_profile=openface_encoder_profile,
                )
                if "failure" not in cache_result:
                    openface_flags = frame_flags_from_openface_rows(cache_result["rows"])
                    source = cache_result["source"]
            if mediapipe_frame_rows_by_sample is not None and sample_id in mediapipe_frame_rows_by_sample:
                mediapipe_rows = mediapipe_frame_rows_by_sample[sample_id]
            elif isinstance(mediapipe_backend, RuntimeError):
                record_failure(
                    _backend_failure(
                        window,
                        stage="video_behavior_flags_mediapipe_holistic",
                        error_type="dependency_missing",
                        error=str(mediapipe_backend),
                    )
                )
                progress.done(index, sample_id, status="failure", written=len(rows), failures=len(failures))
                continue
            elif mediapipe_backend is not None:
                try:
                    mediapipe_rows = mediapipe_backend.rows_for_window(
                        window,
                        clip_path=cached_clip_path,
                    )
                except RuntimeError as exc:
                    record_failure(
                        _backend_failure(
                            window,
                            stage="video_behavior_flags_mediapipe_holistic",
                            error_type="mediapipe_extract_failed",
                            error=str(exc),
                        )
                    )
                    progress.done(index, sample_id, status="failure", written=len(rows), failures=len(failures))
                    continue
            else:
                record_failure(
                    _backend_failure(
                        window,
                        stage="video_behavior_flags_mediapipe_holistic",
                        error_type="mediapipe_rows_missing",
                        error="MediaPipe Holistic frame rows were not provided for this sample.",
                    )
                )
                progress.done(index, sample_id, status="failure", written=len(rows), failures=len(failures))
                continue
            mediapipe_flags = frame_flags_from_mediapipe_rows(mediapipe_rows)
            record_success(
                aggregate_behavior_window(
                    window,
                    merge_openface_and_mediapipe_flags(openface_flags, mediapipe_flags),
                    source,
                    detectors={
                        "behavior_backend": "mediapipe_holistic_v1",
                        "face": "openface_csv_cache" if openface_flags else "mediapipe_holistic_fallback",
                        "hand": "mediapipe_holistic",
                        "person": "mediapipe_holistic",
                    },
                )
            )
            progress.done(index, sample_id, status="success", written=len(rows), failures=len(failures))
            continue
        if openface_cache_root is not None:
            cache_result = _openface_cache_behavior_source(
                window,
                cache_root=openface_cache_root,
                encoder_profile=openface_encoder_profile,
            )
            if "failure" in cache_result:
                record_failure(cache_result["failure"])
                progress.done(index, sample_id, status="failure", written=len(rows), failures=len(failures))
                continue
            record_success(
                aggregate_behavior_window(
                    window,
                    frame_flags_from_openface_rows(cache_result["rows"]),
                    cache_result["source"],
                    detectors={
                        "face": "openface_csv_cache",
                        "hand": "unavailable",
                        "person": "unavailable",
                    },
                )
            )
            progress.done(index, sample_id, status="success", written=len(rows), failures=len(failures))
            continue
        record_failure(_not_implemented_failure(window, fps))
        progress.done(index, sample_id, status="failure", written=len(rows), failures=len(failures))
    if hasattr(mediapipe_backend, "close"):
        mediapipe_backend.close()
    write_json(failures, failures_out)
    progress.finish(written=len(rows), failures=len(failures))
    return {
        "selected_count": len(selected),
        "written_count": len(rows),
        "failure_count": len(failures),
    }


def audit_behavior_flags(
    flags: Path | str,
    *,
    openface_embeddings: Path | str | None = None,
    top_k: int = 10,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Summarize behavior flag JSONL rows and select windows for human review."""
    rows = load_jsonl(flags)
    openface_by_sample = (
        load_openface_review_metadata(openface_embeddings) if openface_embeddings is not None else {}
    )
    successful_rows = [row for row in rows if _has_all_ratios(row)]
    ratio_summary = {
        ratio_name: _ratio_stats(_numeric_ratio_values(successful_rows, ratio_name))
        for ratio_name in BEHAVIOR_RATIO_NAMES
    }
    review_source_rows = [_review_row(row, openface_by_sample) for row in successful_rows]
    rng = random.Random(random_seed)
    random_rows = (
        rng.sample(review_source_rows, k=min(top_k, len(review_source_rows)))
        if review_source_rows
        else []
    )
    report = {
        "window_count": len(rows),
        "success_count": len(successful_rows),
        "missing_count": len(rows) - len(successful_rows),
        "ratios": ratio_summary,
        "review_sets": {
            "top_head_down_ratio": _top_rows(review_source_rows, "head_down_ratio", top_k),
            "top_hand_occlusion_ratio": _top_rows(review_source_rows, "hand_occlusion_ratio", top_k),
            "top_offscreen_ratio": _top_rows(review_source_rows, "offscreen_ratio", top_k),
            "top_large_motion_ratio": _top_rows(review_source_rows, "large_motion_ratio", top_k),
            "random_windows": random_rows,
        },
    }
    return _json_ready(report)


def write_behavior_audit_markdown(report: dict[str, Any], output: Path | str) -> Path:
    """Write review-set rows as a compact Markdown table with clip coordinates."""
    columns = [
        "review_set",
        "sample_id",
        "event_id",
        "subject_id",
        "source_mp4_path",
        "clip_start_seconds",
        "clip_end_seconds",
        *BEHAVIOR_RATIO_NAMES,
        "openface_mask_value",
        "openface_quality_flags",
    ]
    lines = [
        "# Video Behavior Flags Audit",
        "",
        f"- window_count: {report.get('window_count', 0)}",
        f"- success_count: {report.get('success_count', 0)}",
        f"- missing_count: {report.get('missing_count', 0)}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for review_set, rows in (report.get("review_sets") or {}).items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            values = [review_set, *[_markdown_cell(row.get(column)) for column in columns[1:]]]
            lines.append("| " + " | ".join(values) + " |")
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path} line {line_number} must decode to a JSON object")
            rows.append(payload)
    return rows


def load_openface_review_metadata(path: Path | str) -> dict[str, dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - numpy is required by the embedding stack.
        raise RuntimeError("numpy is required to read OpenFace embedding npz files") from exc

    metadata: dict[str, dict[str, Any]] = {}
    with np.load(path, allow_pickle=True) as loaded:
        sample_ids = loaded["sample_id"].astype(str).tolist()
        masks = loaded["modality_mask"] if "modality_mask" in loaded.files else None
        mask_values = loaded["mask_value"] if "mask_value" in loaded.files else None
        quality_values = loaded["quality_flags"].tolist() if "quality_flags" in loaded.files else ["{}"] * len(sample_ids)
        for idx, sample_id in enumerate(sample_ids):
            joined: dict[str, Any] = {}
            mask_value = _openface_mask_value(masks, mask_values, idx)
            if mask_value is not None:
                joined["openface_mask_value"] = mask_value
            quality = _parse_json_object(quality_values[idx])
            if quality:
                joined["openface_quality_flags"] = quality
            metadata[sample_id] = joined
    return metadata


def write_jsonl(rows: Iterable[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")
    return out


def _initialize_jsonl(output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("", encoding="utf-8")
    return out


def _append_jsonl_row(row: dict[str, Any], output: Path) -> None:
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


class _ProgressReporter:
    def __init__(self, output: Path | str | None, *, total: int, every: int = 10) -> None:
        self._path = Path(output) if output is not None else None
        self._total = max(0, int(total))
        self._every = max(1, int(every))
        self._started_at = time.monotonic()
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text("", encoding="utf-8")

    def start(self, index: int, sample_id: str, *, written: int, failures: int) -> None:
        self._write("start", index, sample_id, written=written, failures=failures, force=True)

    def done(self, index: int, sample_id: str, *, status: str, written: int, failures: int) -> None:
        self._write("done", index, sample_id, written=written, failures=failures, force=index == self._total, outcome=status)

    def finish(self, *, written: int, failures: int) -> None:
        self._write("finish", self._total, "-", written=written, failures=failures, force=True)

    def _write(
        self,
        status: str,
        index: int,
        sample_id: str,
        *,
        written: int,
        failures: int,
        force: bool = False,
        outcome: str | None = None,
    ) -> None:
        if self._path is None:
            return
        if not force and status != "start" and index % self._every != 0:
            return
        percent = (index / self._total * 100.0) if self._total else 100.0
        elapsed = time.monotonic() - self._started_at
        line = (
            f"{status} {index}/{self._total} {_progress_bar(index, self._total)} "
            f"{percent:5.1f}% sample_id={sample_id} written={written} "
            f"failures={failures} elapsed={elapsed:.1f}s"
        )
        if outcome is not None:
            line += f" status={outcome}"
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()


def _progress_bar(index: int, total: int, *, width: int = 24) -> str:
    if total <= 0:
        return "[" + "#" * width + "]"
    filled = min(width, max(0, int(round(width * index / total))))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _has_all_ratios(row: dict[str, Any]) -> bool:
    return all(_optional_float(row.get(ratio_name)) is not None for ratio_name in BEHAVIOR_RATIO_NAMES)


def _numeric_ratio_values(rows: Iterable[dict[str, Any]], ratio_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _optional_float(row.get(ratio_name))
        if value is not None:
            values.append(value)
    return values


def _ratio_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
    }


def _review_row(row: dict[str, Any], openface_by_sample: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id = str(row.get("sample_id", ""))
    review = {
        "sample_id": sample_id,
        "event_id": str(row.get("event_id", "")),
        "subject_id": str(row.get("subject_id", "")),
        "source_mp4_path": str(row.get("source_mp4_path", "")),
        "clip_start_seconds": _number_or_none(row.get("clip_start_seconds")),
        "clip_end_seconds": _number_or_none(row.get("clip_end_seconds")),
    }
    for ratio_name in BEHAVIOR_RATIO_NAMES:
        review[ratio_name] = _float_value(row.get(ratio_name))
    review.update(openface_by_sample.get(sample_id, {}))
    return review


def _top_rows(rows: list[dict[str, Any]], ratio_name: str, top_k: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (-_float_value(row.get(ratio_name)), str(row.get("sample_id", ""))),
    )[:top_k]


def _openface_mask_value(masks: Any, mask_values: Any, idx: int) -> int | None:
    value = None
    if masks is not None:
        row = masks[idx]
        try:
            value = row[2] if len(row) > 2 else row[0]
        except TypeError:
            value = row
    elif mask_values is not None:
        value = mask_values[idx]
    if value is None:
        return None
    return int(value)


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if value is None:
        return {}
    text = str(value)
    if not text:
        return {}
    decoded = json.loads(text)
    if not isinstance(decoded, dict):
        raise ValueError("quality_flags entries must decode to JSON objects")
    return decoded


def _markdown_cell(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True)
    elif value is None:
        text = ""
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def write_json(payload: Any, output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return out


def _is_low_confidence(
    row: dict[str, Any],
    mediapipe_face_present: bool | None,
) -> bool:
    confidence = _optional_float(row.get("confidence"))
    yolo_face_confidence = _optional_float(
        row.get("yolo_face_confidence", row.get("face_confidence"))
    )
    return bool(
        (confidence is not None and confidence < 0.80)
        or (yolo_face_confidence is not None and yolo_face_confidence < 0.50)
        or mediapipe_face_present is False
    )


def _hand_near_face(
    hand_bbox: tuple[float, float, float, float] | None,
    face_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if hand_bbox is None or face_bbox is None:
        return False
    hand_center = _bbox_center(hand_bbox)
    face_center = _bbox_center(face_bbox)
    face_width = max(0.0, face_bbox[2] - face_bbox[0])
    distance = math.dist(hand_center, face_center)
    return bool(face_width > 0 and distance < 0.75 * face_width)


def _hand_occlusion(
    hand_bbox: tuple[float, float, float, float] | None,
    face_bbox: tuple[float, float, float, float] | None,
    hand_landmarks: Any,
) -> bool:
    if face_bbox is None:
        return False
    if hand_bbox is not None and _intersection_area(hand_bbox, face_bbox) / _bbox_area(face_bbox) > 0.10:
        return True
    points = _points(hand_landmarks)
    if not points:
        return False
    inside = sum(1 for point in points if _point_in_bbox(point, face_bbox))
    return inside / len(points) > 0.20


def _large_motion(
    previous_bbox: tuple[float, float, float, float] | None,
    person_bbox: tuple[float, float, float, float] | None,
) -> bool:
    if previous_bbox is None or person_bbox is None:
        return False
    height = max(0.0, person_bbox[3] - person_bbox[1])
    if height <= 0:
        return False
    return math.dist(_bbox_center(previous_bbox), _bbox_center(person_bbox)) / height > 0.10


def _mediapipe_head_down(row: dict[str, Any]) -> bool:
    pitch = _optional_float(
        row.get("mediapipe_head_pitch", row.get("head_pitch", row.get("pose_Rx")))
    )
    return bool(pitch is not None and pitch > math.radians(20))


def _mediapipe_side_turn(row: dict[str, Any]) -> bool:
    yaw = _optional_float(row.get("mediapipe_head_yaw", row.get("head_yaw", row.get("pose_Ry"))))
    return bool(yaw is not None and abs(yaw) > math.radians(30))


def _bbox_from_row(row: dict[str, Any], key: str) -> tuple[float, float, float, float] | None:
    value = row.get(key)
    if value is None and key == "face_bbox":
        value = _bbox_from_columns(row, "face")
    if value is None and key == "person_bbox":
        value = _bbox_from_columns(row, "person")
    return _coerce_bbox(value)


def _bbox_from_columns(row: dict[str, Any], prefix: str) -> Any:
    if all(name in row for name in (f"{prefix}_x1", f"{prefix}_y1", f"{prefix}_x2", f"{prefix}_y2")):
        return [row[f"{prefix}_x1"], row[f"{prefix}_y1"], row[f"{prefix}_x2"], row[f"{prefix}_y2"]]
    if all(name in row for name in (f"{prefix}_x", f"{prefix}_y", f"{prefix}_w", f"{prefix}_h")):
        x = _float_value(row[f"{prefix}_x"])
        y = _float_value(row[f"{prefix}_y"])
        return [x, y, x + _float_value(row[f"{prefix}_w"]), y + _float_value(row[f"{prefix}_h"])]
    return None


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, dict):
        if all(name in value for name in ("x1", "y1", "x2", "y2")):
            return tuple(_float_value(value[name]) for name in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
        if all(name in value for name in ("x", "y", "w", "h")):
            x = _float_value(value["x"])
            y = _float_value(value["y"])
            return (x, y, x + _float_value(value["w"]), y + _float_value(value["h"]))
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(_float_value(item) for item in value[:4])  # type: ignore[return-value]
    return None


def _source_fields(window: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    merged = _default_source_from_window(window)
    merged.update({key: value for key, value in source.items() if value is not None})
    return merged


def _default_source_from_window(window: dict[str, Any]) -> dict[str, Any]:
    candidate = _first_video_candidate(window)
    return {
        "source_mp4_path": str(
            window.get("source_mp4_path")
            or window.get("mp4_path")
            or candidate.get("mp4_path")
            or candidate.get("source_mp4_path")
            or ""
        ),
        "clip_start_seconds": _number_or_none(
            window.get("clip_start_seconds", candidate.get("clip_start_seconds"))
        ),
        "clip_end_seconds": _number_or_none(
            window.get("clip_end_seconds", candidate.get("clip_end_seconds"))
        ),
    }


def _first_video_candidate(window: dict[str, Any]) -> dict[str, Any]:
    candidates = window.get("video_candidates") or []
    if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def _not_implemented_failure(window: dict[str, Any], fps: float) -> dict[str, Any]:
    source = _default_source_from_window(window)
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "stage": "video_behavior_flags",
        "error_type": "not_implemented",
        "error": (
            "Real video detector extraction is not implemented in this first pass; "
            "provide precomputed frame flags or add MediaPipe/YOLO detector integration."
        ),
        "source_path": source["source_mp4_path"],
        "clip_start_seconds": source["clip_start_seconds"],
        "clip_end_seconds": source["clip_end_seconds"],
        "fps": fps,
        "recoverable": True,
    }


def _backend_failure(
    window: dict[str, Any],
    *,
    stage: str,
    error_type: str,
    error: str,
) -> dict[str, Any]:
    source = _default_source_from_window(window)
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "stage": stage,
        "error_type": error_type,
        "error": error,
        "source_path": source["source_mp4_path"],
        "clip_start_seconds": source["clip_start_seconds"],
        "clip_end_seconds": source["clip_end_seconds"],
        "recoverable": True,
    }


class _MediaPipeHolisticBackend:
    def __init__(
        self,
        *,
        fps: float,
        max_frames_per_window: int | None = None,
        max_image_size: int = 640,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on server runtime
            raise RuntimeError(f"missing MediaPipe Holistic dependency: {exc.name}") from exc
        self._cv2 = cv2
        self._fps = float(fps)
        self._max_frames_per_window = max_frames_per_window
        self._max_image_size = int(max_image_size)
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def rows_for_window(self, window: dict[str, Any], *, clip_path: Path | None = None) -> list[dict[str, Any]]:
        source = _default_source_from_window(window)
        source_path = clip_path or Path(str(source["source_mp4_path"] or ""))
        if not source_path.is_file():
            raise RuntimeError(f"source video not found: {source_path}")
        start = None if clip_path is not None else _optional_float(source["clip_start_seconds"])
        end = None if clip_path is not None else _optional_float(source["clip_end_seconds"])
        max_frames = self._max_frames_per_window
        if max_frames is None and start is not None and end is not None and end > start:
            max_frames = max(1, int(math.ceil((end - start) * self._fps)))
        if max_frames is None:
            max_frames = 20
        try:
            from daily_multimodal.embeddings.face_real import _sample_video_frames
        except ImportError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"could not import video sampler: {exc}") from exc
        frames = _sample_video_frames(
            source_path,
            start_seconds=start,
            end_seconds=end,
            fps=self._fps,
            max_frames=max_frames,
        )
        if not frames:
            raise RuntimeError(f"no readable sampled frames: {source_path}")
        rows: list[dict[str, Any]] = []
        for _frame_index, _timestamp, frame in frames:
            frame = _resize_for_mediapipe(frame, self._cv2, self._max_image_size)
            rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            result = self._holistic.process(rgb)
            rows.append(_mediapipe_result_row(result, frame.shape))
        return rows

    def close(self) -> None:
        self._holistic.close()


def _mediapipe_result_row(result: Any, frame_shape: Any) -> dict[str, Any]:
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    face_points = _landmark_points(getattr(result, "face_landmarks", None), width, height)
    pose_points = _landmark_points(getattr(result, "pose_landmarks", None), width, height, min_visibility=0.5)
    left_hand_points = _landmark_points(getattr(result, "left_hand_landmarks", None), width, height)
    right_hand_points = _landmark_points(getattr(result, "right_hand_landmarks", None), width, height)
    hand_points = [*left_hand_points, *right_hand_points]
    pose_visible_count = len(pose_points)
    return {
        "mediapipe_face_landmarks_present": bool(face_points),
        "mediapipe_pose_landmarks_present": pose_visible_count >= 8,
        "mediapipe_left_hand_landmarks_present": bool(left_hand_points),
        "mediapipe_right_hand_landmarks_present": bool(right_hand_points),
        "pose_visible_landmark_count": pose_visible_count,
        "face_bbox": _bbox_from_points(face_points),
        "person_bbox": _bbox_from_points(pose_points),
        "hand_bbox": _bbox_from_points(hand_points),
        "hand_landmarks": hand_points,
    }


def _resize_for_mediapipe(frame: Any, cv2: Any, max_image_size: int) -> Any:
    if max_image_size <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_image_size:
        return frame
    scale = max_image_size / float(longest)
    return cv2.resize(frame, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))))


def _landmark_points(
    landmark_list: Any,
    width: int,
    height: int,
    *,
    min_visibility: float | None = None,
) -> list[tuple[float, float]]:
    landmarks = getattr(landmark_list, "landmark", None)
    if not landmarks:
        return []
    points: list[tuple[float, float]] = []
    for landmark in landmarks:
        visibility = getattr(landmark, "visibility", None)
        if min_visibility is not None and visibility is not None and float(visibility) < min_visibility:
            continue
        points.append((float(landmark.x) * width, float(landmark.y) * height))
    return points


def _bbox_from_points(points: list[tuple[float, float]]) -> list[float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _openface_cache_behavior_source(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    encoder_profile: str,
) -> dict[str, Any]:
    sample_id = str(window.get("sample_id", ""))
    cache_dir = Path(cache_root) / "openface" / sample_id / encoder_profile
    metadata_path = cache_dir / "openface_target.json"
    if not metadata_path.is_file():
        return {
            "failure": _openface_cache_failure(
                window,
                "cache_metadata_missing",
                f"OpenFace cache metadata not found: {metadata_path}",
                source_path=str(metadata_path),
            )
        }
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "failure": _openface_cache_failure(
                window,
                "cache_metadata_invalid",
                str(exc),
                source_path=str(metadata_path),
            )
        }

    csv_path = Path(str(metadata.get("target_csv_path") or cache_dir / "openface.csv"))
    if not csv_path.is_file():
        return {
            "failure": _openface_cache_failure(
                window,
                "openface_csv_missing",
                f"OpenFace CSV not found: {csv_path}",
                source_path=str(csv_path),
            )
        }
    try:
        rows = _read_openface_csv_rows(csv_path)
    except ValueError as exc:
        return {
            "failure": _openface_cache_failure(
                window,
                "openface_csv_invalid",
                str(exc),
                source_path=str(csv_path),
            )
        }
    if not rows:
        return {
            "failure": _openface_cache_failure(
                window,
                "openface_csv_empty",
                f"OpenFace CSV has no frame rows: {csv_path}",
                source_path=str(csv_path),
            )
        }

    source = _default_source_from_window(window)
    source["source_mp4_path"] = str(metadata.get("source_path") or source["source_mp4_path"])
    if metadata.get("clip_start_seconds") is not None:
        source["clip_start_seconds"] = _number_or_none(metadata.get("clip_start_seconds"))
    if metadata.get("clip_end_seconds") is not None:
        source["clip_end_seconds"] = _number_or_none(metadata.get("clip_end_seconds"))
    return {"rows": rows, "source": source}


def _openface_cache_clip_path(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    encoder_profile: str,
) -> Path | None:
    clip_path = Path(cache_root) / "openface" / str(window.get("sample_id", "")) / encoder_profile / "window.mp4"
    return clip_path if clip_path.is_file() else None


def _read_openface_csv_rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"OpenFace CSV has no header: {path}")
            rows: list[dict[str, Any]] = []
            for row in reader:
                rows.append(
                    {
                        str(key).strip(): value.strip() if isinstance(value, str) else value
                        for key, value in row.items()
                        if key is not None
                    }
                )
            return rows
    except OSError as exc:
        raise ValueError(str(exc)) from exc


def _openface_cache_failure(
    window: dict[str, Any],
    error_type: str,
    error: str,
    *,
    source_path: str,
) -> dict[str, Any]:
    source = _default_source_from_window(window)
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "stage": "video_behavior_flags_openface_cache",
        "error_type": error_type,
        "error": error,
        "source_path": source_path,
        "source_mp4_path": source["source_mp4_path"],
        "clip_start_seconds": source["clip_start_seconds"],
        "clip_end_seconds": source["clip_end_seconds"],
        "recoverable": True,
    }


def _optional_bool(row: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in row:
            return _bool_or_none(row[key])
    return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "unknown", "none", "null", "nan"}:
            return None
        return text in {"1", "true", "yes", "y"}
    return bool(value)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _float_value(value: Any) -> float:
    number = _optional_float(value)
    return 0.0 if number is None else number


def _number_or_none(value: Any) -> float | int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(number) if number.is_integer() else number


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return max(1e-12, max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1]))


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _points(value: Any) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)):
        return []
    points: list[tuple[float, float]] = []
    for item in value:
        if isinstance(item, dict) and "x" in item and "y" in item:
            points.append((_float_value(item["x"]), _float_value(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((_float_value(item[0]), _float_value(item[1])))
    return points


def _point_in_bbox(point: tuple[float, float], bbox: tuple[float, float, float, float]) -> bool:
    return bool(bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return str(value)
