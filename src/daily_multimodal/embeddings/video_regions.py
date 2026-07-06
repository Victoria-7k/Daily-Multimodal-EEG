from __future__ import annotations

import json
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np


VIDEO_REGIONS = ("2x_face_roi", "upper_body", "full_frame")
REGION_CLIP_FRAME_COUNT = 16
REGION_CLIP_FPS = 2.0
ClipWriter = Callable[..., None]
UpperBodyLocalizer = Callable[[dict[str, Any]], list[int] | tuple[int, int, int, int] | None]


def build_video_region_cache(
    *,
    window_index_path: Path | str,
    out_root: Path | str,
    roi_cache_root: Path | str | None = None,
    roi_encoder_profile: str = "openface_temporal_v1",
    regions: tuple[str, ...] | list[str] = VIDEO_REGIONS,
    upper_body_localizer: UpperBodyLocalizer | None = None,
    clip_writer: ClipWriter | None = None,
    start_index: int = 0,
    max_windows: int | None = None,
    manifest_out: Path | str | None = None,
    failures_out: Path | str | None = None,
    skip_existing: bool = True,
    progress_every: int = 0,
) -> dict[str, int]:
    selected_regions = tuple(regions)
    for region in selected_regions:
        if region not in VIDEO_REGIONS:
            raise ValueError(f"unsupported video region: {region}")
    windows = _read_jsonl(Path(window_index_path))
    if start_index:
        windows = windows[int(start_index) :]
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if clip_writer is None and set(selected_regions).issubset({"upper_body", "full_frame"}):
        skipped_existing = 0
        processed_windows = 0
        for group_windows in _source_event_groups(windows):
            group_result = _write_source_event_group_regions(
                group_windows,
                regions=selected_regions,
                out_root=out,
                upper_body_localizer=upper_body_localizer,
                skip_existing=skip_existing,
            )
            rows.extend(group_result["rows"])
            failures.extend(group_result["failures"])
            skipped_existing += int(group_result["skipped_existing_count"])
            processed_windows += len(group_windows)
            if progress_every and processed_windows % int(progress_every) == 0:
                print(
                    f"processed_windows={processed_windows}/{len(windows)} "
                    f"written_or_found={len(rows)} failures={len(failures)}",
                    file=sys.stderr,
                    flush=True,
                )
        manifest_path = Path(manifest_out) if manifest_out is not None else out / "video_regions_manifest.jsonl"
        failures_path = Path(failures_out) if failures_out is not None else out / "video_regions_failures.json"
        _write_jsonl(manifest_path, rows)
        _write_json(failures_path, failures)
        return {
            "selected_count": int(len(windows)),
            "written_count": int(len(rows)),
            "skipped_existing_count": int(skipped_existing),
            "failure_count": int(len(failures)),
        }

    skipped_existing = 0
    for window_index, window in enumerate(windows, start=1):
        frame_cache: dict[tuple[str, Any, Any], list[Any]] = {}
        if clip_writer is None:
            def writer(**kwargs: Any) -> None:
                _opencv_clip_writer(**kwargs, frame_cache=frame_cache)
        else:
            writer = clip_writer
        for region in selected_regions:
            try:
                row = _write_region(
                    window,
                    region=region,
                    out_root=out,
                    roi_cache_root=Path(roi_cache_root) if roi_cache_root is not None else None,
                    roi_encoder_profile=roi_encoder_profile,
                    upper_body_localizer=upper_body_localizer,
                    clip_writer=writer,
                    skip_existing=skip_existing,
                )
            except RegionBuildError as exc:
                failures.append(exc.failure)
                continue
            if row.get("cache_status") == "existing":
                skipped_existing += 1
            rows.append(row)
        if progress_every and window_index % int(progress_every) == 0:
            print(
                f"processed_windows={window_index}/{len(windows)} "
                f"written_or_found={len(rows)} failures={len(failures)}",
                file=sys.stderr,
                flush=True,
            )

    manifest_path = Path(manifest_out) if manifest_out is not None else out / "video_regions_manifest.jsonl"
    failures_path = Path(failures_out) if failures_out is not None else out / "video_regions_failures.json"
    _write_jsonl(manifest_path, rows)
    _write_json(failures_path, failures)
    return {
        "selected_count": int(len(windows)),
        "written_count": int(len(rows)),
        "skipped_existing_count": int(skipped_existing),
        "failure_count": int(len(failures)),
    }


def _write_region(
    window: dict[str, Any],
    *,
    region: str,
    out_root: Path,
    roi_cache_root: Path | None,
    roi_encoder_profile: str,
    upper_body_localizer: UpperBodyLocalizer | None,
    clip_writer: ClipWriter,
    skip_existing: bool = True,
) -> dict[str, Any]:
    sample_id = str(window.get("sample_id", ""))
    region_dir = out_root / region / sample_id
    output_video = region_dir / "window.mp4"
    sidecar = region_dir / "region.json"
    source = _default_source_from_window(window)
    crop_bbox = None
    effective_region = region
    fallback_full_frame = False
    region_source = "source_video"
    existing = _existing_region_row(output_video, sidecar, skip_existing=skip_existing)
    if existing is not None:
        return existing

    if region == "2x_face_roi":
        roi_video = _roi_video_path(roi_cache_root, sample_id, roi_encoder_profile)
        if roi_video is None or not roi_video.is_file():
            raise RegionBuildError(_failure(window, region, "source_video_missing", str(roi_video or "")))
        source_video = roi_video
        clip_start = None
        clip_end = None
        region_source = "cached_2x_face_roi"
    else:
        source_video = Path(str(source["source_video_path"] or ""))
        if not source_video.is_file():
            raise RegionBuildError(_failure(window, region, "source_video_missing", str(source_video)))
        clip_start = source["clip_start_seconds"]
        clip_end = source["clip_end_seconds"]
        if region == "upper_body":
            located = upper_body_localizer(window) if upper_body_localizer is not None else _upper_body_bbox_from_window(window)
            crop_bbox = _coerce_bbox(located)
            if crop_bbox is None:
                fallback_full_frame = True
                effective_region = "full_frame"

    clip_writer(
        source_video=source_video,
        output_video=output_video,
        clip_start_seconds=clip_start,
        clip_end_seconds=clip_end,
        crop_bbox=crop_bbox,
    )
    row = {
        "sample_id": sample_id,
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "region": region,
        "effective_region": effective_region,
        "source_video_path": str(source_video),
        "output_video_path": str(output_video),
        "clip_start_seconds": _json_ready(clip_start),
        "clip_end_seconds": _json_ready(clip_end),
        "crop_bbox": crop_bbox,
        "region_source": region_source,
        "upper_body_fallback_full_frame": bool(fallback_full_frame),
        "cache_status": "written",
    }
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    _write_json(sidecar, row)
    return row


def _source_event_groups(windows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: OrderedDict[tuple[str, str], list[dict[str, Any]]] = OrderedDict()
    for window in windows:
        source = _default_source_from_window(window)
        key = (str(source["source_video_path"] or ""), str(window.get("event_id", "")))
        groups.setdefault(key, []).append(window)
    return list(groups.values())


def _write_source_event_group_regions(
    windows: list[dict[str, Any]],
    *,
    regions: tuple[str, ...],
    out_root: Path,
    upper_body_localizer: UpperBodyLocalizer | None,
    skip_existing: bool,
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video region cache preparation requires opencv-python or opencv-python-headless") from exc
    if not windows:
        return {"rows": [], "failures": [], "skipped_existing_count": 0}
    source = _default_source_from_window(windows[0])
    source_video = Path(str(source["source_video_path"] or ""))
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped_existing = 0
    if not source_video.is_file():
        for window in windows:
            for region in regions:
                failures.append(_failure(window, region, "source_video_missing", str(source_video)))
        return {"rows": rows, "failures": failures, "skipped_existing_count": skipped_existing}

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        for window in windows:
            for region in regions:
                failures.append(_failure(window, region, "source_video_open_failed", str(source_video)))
        return {"rows": rows, "failures": failures, "skipped_existing_count": skipped_existing}
    tasks: list[dict[str, Any]] = []
    targets: dict[int, list[int]] = {}
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0:
            source_fps = 30.0
        for window in windows:
            window_source = _default_source_from_window(window)
            start_frame, end_frame = _source_window_frame_bounds(
                window_source,
                source_fps=source_fps,
                frame_count=frame_count,
            )
            target_indices = [
                int(round(float(raw_index)))
                for raw_index in np.linspace(start_frame, end_frame - 1, num=REGION_CLIP_FRAME_COUNT)
            ]
            for region in regions:
                sample_id = str(window.get("sample_id", ""))
                region_dir = out_root / region / sample_id
                output_video = region_dir / "window.mp4"
                sidecar = region_dir / "region.json"
                existing = _existing_region_row(output_video, sidecar, skip_existing=skip_existing)
                if existing is not None:
                    rows.append(existing)
                    skipped_existing += 1
                    continue
                crop_bbox = None
                effective_region = region
                fallback_full_frame = False
                if region == "upper_body":
                    located = upper_body_localizer(window) if upper_body_localizer is not None else _upper_body_bbox_from_window(window)
                    crop_bbox = _coerce_bbox(located)
                    if crop_bbox is None:
                        effective_region = "full_frame"
                        fallback_full_frame = True
                task_index = len(tasks)
                tasks.append(
                    {
                        "window": window,
                        "region": region,
                        "output_video": output_video,
                        "sidecar": sidecar,
                        "crop_bbox": crop_bbox,
                        "effective_region": effective_region,
                        "fallback_full_frame": fallback_full_frame,
                        "clip_start_seconds": window_source["clip_start_seconds"],
                        "clip_end_seconds": window_source["clip_end_seconds"],
                        "target_indices": target_indices,
                        "frames": [],
                    }
                )
                for frame_index in target_indices:
                    targets.setdefault(frame_index, []).append(task_index)
        if tasks and targets:
            min_frame = min(targets)
            max_frame = max(targets)
            cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)
            for frame_index in range(min_frame, max_frame + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                for task_index in targets.get(frame_index, []):
                    task = tasks[task_index]
                    task["frames"].append(_prepare_region_frame(frame, crop_bbox=task["crop_bbox"], cv2=cv2))
    finally:
        cap.release()

    for task in tasks:
        window = task["window"]
        frames = task["frames"]
        if not frames:
            failures.append(_failure(window, str(task["region"]), "source_video_no_frames", str(source_video)))
            continue
        if len(frames) != REGION_CLIP_FRAME_COUNT:
            frames = _resample_frame_list(frames, REGION_CLIP_FRAME_COUNT)
        output_video = task["output_video"]
        output_video.parent.mkdir(parents=True, exist_ok=True)
        _write_region_frames(frames, output_video, cv2=cv2)
        row = {
            "sample_id": str(window.get("sample_id", "")),
            "event_id": str(window.get("event_id", "")),
            "subject_id": str(window.get("subject_id", "")),
            "region": str(task["region"]),
            "effective_region": str(task["effective_region"]),
            "source_video_path": str(source_video),
            "output_video_path": str(output_video),
            "clip_start_seconds": _json_ready(task["clip_start_seconds"]),
            "clip_end_seconds": _json_ready(task["clip_end_seconds"]),
            "crop_bbox": task["crop_bbox"],
            "region_source": "source_video_event_group",
            "upper_body_fallback_full_frame": bool(task["fallback_full_frame"]),
            "cache_status": "written",
        }
        task["sidecar"].parent.mkdir(parents=True, exist_ok=True)
        _write_json(task["sidecar"], row)
        rows.append(row)
    return {"rows": rows, "failures": failures, "skipped_existing_count": skipped_existing}


def _existing_region_row(output_video: Path, sidecar: Path, *, skip_existing: bool) -> dict[str, Any] | None:
    if not skip_existing or not output_video.is_file() or not sidecar.is_file():
        return None
    try:
        row = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict):
        return None
    row["cache_status"] = "existing"
    return row


class RegionBuildError(Exception):
    def __init__(self, failure: dict[str, Any]) -> None:
        super().__init__(str(failure.get("error_type", "region_build_failed")))
        self.failure = failure


def _opencv_clip_writer(
    *,
    source_video: Path | str,
    output_video: Path | str,
    clip_start_seconds: float | int | None,
    clip_end_seconds: float | int | None,
    crop_bbox: list[int] | None,
    frame_cache: dict[tuple[str, Any, Any], list[Any]] | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("video region cache preparation requires opencv-python or opencv-python-headless") from exc
    source = Path(source_video)
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_key = (str(source), _json_ready(clip_start_seconds), _json_ready(clip_end_seconds))
    raw_frames = frame_cache.get(cache_key) if frame_cache is not None else None
    if raw_frames is None:
        raw_frames = _sample_source_window_frames(
            source,
            clip_start_seconds=clip_start_seconds,
            clip_end_seconds=clip_end_seconds,
            cv2=cv2,
        )
        if frame_cache is not None:
            frame_cache[cache_key] = raw_frames
    frames = [_prepare_region_frame(frame, crop_bbox=crop_bbox, cv2=cv2) for frame in raw_frames]
    _write_region_frames(frames, output, cv2=cv2)


def _sample_source_window_frames(
    source: Path,
    *,
    clip_start_seconds: float | int | None,
    clip_end_seconds: float | int | None,
    cv2: Any,
) -> list[Any]:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"could not open source video: {source}")
    window_frames: list[Any] = []
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0:
            source_fps = 30.0
        start_frame = int(round(float(clip_start_seconds or 0.0) * source_fps))
        if clip_end_seconds is None:
            end_frame = max(start_frame + 1, frame_count)
        else:
            end_frame = int(round(float(clip_end_seconds) * source_fps))
        if frame_count > 0:
            start_frame = max(0, min(start_frame, frame_count - 1))
            end_frame = max(start_frame + 1, min(end_frame, frame_count))
        else:
            end_frame = max(start_frame + 1, end_frame)
        target_indices = [int(round(float(raw_index))) for raw_index in np.linspace(start_frame, end_frame - 1, num=REGION_CLIP_FRAME_COUNT)]
        target_position = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _frame_index in range(start_frame, end_frame):
            ok, frame = cap.read()
            if not ok:
                break
            while target_position < len(target_indices) and _frame_index == target_indices[target_position]:
                window_frames.append(frame)
                target_position += 1
            if target_position >= len(target_indices):
                break
    finally:
        cap.release()
    if not window_frames:
        raise ValueError(f"no frames sampled from source video: {source}")
    if len(window_frames) == REGION_CLIP_FRAME_COUNT:
        return window_frames
    sample_indices = np.linspace(0, len(window_frames) - 1, num=REGION_CLIP_FRAME_COUNT)
    return [window_frames[int(round(float(raw_index)))] for raw_index in sample_indices]


def _source_window_frame_bounds(source: dict[str, Any], *, source_fps: float, frame_count: int) -> tuple[int, int]:
    start_frame = int(round(float(source["clip_start_seconds"] or 0.0) * source_fps))
    if source["clip_end_seconds"] is None:
        end_frame = max(start_frame + 1, frame_count)
    else:
        end_frame = int(round(float(source["clip_end_seconds"]) * source_fps))
    if frame_count > 0:
        start_frame = max(0, min(start_frame, frame_count - 1))
        end_frame = max(start_frame + 1, min(end_frame, frame_count))
    else:
        end_frame = max(start_frame + 1, end_frame)
    return start_frame, end_frame


def _resample_frame_list(frames: list[Any], count: int) -> list[Any]:
    if count <= 0 or not frames:
        return []
    if len(frames) == count:
        return frames
    if len(frames) == 1:
        return [frames[0]] * count
    sample_indices = np.linspace(0, len(frames) - 1, num=count)
    return [frames[int(round(float(raw_index)))] for raw_index in sample_indices]


def _write_region_frames(frames: list[Any], output: Path, *, cv2: Any) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), REGION_CLIP_FPS, (int(width), int(height)))
    if not writer.isOpened():
        raise ValueError(f"could not open output video writer: {output}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _prepare_region_frame(frame: Any, *, crop_bbox: list[int] | None, cv2: Any) -> Any:
    prepared = frame
    if crop_bbox is not None:
        height, width = prepared.shape[:2]
        x1, y1, x2, y2 = crop_bbox
        left = max(0, min(width - 1, int(x1)))
        top = max(0, min(height - 1, int(y1)))
        right = max(left + 1, min(width, int(x2)))
        bottom = max(top + 1, min(height, int(y2)))
        prepared = prepared[top:bottom, left:right]
    height, width = prepared.shape[:2]
    target_width = 640
    target_height = max(2, int(round(float(height) * float(target_width) / float(max(1, width)))))
    if target_height % 2:
        target_height += 1
    return cv2.resize(prepared, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _roi_video_path(roi_cache_root: Path | None, sample_id: str, profile: str) -> Path | None:
    if roi_cache_root is None:
        return None
    return roi_cache_root / "openface" / sample_id / profile / "window.mp4"


def _upper_body_bbox_from_window(window: dict[str, Any]) -> list[int] | None:
    for key in ("upper_body_bbox", "person_bbox"):
        bbox = _coerce_bbox(window.get(key))
        if bbox is not None:
            return bbox
    face_bbox = _coerce_face_bbox(window.get("main_face_bbox"))
    if face_bbox is not None:
        return _upper_body_from_face_bbox(face_bbox)
    face_presence = window.get("face_presence")
    if isinstance(face_presence, dict):
        return _upper_body_bbox_from_window(face_presence)
    return None


def _coerce_face_bbox(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "w", "h")):
            x = _number(value["x"])
            y = _number(value["y"])
            return [int(round(x)), int(round(y)), int(round(x + _number(value["w"]))), int(round(y + _number(value["h"])))]
        return _coerce_bbox(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x, y, w, h = [_number(item) for item in value[:4]]
        return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]
    return None


def _upper_body_from_face_bbox(bbox: list[int]) -> list[int]:
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return [
        max(0, x1 - width),
        max(0, y1 - int(0.5 * height)),
        x2 + width,
        y2 + int(3.0 * height),
    ]


def _coerce_bbox(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            raw = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("x", "y", "w", "h")):
            x = _number(value["x"])
            y = _number(value["y"])
            raw = [x, y, x + _number(value["w"]), y + _number(value["h"])]
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw = list(value[:4])
    else:
        return None
    return [int(round(_number(item))) for item in raw]


def _default_source_from_window(window: dict[str, Any]) -> dict[str, Any]:
    candidate = _first_video_candidate(window)
    return {
        "source_video_path": str(
            window.get("source_mp4_path")
            or window.get("mp4_path")
            or candidate.get("mp4_path")
            or candidate.get("source_mp4_path")
            or _first_candidate_path(window)
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


def _first_candidate_path(window: dict[str, Any]) -> str:
    candidates = window.get("candidate_mp4_paths") or []
    if isinstance(candidates, list) and candidates:
        return str(candidates[0])
    return ""


def _failure(window: dict[str, Any], region: str, error_type: str, source_path: str) -> dict[str, Any]:
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "region": region,
        "stage": "video_region_cache",
        "error_type": error_type,
        "source_path": source_path,
        "recoverable": True,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, allow_nan=False) + "\n")
    return path


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _number_or_none(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
