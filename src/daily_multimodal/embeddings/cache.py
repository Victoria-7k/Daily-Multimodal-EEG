from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list


AudioExtractor = Callable[[Path, float, float, Path], None]
FacePresenceDetector = Callable[[Path, float, float], "FacePresenceResult"]


@dataclass(frozen=True)
class FacePresenceResult:
    has_face: bool
    detector: str
    frame_count: int
    detected_frame_count: int
    start_seconds: float
    end_seconds: float
    max_face_count: int = 0
    main_face_bbox: list[int] | None = None
    main_face_area_ratio: float = 0.0
    detected_orientations: list[str] = field(default_factory=list)
    retained_without_detected_face: bool = False
    retention_reason: str = ""


@dataclass(frozen=True)
class FacePresenceTask:
    window: dict[str, Any]
    source: Path
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class FacePresenceEvaluation:
    window: dict[str, Any]
    source_path: str
    result: FacePresenceResult


@dataclass(frozen=True)
class RealCacheProfiles:
    eeg: str = "eeg_real_frozen_v1"
    wear: str = "wear_sequence_v1"
    face: str = "openface_temporal_v1"
    audio: str = "wavlm_frozen_v1"

    def for_modality(self, modality: str) -> str:
        return getattr(self, modality)


class DependencyMissingError(RuntimeError):
    pass


def build_cache_key(sample_id: str, modality: str, encoder_profile: str) -> str:
    parts = [sample_id, modality, encoder_profile]
    if any(not part or _unsafe_path_part(part) for part in parts):
        raise ValueError(f"cache key parts must not be empty, absolute, or path-traversing: {parts!r}")
    return "/".join(parts)


def prepare_real_embedding_cache(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    report_out: Path | str,
    failures_out: Path | str,
    profiles: RealCacheProfiles | None = None,
    audio_extractor: AudioExtractor | None = None,
    filter_no_face: bool = False,
    face_detector: FacePresenceDetector | None = None,
    filtered_window_index_out: Path | str | None = None,
) -> dict[str, Any]:
    profiles = profiles or RealCacheProfiles()
    extractor = audio_extractor or extract_audio_clip_ffmpeg
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    failures: list[EmbeddingFailure] = []
    selected_windows, face_filter_summary = _filter_windows_by_face_presence(
        windows,
        profiles=profiles,
        filter_no_face=filter_no_face,
        face_detector=face_detector or detect_face_presence_opencv_haar,
        failures=failures,
    )
    summary: dict[str, Any] = {
        "stage": 12,
        "window_count": len(windows),
        "selected_window_count": len(selected_windows),
        "cache_root": str(root),
        "failures_path": str(failures_out),
        "filtered_window_index_path": str(filtered_window_index_out) if filtered_window_index_out else "",
        "face_filter": face_filter_summary,
        "modalities": {
            "eeg": _empty_modality_summary(),
            "wear": _empty_modality_summary(),
            "face": _empty_modality_summary(),
            "audio": _empty_modality_summary(),
        },
    }
    summary["modalities"]["face"]["missing_count"] += face_filter_summary["dropped_count"]

    for window in selected_windows:
        _prepare_audio_cache(window, root, profiles, extractor, summary, failures)
        _prepare_face_cache(window, root, profiles, summary, failures)
        _prepare_eeg_cache(window, root, profiles, summary, failures)
        _prepare_wear_cache(window, root, profiles, summary, failures)

    for modality, modality_summary in summary["modalities"].items():
        modality_summary["failure_count"] = sum(
            1 for failure in failures if failure.modality == modality
        )
    if filtered_window_index_out:
        _write_jsonl(selected_windows, filtered_window_index_out)
    write_failure_list(failures, failures_out)
    _write_readiness_report(summary, report_out)
    return summary


def detect_face_presence_opencv_haar(
    source: Path,
    start_seconds: float,
    end_seconds: float,
) -> FacePresenceResult:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise DependencyMissingError(f"missing OpenCV dependency: {exc.name}") from exc

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise DependencyMissingError(f"OpenCV Haar cascade not found: {cascade_path}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")

    frame_count = 0
    detected_frame_count = 0
    max_face_count = 0
    main_face_bbox: list[int] | None = None
    main_face_area_ratio = 0.0
    detected_orientations: list[str] = []

    try:
        for timestamp in face_presence_sample_times(start_seconds, end_seconds, max_frames=10):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            detection_summary = _opencv_haar_detection_summary(frame, cascade_path)
            frame_count += 1
            if detection_summary["has_face"]:
                detected_frame_count += 1
                max_face_count = max(max_face_count, int(detection_summary["max_face_count"]))
                if (
                    main_face_bbox is None
                    or float(detection_summary["main_face_area_ratio"]) > main_face_area_ratio
                ):
                    main_face_bbox = detection_summary["main_face_bbox"]
                    main_face_area_ratio = float(detection_summary["main_face_area_ratio"])
                for orientation in detection_summary["detected_orientations"]:
                    if orientation not in detected_orientations:
                        detected_orientations.append(orientation)
    finally:
        capture.release()

    if frame_count == 0:
        raise RuntimeError(f"video produced no readable frames: {source}")
    return FacePresenceResult(
        has_face=detected_frame_count > 0,
        detector="opencv_haar_frontalface_default",
        frame_count=frame_count,
        detected_frame_count=detected_frame_count,
        start_seconds=float(start_seconds),
        end_seconds=float(end_seconds),
        max_face_count=max_face_count,
        main_face_bbox=main_face_bbox,
        main_face_area_ratio=main_face_area_ratio,
        detected_orientations=detected_orientations,
    )


def detect_face_presence_opencv_haar_batch(
    source: Path,
    tasks: list[FacePresenceTask],
) -> list[FacePresenceResult]:
    if not tasks:
        return []
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise DependencyMissingError(f"missing OpenCV dependency: {exc.name}") from exc

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise DependencyMissingError(f"OpenCV Haar cascade not found: {cascade_path}")
    fast_results = _detect_face_presence_midpoints_parallel(source, tasks, cascade_path=cascade_path)
    if len(fast_results) == len(tasks):
        return fast_results

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")

    try:
        results: list[FacePresenceResult] = []
        for task in tasks:
            frame_count = 0
            detected_frame_count = 0
            max_face_count = 0
            main_face_bbox: list[int] | None = None
            main_face_area_ratio = 0.0
            detected_orientations: list[str] = []
            for timestamp in face_presence_sample_times(task.start_seconds, task.end_seconds, max_frames=10):
                capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                detection_summary = _opencv_haar_detection_summary(frame, cascade_path)
                frame_count += 1
                if detection_summary["has_face"]:
                    detected_frame_count += 1
                    max_face_count = max(max_face_count, int(detection_summary["max_face_count"]))
                    if (
                        main_face_bbox is None
                        or float(detection_summary["main_face_area_ratio"]) > main_face_area_ratio
                    ):
                        main_face_bbox = detection_summary["main_face_bbox"]
                        main_face_area_ratio = float(detection_summary["main_face_area_ratio"])
                    for orientation in detection_summary["detected_orientations"]:
                        if orientation not in detected_orientations:
                            detected_orientations.append(orientation)
            if frame_count == 0:
                raise RuntimeError(f"video produced no readable frames: {source}")
            results.append(
                FacePresenceResult(
                    has_face=detected_frame_count > 0,
                    detector="opencv_haar_frontalface_default",
                    frame_count=frame_count,
                    detected_frame_count=detected_frame_count,
                    start_seconds=float(task.start_seconds),
                    end_seconds=float(task.end_seconds),
                    max_face_count=max_face_count,
                    main_face_bbox=main_face_bbox,
                    main_face_area_ratio=main_face_area_ratio,
                    detected_orientations=detected_orientations,
                )
            )
        return results
    finally:
        capture.release()


def face_presence_summary_from_detections(
    *,
    frame_shape: tuple[int, int],
    detections_by_orientation: dict[str, list[tuple[int, int, int, int]]],
) -> dict[str, Any]:
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    all_faces: list[tuple[str, tuple[int, int, int, int]]] = []
    for orientation, faces in detections_by_orientation.items():
        for face in faces:
            x, y, w, h = [int(value) for value in face]
            all_faces.append((orientation, (x, y, w, h)))

    if not all_faces:
        return {
            "has_face": False,
            "max_face_count": 0,
            "main_face_bbox": None,
            "main_face_area_ratio": 0.0,
            "detected_orientations": [],
        }

    _orientation, main_face = max(all_faces, key=lambda item: item[1][2] * item[1][3])
    main_area = int(main_face[2]) * int(main_face[3])
    frame_area = max(1, frame_height * frame_width)
    return {
        "has_face": True,
        "max_face_count": max(len(faces) for faces in detections_by_orientation.values()),
        "main_face_bbox": [int(value) for value in main_face],
        "main_face_area_ratio": float(main_area / frame_area),
        "detected_orientations": [
            orientation
            for orientation, faces in detections_by_orientation.items()
            if faces
        ],
    }


def _opencv_haar_detection_summary(image: Any, cascade_path: Path) -> dict[str, Any]:
    import cv2

    frame_height, frame_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frontal = cv2.CascadeClassifier(str(cascade_path))
    frontal_alt = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt.xml")
    )
    profile = cv2.CascadeClassifier(
        str(Path(cv2.data.haarcascades) / "haarcascade_profileface.xml")
    )

    detections: dict[str, list[tuple[int, int, int, int]]] = {
        "upright": _bbox_list(
            frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        ),
    }
    if not frontal_alt.empty():
        detections["upright_alt"] = _bbox_list(
            frontal_alt.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        )
    rot180 = cv2.rotate(gray, cv2.ROTATE_180)
    detections["rot180"] = _rotate_180_bboxes(
        _bbox_list(frontal.detectMultiScale(rot180, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))),
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if not profile.empty():
        detections["profile"] = _bbox_list(
            profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        )
        detections["profile_flipped"] = _flip_horizontal_bboxes(
            _bbox_list(
                profile.detectMultiScale(
                    cv2.flip(gray, 1),
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                )
            ),
            frame_width=frame_width,
        )
    return face_presence_summary_from_detections(
        frame_shape=(frame_height, frame_width),
        detections_by_orientation=detections,
    )


def _bbox_list(faces: Any) -> list[tuple[int, int, int, int]]:
    return [tuple(int(value) for value in face) for face in faces]


def _rotate_180_bboxes(
    faces: list[tuple[int, int, int, int]],
    *,
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int]]:
    return [
        (int(frame_width - x - w), int(frame_height - y - h), int(w), int(h))
        for x, y, w, h in faces
    ]


def _flip_horizontal_bboxes(
    faces: list[tuple[int, int, int, int]],
    *,
    frame_width: int,
) -> list[tuple[int, int, int, int]]:
    return [(int(frame_width - x - w), int(y), int(w), int(h)) for x, y, w, h in faces]


def _detect_face_presence_midpoints_parallel(
    source: Path,
    tasks: list[FacePresenceTask],
    *,
    cascade_path: Path,
) -> list[FacePresenceResult]:
    workers = _face_filter_workers()
    indexed_results: dict[int, FacePresenceResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(_detect_face_presence_midpoint, source, task, cascade_path): idx
            for idx, task in enumerate(tasks)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                indexed_results[idx] = future.result()
            except Exception:
                return []
    return [indexed_results[idx] for idx in range(len(tasks))]


def _detect_face_presence_midpoint(
    source: Path,
    task: FacePresenceTask,
    cascade_path: Path,
) -> FacePresenceResult:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise DependencyMissingError(f"missing OpenCV dependency: {exc.name}") from exc

    midpoint = (float(task.start_seconds) + float(task.end_seconds)) / 2.0
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{midpoint:.6f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        "scale=960:-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout:
        raise RuntimeError("ffmpeg did not return a sampled frame")
    image = cv2.imdecode(np.frombuffer(completed.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("sampled frame could not be decoded")
    detection_summary = _opencv_haar_detection_summary(image, cascade_path)
    detected = bool(detection_summary["has_face"])
    return FacePresenceResult(
        has_face=detected,
        detector="opencv_haar_frontalface_default_alt_profile_rot180_ffmpeg_midpoint",
        frame_count=1,
        detected_frame_count=1 if detected else 0,
        start_seconds=float(task.start_seconds),
        end_seconds=float(task.end_seconds),
        max_face_count=int(detection_summary["max_face_count"]),
        main_face_bbox=detection_summary["main_face_bbox"],
        main_face_area_ratio=float(detection_summary["main_face_area_ratio"]),
        detected_orientations=list(detection_summary["detected_orientations"]),
    )


def _face_filter_workers() -> int:
    value = os.environ.get("FACE_FILTER_WORKERS", "8")
    try:
        return max(1, int(value))
    except ValueError:
        return 8


def group_face_presence_tasks(
    tasks: list[FacePresenceTask],
    *,
    max_gap_seconds: float = 1.0,
) -> list[list[FacePresenceTask]]:
    if not tasks:
        return []
    ordered = sorted(tasks, key=lambda task: (float(task.start_seconds), float(task.end_seconds)))
    groups: list[list[FacePresenceTask]] = [[ordered[0]]]
    current_end = float(ordered[0].end_seconds)
    for task in ordered[1:]:
        if float(task.start_seconds) <= current_end + float(max_gap_seconds):
            groups[-1].append(task)
            current_end = max(current_end, float(task.end_seconds))
            continue
        groups.append([task])
        current_end = float(task.end_seconds)
    return groups


def _sample_face_detections_ffmpeg(
    source: Path,
    *,
    detector: Any,
    start_seconds: float,
    end_seconds: float,
    fps: float = 0.5,
) -> list[tuple[float, bool]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []

    duration = max(0.01, float(end_seconds) - float(start_seconds))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(start_seconds):.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"fps={float(fps):g},scale=320:-2",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout:
        return []

    data = completed.stdout
    detections: list[tuple[float, bool]] = []
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"
    cursor = 0
    while True:
        jpeg_start = data.find(start_marker, cursor)
        if jpeg_start < 0:
            break
        jpeg_end = data.find(end_marker, jpeg_start + 2)
        if jpeg_end < 0:
            break
        chunk = data[jpeg_start : jpeg_end + 2]
        cursor = jpeg_end + 2
        image = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        timestamp = float(start_seconds) + len(detections) / float(fps)
        detections.append((round(timestamp, 6), bool(len(faces) > 0)))
    return detections


def face_presence_results_from_frame_detections(
    tasks: list[FacePresenceTask],
    frame_detections: list[tuple[float, bool]],
    *,
    detector: str,
) -> list[FacePresenceResult]:
    results: list[FacePresenceResult] = []
    for task in tasks:
        matched = [
            detected
            for timestamp, detected in frame_detections
            if float(task.start_seconds) <= float(timestamp) < float(task.end_seconds)
        ]
        if not matched:
            matched = [
                detected
                for timestamp, detected in frame_detections
                if abs(float(timestamp) - float(task.start_seconds)) < 1e-6
            ]
        results.append(
            FacePresenceResult(
                has_face=any(matched),
                detector=detector,
                frame_count=len(matched),
                detected_frame_count=sum(1 for detected in matched if detected),
                start_seconds=float(task.start_seconds),
                end_seconds=float(task.end_seconds),
            )
        )
    return results


def face_presence_sample_times(
    start_seconds: float,
    end_seconds: float,
    *,
    max_frames: int = 10,
) -> list[float]:
    start = float(start_seconds)
    end = float(end_seconds)
    if max_frames <= 1 or end <= start:
        return [round(start, 6)]
    duration = end - start
    frames = min(max_frames, max(2, int(duration) + 1))
    if frames == 2:
        return [round(start, 6), round(end, 6)]
    step = duration / float(frames - 1)
    return [round(start + step * idx, 6) for idx in range(frames)]


def extract_audio_clip_ffmpeg(
    source: Path,
    start_seconds: float,
    end_seconds: float,
    output: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise DependencyMissingError("ffmpeg executable was not found on PATH")
    duration = end_seconds - start_seconds
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.6f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg failed"
        raise RuntimeError(message)


def _prepare_audio_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    extractor: AudioExtractor,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "audio"
    encoder_profile = profiles.audio
    candidate = _select_video_candidate(window)
    if candidate is None:
        _record_missing(summary, failures, window, modality, encoder_profile, "select_audio_source", "<missing-video-candidate>")
        return
    source = Path(str(candidate.get("mp4_path", "")))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "select_audio_source", str(source))
        return
    start_seconds, end_seconds = _candidate_clip_bounds(candidate, window)
    if end_seconds <= start_seconds:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "extraction_failed",
            f"invalid clip bounds: {start_seconds} >= {end_seconds}",
            str(source),
        )
        return
    output = _sample_cache_dir(cache_root, "audio_clips", window, encoder_profile) / "audio.wav"
    metadata_out = output.with_suffix(".json")
    try:
        if not output.is_file():
            extractor(source, start_seconds, end_seconds, output)
        if not output.is_file():
            raise RuntimeError("audio extractor did not create output wav")
    except DependencyMissingError as exc:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "dependency_missing",
            str(exc),
            str(source),
        )
        return
    except Exception as exc:
        _record_failure(
            summary,
            failures,
            window,
            modality,
            encoder_profile,
            "extract_audio_clip",
            "extraction_failed",
            str(exc),
            str(source),
        )
        return
    _write_json(
        metadata_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "wav_path": str(output),
            "clip_start_seconds": start_seconds,
            "clip_end_seconds": end_seconds,
            "target_sample_rate_hz": 16000,
            "target_channels": 1,
        },
    )
    _record_ready(summary, modality, metadata_out)


def _filter_windows_by_face_presence(
    windows: list[dict[str, Any]],
    *,
    profiles: RealCacheProfiles,
    filter_no_face: bool,
    face_detector: FacePresenceDetector,
    failures: list[EmbeddingFailure],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = {
        "enabled": bool(filter_no_face),
        "requested_windows": len(windows),
        "kept_count": len(windows),
        "dropped_count": 0,
        "dropped_no_face_count": 0,
        "dropped_failure_count": 0,
        "dropped_windows": [],
    }
    if not filter_no_face:
        return list(windows), summary

    if face_detector is detect_face_presence_opencv_haar:
        return _filter_windows_by_face_presence_batched(
            windows,
            profiles=profiles,
            failures=failures,
            summary=summary,
        )

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    evaluations: list[FacePresenceEvaluation] = []
    encoder_profile = profiles.face
    for window in windows:
        candidate = _select_video_candidate(window)
        if candidate is None:
            failure = _failure(
                window,
                "face",
                encoder_profile,
                "face_presence_filter",
                "source_missing",
                "required video source for face detection is missing",
                "<missing-video-candidate>",
            )
            failures.append(failure)
            dropped.append(_dropped_window_record(window, "source_missing", None))
            continue
        source = Path(str(candidate.get("mp4_path", "")))
        if not source.is_file():
            failure = _failure(
                window,
                "face",
                encoder_profile,
                "face_presence_filter",
                "source_missing",
                "required video source for face detection is missing",
                str(source),
            )
            failures.append(failure)
            dropped.append(_dropped_window_record(window, "source_missing", None))
            continue
        start_seconds, end_seconds = _candidate_clip_bounds(candidate, window)
        try:
            result = face_detector(source, start_seconds, end_seconds)
        except DependencyMissingError as exc:
            failures.append(
                _failure(
                    window,
                    "face",
                    encoder_profile,
                    "face_presence_filter",
                    "dependency_missing",
                    str(exc),
                    str(source),
                )
            )
            dropped.append(_dropped_window_record(window, "dependency_missing", None))
            continue
        except Exception as exc:
            failures.append(
                _failure(
                    window,
                    "face",
                    encoder_profile,
                    "face_presence_filter",
                    "face_detection_failed",
                    str(exc),
                    str(source),
                )
            )
            dropped.append(_dropped_window_record(window, "face_detection_failed", None))
            continue

        evaluations.append(FacePresenceEvaluation(window=window, source_path=str(source), result=result))

    kept.extend(
        _finalize_face_presence_evaluations(
            evaluations,
            profiles=profiles,
            failures=failures,
            dropped=dropped,
        )
    )

    summary["kept_count"] = len(kept)
    summary["dropped_count"] = len(dropped)
    summary["dropped_no_face_count"] = sum(1 for row in dropped if row["reason"] == "no_face_detected")
    summary["dropped_failure_count"] = summary["dropped_count"] - summary["dropped_no_face_count"]
    summary["dropped_windows"] = dropped
    return kept, summary


def _finalize_face_presence_evaluations(
    evaluations: list[FacePresenceEvaluation],
    *,
    profiles: RealCacheProfiles,
    failures: list[EmbeddingFailure],
    dropped: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    encoder_profile = profiles.face
    events_with_face = {
        str(evaluation.window.get("event_id", ""))
        for evaluation in evaluations
        if evaluation.result.has_face and evaluation.window.get("event_id")
    }
    for evaluation in evaluations:
        result_payload = asdict(evaluation.result)
        event_id = str(evaluation.window.get("event_id", ""))
        retain_by_context = bool(
            not evaluation.result.has_face
            and event_id
            and event_id in events_with_face
        )
        if evaluation.result.has_face or retain_by_context:
            if retain_by_context:
                result_payload["retained_without_detected_face"] = True
                result_payload["retention_reason"] = "same_event_face_detected"
            kept_window = dict(evaluation.window)
            kept_window["face_presence"] = result_payload
            kept.append(kept_window)
            continue

        failures.append(
            _failure(
                evaluation.window,
                "face",
                encoder_profile,
                "face_presence_filter",
                "no_face_detected",
                "no face detected in sampled window frames",
                evaluation.source_path,
            )
        )
        dropped.append(_dropped_window_record(evaluation.window, "no_face_detected", result_payload))
    return kept


def _filter_windows_by_face_presence_batched(
    windows: list[dict[str, Any]],
    *,
    profiles: RealCacheProfiles,
    failures: list[EmbeddingFailure],
    summary: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_by_source: dict[str, list[FacePresenceTask]] = {}
    dropped: list[dict[str, Any]] = []
    encoder_profile = profiles.face
    for window in windows:
        candidate = _select_video_candidate(window)
        if candidate is None:
            failures.append(
                _failure(
                    window,
                    "face",
                    encoder_profile,
                    "face_presence_filter",
                    "source_missing",
                    "required video source for face detection is missing",
                    "<missing-video-candidate>",
                )
            )
            dropped.append(_dropped_window_record(window, "source_missing", None))
            continue
        source = Path(str(candidate.get("mp4_path", "")))
        if not source.is_file():
            failures.append(
                _failure(
                    window,
                    "face",
                    encoder_profile,
                    "face_presence_filter",
                    "source_missing",
                    "required video source for face detection is missing",
                    str(source),
                )
            )
            dropped.append(_dropped_window_record(window, "source_missing", None))
            continue
        start_seconds, end_seconds = _candidate_clip_bounds(candidate, window)
        tasks_by_source.setdefault(str(source), []).append(
            FacePresenceTask(
                window=window,
                source=source,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )

    kept: list[dict[str, Any]] = []
    evaluations: list[FacePresenceEvaluation] = []
    for source_text, tasks in tasks_by_source.items():
        try:
            results = detect_face_presence_opencv_haar_batch(Path(source_text), tasks)
        except DependencyMissingError as exc:
            for task in tasks:
                failures.append(
                    _failure(
                        task.window,
                        "face",
                        encoder_profile,
                        "face_presence_filter",
                        "dependency_missing",
                        str(exc),
                        source_text,
                    )
                )
                dropped.append(_dropped_window_record(task.window, "dependency_missing", None))
            continue
        except Exception as exc:
            for task in tasks:
                failures.append(
                    _failure(
                        task.window,
                        "face",
                        encoder_profile,
                        "face_presence_filter",
                        "face_detection_failed",
                        str(exc),
                        source_text,
                    )
                )
                dropped.append(_dropped_window_record(task.window, "face_detection_failed", None))
            continue

        for task, result in zip(tasks, results):
            evaluations.append(
                FacePresenceEvaluation(window=task.window, source_path=source_text, result=result)
            )

    kept.extend(
        _finalize_face_presence_evaluations(
            evaluations,
            profiles=profiles,
            failures=failures,
            dropped=dropped,
        )
    )

    summary["kept_count"] = len(kept)
    summary["dropped_count"] = len(dropped)
    summary["dropped_no_face_count"] = sum(1 for row in dropped if row["reason"] == "no_face_detected")
    summary["dropped_failure_count"] = summary["dropped_count"] - summary["dropped_no_face_count"]
    summary["dropped_windows"] = dropped
    return kept, summary


def _prepare_face_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "face"
    encoder_profile = profiles.face
    candidate = _select_video_candidate(window)
    if candidate is None:
        _record_missing(summary, failures, window, modality, encoder_profile, "select_face_source", "<missing-video-candidate>")
        return
    source = Path(str(candidate.get("mp4_path", "")))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "select_face_source", str(source))
        return
    start_seconds, end_seconds = _candidate_clip_bounds(candidate, window)
    output_dir = _sample_cache_dir(cache_root, "openface", window, encoder_profile)
    record_out = output_dir / "openface_target.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "target_csv_path": str(output_dir / "openface.csv"),
            "clip_start_seconds": start_seconds,
            "clip_end_seconds": end_seconds,
            "openface_required": False,
            "note": "Stage 12 records the target CSV path; OpenFace runs in stage 14.",
        },
    )
    _record_ready(summary, modality, record_out)


def _prepare_eeg_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "eeg"
    encoder_profile = profiles.eeg
    source = Path(str(window.get("eeg_bdf_path") or ""))
    if not source.is_file():
        _record_missing(summary, failures, window, modality, encoder_profile, "prepare_eeg_window", str(source))
        return
    record_out = _sample_cache_dir(cache_root, "eeg_windows", window, encoder_profile) / "window.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_path": str(source),
            "window_start_time": window.get("window_start_time", ""),
            "window_end_time": window.get("window_end_time", ""),
            "source_sampling_frequency_hz": window.get("eeg_sampling_frequency"),
            "target_resample_hz": 250,
            "target_window_samples": 2500,
        },
    )
    _record_ready(summary, modality, record_out)


def _prepare_wear_cache(
    window: dict[str, Any],
    cache_root: Path,
    profiles: RealCacheProfiles,
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
) -> None:
    modality = "wear"
    encoder_profile = profiles.wear
    source_paths = {
        "ppg": Path(str(window.get("wear_ppg_path") or "")),
        "gsr": Path(str(window.get("wear_gsr_path") or "")),
        "acc": Path(str(window.get("wear_acc_path") or "")),
    }
    missing = {name: path for name, path in source_paths.items() if not path.is_file()}
    if missing:
        source_text = ";".join(f"{name}={path}" for name, path in missing.items()) or "<missing-wear-source>"
        _record_missing(summary, failures, window, modality, encoder_profile, "prepare_wear_window", source_text)
        return
    record_out = _sample_cache_dir(cache_root, "wear_windows", window, encoder_profile) / "window.json"
    _write_json(
        record_out,
        {
            **_base_cache_record(window, modality, encoder_profile),
            "source_paths": {name: str(path) for name, path in source_paths.items()},
            "window_start_time": window.get("window_start_time", ""),
            "window_end_time": window.get("window_end_time", ""),
            "target_sample_rates_hz": {"ppg": 64, "gsr": 32, "acc": 32},
        },
    )
    _record_ready(summary, modality, record_out)


def _select_video_candidate(window: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [candidate for candidate in window.get("video_candidates", []) or [] if candidate.get("mp4_path")]
    if not candidates:
        for path in window.get("candidate_mp4_paths", []) or []:
            if path:
                return {"mp4_path": path, "clip_start_seconds": 0.0, "clip_end_seconds": _window_duration_seconds(window)}
        return None
    for candidate in candidates:
        if candidate.get("covers_window"):
            return candidate
    return candidates[0]


def _candidate_clip_bounds(candidate: dict[str, Any], window: dict[str, Any]) -> tuple[float, float]:
    source_path = str(candidate.get("mp4_path") or "")
    mp4_start_text = candidate.get("mp4_start_time")
    if source_path and mp4_start_text and window.get("window_start_time") and window.get("window_end_time"):
        try:
            mp4_start = parse_absolute_time(str(mp4_start_text))
            window_start = parse_absolute_time(str(window["window_start_time"]))
            window_end = parse_absolute_time(str(window["window_end_time"]))
            start = max(0.0, float((window_start - mp4_start).total_seconds()))
            end = max(start, float((window_end - mp4_start).total_seconds()))
            if candidate.get("duration_seconds") is not None:
                duration = float(candidate["duration_seconds"])
                start = min(start, duration)
                end = min(end, duration)
            return start, end
        except (TypeError, ValueError):
            pass
    start = _as_float(candidate.get("clip_start_seconds"), 0.0)
    end = candidate.get("clip_end_seconds")
    if end is None:
        end = start + _window_duration_seconds(window)
    return start, _as_float(end, start)


def _window_duration_seconds(window: dict[str, Any]) -> float:
    if window.get("window_size_seconds") is not None:
        return float(window["window_size_seconds"])
    try:
        start = parse_absolute_time(str(window["window_start_time"]))
        end = parse_absolute_time(str(window["window_end_time"]))
        return (end - start).total_seconds()
    except Exception:
        return 0.0


def _base_cache_record(window: dict[str, Any], modality: str, encoder_profile: str) -> dict[str, Any]:
    sample_id = str(window.get("sample_id", ""))
    return {
        "sample_id": sample_id,
        "event_id": window.get("event_id", ""),
        "subject_id": window.get("subject_id", ""),
        "modality": modality,
        "encoder_profile": encoder_profile,
        "cache_key": build_cache_key(sample_id, modality, encoder_profile),
    }


def _sample_cache_dir(
    cache_root: Path,
    modality_dir: str,
    window: dict[str, Any],
    encoder_profile: str,
) -> Path:
    sample_id = str(window.get("sample_id", ""))
    if _unsafe_path_part(sample_id) or _unsafe_path_part(encoder_profile):
        raise ValueError(f"cache key parts must be safe: {sample_id!r}, {encoder_profile!r}")
    return cache_root / modality_dir / sample_id / encoder_profile


def _record_ready(summary: dict[str, Any], modality: str, cache_record_path: Path) -> None:
    modality_summary = summary["modalities"][modality]
    modality_summary["ready_count"] += 1
    modality_summary["cache_entries"].append(str(cache_record_path))


def _record_missing(
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
    window: dict[str, Any],
    modality: str,
    encoder_profile: str,
    stage: str,
    source_path: str,
) -> None:
    _record_failure(
        summary,
        failures,
        window,
        modality,
        encoder_profile,
        stage,
        "source_missing",
        "required source file is missing",
        source_path or "<missing-source-path>",
    )


def _record_failure(
    summary: dict[str, Any],
    failures: list[EmbeddingFailure],
    window: dict[str, Any],
    modality: str,
    encoder_profile: str,
    stage: str,
    error_type: str,
    error: str,
    source_path: str,
) -> None:
    summary["modalities"][modality]["missing_count"] += 1
    failures.append(
        EmbeddingFailure(
            sample_id=str(window.get("sample_id") or "<missing-sample-id>"),
            event_id=str(window.get("event_id") or "<missing-event-id>"),
            subject_id=str(window.get("subject_id") or "<missing-subject-id>"),
            modality=modality,
            encoder_profile=encoder_profile,
            stage=stage,
            error_type=error_type,
            error=error,
            source_path=source_path or "<missing-source-path>",
            recoverable=True,
        )
    )


def _failure(
    window: dict[str, Any],
    modality: str,
    encoder_profile: str,
    stage: str,
    error_type: str,
    error: str,
    source_path: str,
) -> EmbeddingFailure:
    return EmbeddingFailure(
        sample_id=str(window.get("sample_id") or "<missing-sample-id>"),
        event_id=str(window.get("event_id") or "<missing-event-id>"),
        subject_id=str(window.get("subject_id") or "<missing-subject-id>"),
        modality=modality,
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path or "<missing-source-path>",
        recoverable=True,
    )


def _dropped_window_record(
    window: dict[str, Any],
    reason: str,
    face_presence: dict[str, Any] | None,
) -> dict[str, Any]:
    record = {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "reason": reason,
    }
    if face_presence is not None:
        record["face_presence"] = face_presence
    return record


def _write_readiness_report(summary: dict[str, Any], report_out: Path | str) -> Path:
    out = Path(report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Real Embedding Readiness Report",
        "",
        f"Window count: {summary['window_count']}",
        f"Selected windows: {summary['selected_window_count']}",
        f"Cache root: {summary['cache_root']}",
        f"Failures: {summary['failures_path']}",
        (
            "Face-filter kept: "
            f"{summary['face_filter']['kept_count']}, "
            f"dropped: {summary['face_filter']['dropped_count']}"
        ),
        "",
        "## Modality readiness",
        "",
    ]
    for modality in ("EEG", "Wear", "Face", "Audio"):
        values = summary["modalities"][modality.lower()]
        lines.append(
            f"- {modality} ready: {values['ready_count']}, "
            f"missing: {values['missing_count']}, failures: {values['failure_count']}"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _write_jsonl(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _empty_modality_summary() -> dict[str, Any]:
    return {
        "ready_count": 0,
        "missing_count": 0,
        "failure_count": 0,
        "cache_entries": [],
    }


def _unsafe_path_part(value: str) -> bool:
    return bool(
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).is_absolute()
        or re.search(r"(^|[._-])\.\.($|[._-])", value)
    )


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
