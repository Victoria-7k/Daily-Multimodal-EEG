from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.failures import EmbeddingFailure, write_failure_list
from daily_multimodal.alignment.time_utils import parse_absolute_time


LOW_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MIN_SUCCESS_RATE = 0.50
OPENFACE_CLIP_FPS = 0.5
OPENFACE_CLIP_WIDTH = 640
FACE_ROI_CROP_SCALE = 2.0
FACE_ROI_MIN_AREA_RATIO = 0.005
OPENFACE_CONTAINER_HAAR_CASCADE = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
FaceCsvGenerator = Callable[[Path, Path], None]
VideoClipExtractor = Callable[[Path, float, float, Path], None]
FaceRoiClipExtractor = Callable[..., None]
OpenFaceRunner = Callable[[Path, Path, Path], None]


def extract_face_real_embeddings(
    windows: list[dict[str, Any]],
    *,
    cache_root: Path | str,
    output_npz: Path | str,
    failures_out: Path | str,
    encoder_profile: str,
    openface_executable: Path | str | None = None,
    csv_generator: FaceCsvGenerator | None = None,
    clip_extractor: VideoClipExtractor | None = None,
    face_roi_crop_scale: float | None = None,
    face_roi_clip_extractor: FaceRoiClipExtractor | None = None,
    openface_runner: OpenFaceRunner | None = None,
    allow_opencv_fallback: bool = False,
    min_success_rate: float = DEFAULT_MIN_SUCCESS_RATE,
    projection_seed: int = 14014,
) -> dict[str, Any]:
    failures: list[EmbeddingFailure] = []
    samples: list[dict[str, Any]] = []

    for window in windows:
        cache = _read_face_cache(window, cache_root=cache_root, encoder_profile=encoder_profile)
        if cache is None:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="read_face_cache",
                    error_type="source_missing",
                    error="face cache metadata or source video is missing",
                    source_path=str(_face_cache_dir(window, cache_root, encoder_profile)),
                )
            )
            continue

        source_path = Path(str(cache["source_path"]))
        csv_path = Path(str(cache["target_csv_path"]))
        openface_fallback_quality: dict[str, Any] | None = None
        face_roi_quality: dict[str, Any] | None = None
        if not csv_path.is_file():
            explicit_openface = openface_executable is not None
            executable = _resolve_openface_executable(openface_executable)
            if executable is None:
                if explicit_openface:
                    failures.append(
                        _failure(
                            window,
                            encoder_profile,
                            stage="run_openface",
                            error_type="dependency_missing",
                            error="OpenFace executable was provided but was not found",
                            source_path=str(openface_executable),
                        )
                    )
                    continue
                generator = csv_generator or (_generate_opencv_face_csv if allow_opencv_fallback else None)
                if generator is None:
                    failures.append(
                        _failure(
                            window,
                            encoder_profile,
                            stage="run_openface",
                            error_type="dependency_missing",
                            error="OpenFace FeatureExtraction executable was not found",
                            source_path=str(source_path),
                        )
                    )
                    continue
                try:
                    if generator is csv_generator:
                        generator(source_path, csv_path)
                    else:
                        start_seconds, end_seconds = _clip_bounds(window, cache)
                        _generate_opencv_face_csv(
                            source_path,
                            csv_path,
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                        )
                except Exception as exc:
                    failures.append(
                        _failure(
                            window,
                            encoder_profile,
                            stage="run_face_fallback",
                            error_type="extraction_failed",
                            error=str(exc),
                            source_path=str(source_path),
                        )
                    )
                    continue
            else:
                try:
                    clip_path = csv_path.parent / "window.mp4"
                    _ensure_window_clip(
                        source_path,
                        cache,
                        clip_path,
                        window=window,
                        clip_extractor=clip_extractor,
                        face_roi_crop_scale=face_roi_crop_scale,
                        face_roi_clip_extractor=face_roi_clip_extractor,
                    )
                    face_roi_quality = _read_face_roi_metadata(clip_path)
                    runner = openface_runner or _run_openface
                    runner(executable, clip_path, csv_path)
                except Exception as exc:
                    generator = csv_generator or (_generate_opencv_face_csv if allow_opencv_fallback else None)
                    if generator is None:
                        failures.append(
                            _failure(
                                window,
                                encoder_profile,
                                stage="run_openface",
                                error_type="extraction_failed",
                                error=str(exc),
                                source_path=str(source_path),
                            )
                        )
                        continue
                    try:
                        if generator is csv_generator:
                            generator(clip_path, csv_path)
                        else:
                            _generate_opencv_face_csv(clip_path, csv_path)
                        openface_fallback_quality = {
                            "openface_fallback_used": True,
                            "openface_fallback_stage": "run_openface",
                            "openface_fallback_reason": str(exc),
                        }
                    except Exception as fallback_exc:
                        failures.append(
                            _failure(
                                window,
                                encoder_profile,
                                stage="run_openface_fallback",
                                error_type="extraction_failed",
                                error=f"{exc}; fallback failed: {fallback_exc}",
                                source_path=str(source_path),
                            )
                        )
                        continue

        try:
            rows = _read_openface_csv(csv_path)
            quality = _face_quality(rows, csv_path=csv_path, source_path=source_path)
            if face_roi_quality is not None:
                quality.update(face_roi_quality)
            if openface_fallback_quality is not None:
                quality.update(openface_fallback_quality)
            features = _openface_stats(rows)
            masked = quality["face_detection_success_rate"] < float(min_success_rate)
            if masked:
                embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)
                failures.append(
                    _failure(
                        window,
                        encoder_profile,
                        stage="quality_gate",
                        error_type="quality_threshold_failed",
                        error=(
                            "face_detection_success_rate "
                            f"{quality['face_detection_success_rate']:.3f} < {min_success_rate:.3f}"
                        ),
                        source_path=str(csv_path),
                    )
                )
            else:
                embedding = _project_to_256(features, seed=projection_seed, salt=encoder_profile)
                embedding = validate_embedding_shape("face_emb", embedding)
            quality["masked"] = masked
        except ValueError as exc:
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_face",
                    error_type="shape_mismatch",
                    error=str(exc),
                    source_path=str(csv_path),
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive runtime logging
            failures.append(
                _failure(
                    window,
                    encoder_profile,
                    stage="encode_face",
                    error_type="decode_failed",
                    error=str(exc),
                    source_path=str(csv_path),
                )
            )
            continue

        samples.append(
            {
                "sample_id": window.get("sample_id", cache.get("sample_id", "")),
                "event_id": window.get("event_id", cache.get("event_id", "")),
                "subject_id": window.get("subject_id", cache.get("subject_id", "")),
                "face_emb": embedding,
                "modality_mask": np.array([0, 0, 0 if masked else 1, 0], dtype=np.int8),
                "quality_flags": quality,
                "encoder_version": encoder_profile,
            }
        )

    _write_face_npz(samples, output_npz)
    write_failure_list(failures, failures_out)
    return _summary(samples, failures, encoder_profile)


def write_face_quality_summary(summary: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_face_preprocessing_decision(summary: dict[str, Any], path: Path | str) -> Path:
    success_rate = summary.get("mean_face_detection_success_rate")
    low_conf = summary.get("mean_low_confidence_ratio")
    high_risk = max(
        float(summary.get("mean_pose_bad_ratio") or 0.0),
        float(summary.get("mean_dark_frame_ratio") or 0.0),
        float(summary.get("mean_blur_frame_ratio") or 0.0),
        float(summary.get("mean_multi_face_ratio") or 0.0),
    )
    raw_quality_gate_passed = bool(
        success_rate is not None
        and float(success_rate) >= 0.80
        and low_conf is not None
        and float(low_conf) <= 0.20
        and high_risk <= 0.30
    )
    payload = {
        "stage": 14,
        "raw_profile": summary.get("encoder_profile"),
        "enable_preprocessing": False,
        "default_branch": "face_raw_openface_stats_v1",
        "raw_quality_gate_passed": raw_quality_gate_passed,
        "decision_basis": "single-modality quality audit only; downstream 5-seed/bootstrap gate not run yet",
        "triggered_conditions": [] if raw_quality_gate_passed else ["raw_quality_gate_incomplete_or_failed"],
        "quality_summary": summary,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _read_face_cache(
    window: dict[str, Any],
    *,
    cache_root: Path | str,
    encoder_profile: str,
) -> dict[str, Any] | None:
    cache_dir = _face_cache_dir(window, cache_root, encoder_profile)
    metadata_path = cache_dir / "openface_target.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = Path(str(metadata.get("source_path") or ""))
    if not source_path.is_file():
        return None
    target_csv_path = Path(str(metadata.get("target_csv_path") or cache_dir / "openface.csv"))
    metadata["source_path"] = str(source_path)
    metadata["target_csv_path"] = str(target_csv_path)
    return metadata


def _face_cache_dir(window: dict[str, Any], cache_root: Path | str, encoder_profile: str) -> Path:
    return Path(cache_root) / "openface" / str(window.get("sample_id", "")) / encoder_profile


def _clip_bounds(window: dict[str, Any], cache: dict[str, Any]) -> tuple[float | None, float | None]:
    window_start_text = window.get("window_start_time")
    window_end_text = window.get("window_end_time")
    source_path = str(cache.get("source_path") or "")
    if window_start_text and window_end_text and source_path:
        for candidate in window.get("video_candidates", []) or []:
            if str(candidate.get("mp4_path") or "") != source_path:
                continue
            mp4_start_text = candidate.get("mp4_start_time")
            if not mp4_start_text:
                continue
            try:
                mp4_start = parse_absolute_time(str(mp4_start_text))
                window_start = parse_absolute_time(str(window_start_text))
                window_end = parse_absolute_time(str(window_end_text))
            except ValueError:
                continue
            start = max(0.0, float((window_start - mp4_start).total_seconds()))
            end = max(start, float((window_end - mp4_start).total_seconds()))
            if candidate.get("duration_seconds") is not None:
                duration = float(candidate["duration_seconds"])
                start = min(start, duration)
                end = min(end, duration)
            if end > start:
                return start, end

    if cache.get("clip_start_seconds") is not None and cache.get("clip_end_seconds") is not None:
        return float(cache["clip_start_seconds"]), float(cache["clip_end_seconds"])
    for candidate in window.get("video_candidates", []) or []:
        if candidate.get("clip_start_seconds") is not None and candidate.get("clip_end_seconds") is not None:
            return float(candidate["clip_start_seconds"]), float(candidate["clip_end_seconds"])
        if candidate.get("window_start_seconds") is not None and candidate.get("window_end_seconds") is not None:
            return float(candidate["window_start_seconds"]), float(candidate["window_end_seconds"])
    return None, None


def _ensure_window_clip(
    source_path: Path,
    cache: dict[str, Any],
    clip_path: Path,
    *,
    window: dict[str, Any],
    clip_extractor: VideoClipExtractor | None = None,
    face_roi_crop_scale: float | None = None,
    face_roi_clip_extractor: FaceRoiClipExtractor | None = None,
) -> Path:
    if clip_path.is_file() and clip_path.stat().st_size > 0:
        return clip_path
    start_seconds, end_seconds = _clip_bounds(window, cache)
    if start_seconds is None or end_seconds is None:
        raise RuntimeError("missing clip_start_seconds/clip_end_seconds for OpenFace window clip")
    if end_seconds <= start_seconds:
        raise RuntimeError(f"invalid clip bounds: {start_seconds} >= {end_seconds}")
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    if clip_extractor is not None:
        clip_extractor(source_path, float(start_seconds), float(end_seconds), clip_path)
    elif face_roi_crop_scale is not None and float(face_roi_crop_scale) > 0:
        extractor = face_roi_clip_extractor or _extract_face_roi_clip_cv2
        extractor(
            source_path,
            float(start_seconds),
            float(end_seconds),
            clip_path,
            roi_scale=float(face_roi_crop_scale),
        )
    else:
        _extract_video_clip_ffmpeg(source_path, float(start_seconds), float(end_seconds), clip_path)
    if not clip_path.is_file() or clip_path.stat().st_size == 0:
        raise RuntimeError("window clip extractor did not create a non-empty MP4")
    return clip_path


def _extract_video_clip_ffmpeg(
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_clip: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg executable was not found on PATH")
    duration = max(0.001, float(end_seconds) - float(start_seconds))
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{float(start_seconds):.6f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-vf",
        f"fps={OPENFACE_CLIP_FPS:g},scale={OPENFACE_CLIP_WIDTH}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        str(output_clip),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "ffmpeg clip extraction failed"
        raise RuntimeError(message)


def _extract_face_roi_clip_cv2(
    source_path: Path,
    start_seconds: float,
    end_seconds: float,
    output_clip: Path,
    *,
    roi_scale: float = FACE_ROI_CROP_SCALE,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise RuntimeError(f"missing OpenCV dependency: {exc.name}") from exc

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source_path}")

    frames: list[np.ndarray] = []
    detections: list[tuple[int, int, int, int] | None] = []
    try:
        duration = max(0.001, float(end_seconds) - float(start_seconds))
        sample_count = max(1, int(math.ceil(duration * OPENFACE_CLIP_FPS)))
        for idx in range(sample_count):
            timestamp = float(start_seconds) + float(idx) / OPENFACE_CLIP_FPS
            if timestamp >= float(end_seconds):
                break
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue
            frames.append(frame)
            detections.append(_detect_main_face_bbox_cv2(frame))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"video produced no readable frames: {source_path}")

    frame_height, frame_width = frames[0].shape[:2]
    detected_roi_frame_count = sum(1 for detection in detections if detection is not None)
    rois = _face_roi_sequence(
        detections,
        frame_width=frame_width,
        frame_height=frame_height,
        scale=roi_scale,
    )
    output_clip.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_clip),
        fourcc,
        float(OPENFACE_CLIP_FPS),
        (OPENFACE_CLIP_WIDTH, OPENFACE_CLIP_WIDTH),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open ROI video writer: {output_clip}")
    try:
        for frame, (x, y, w, h) in zip(frames, rois):
            crop = frame[int(y) : int(y + h), int(x) : int(x + w)]
            if crop.size == 0:
                crop = frame
            resized = cv2.resize(crop, (OPENFACE_CLIP_WIDTH, OPENFACE_CLIP_WIDTH))
            writer.write(resized)
    finally:
        writer.release()
    metadata = {
        "face_roi_crop_enabled": True,
        "face_roi_crop_scale": float(roi_scale),
        "face_roi_output_size": OPENFACE_CLIP_WIDTH,
        "face_roi_frame_count": len(frames),
        "face_roi_detected_frame_count": detected_roi_frame_count,
        "face_roi_filled_missing_frame_count": len(frames) - detected_roi_frame_count,
        "face_roi_full_frame_fallback": detected_roi_frame_count == 0,
    }
    _face_roi_metadata_path(output_clip).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _detect_main_face_bbox_cv2(frame: np.ndarray) -> tuple[int, int, int, int] | None:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates: list[tuple[int, int, int, int]] = []
    for cascade_name in (
        "haarcascade_frontalface_default.xml",
        "haarcascade_frontalface_alt.xml",
    ):
        detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / cascade_name))
        if detector.empty():
            continue
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        candidates.extend(tuple(int(value) for value in face) for face in faces)
    if not candidates:
        return None
    bbox = max(candidates, key=lambda item: item[2] * item[3])
    frame_height, frame_width = frame.shape[:2]
    area_ratio = float((bbox[2] * bbox[3]) / max(1, frame_width * frame_height))
    if area_ratio < FACE_ROI_MIN_AREA_RATIO:
        return None
    return bbox


def _face_roi_metadata_path(clip_path: Path) -> Path:
    return clip_path.with_name(f"{clip_path.stem}_face_roi.json")


def _read_face_roi_metadata(clip_path: Path) -> dict[str, Any] | None:
    path = _face_roi_metadata_path(clip_path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _expanded_face_roi(
    face_bbox: tuple[int, int, int, int],
    *,
    frame_width: int,
    frame_height: int,
    scale: float = FACE_ROI_CROP_SCALE,
) -> tuple[int, int, int, int]:
    x, y, w, h = [int(value) for value in face_bbox]
    side = int(round(max(w, h) * float(scale)))
    side = max(1, min(side, int(frame_width), int(frame_height)))
    center_x = float(x) + float(w) / 2.0
    center_y = float(y) + float(h) / 2.0
    left = int(round(center_x - float(side) / 2.0))
    top = int(round(center_y - float(side) / 2.0))
    left = max(0, min(left, int(frame_width) - side))
    top = max(0, min(top, int(frame_height) - side))
    return left, top, side, side


def _face_roi_sequence(
    detections: list[tuple[int, int, int, int] | None],
    *,
    frame_width: int,
    frame_height: int,
    scale: float = FACE_ROI_CROP_SCALE,
) -> list[tuple[int, int, int, int]]:
    full_frame = (0, 0, int(frame_width), int(frame_height))
    expanded = [
        _expanded_face_roi(
            detection,
            frame_width=frame_width,
            frame_height=frame_height,
            scale=scale,
        )
        if detection is not None
        else None
        for detection in detections
    ]
    if not any(roi is not None for roi in expanded):
        return [full_frame for _ in detections]

    first_roi = next(roi for roi in expanded if roi is not None)
    rois: list[tuple[int, int, int, int]] = []
    previous = first_roi
    for roi in expanded:
        if roi is not None:
            previous = roi
        rois.append(previous)
    return rois


def _resolve_openface_executable(openface_executable: Path | str | None) -> Path | None:
    candidates = []
    if openface_executable:
        candidates.append(str(openface_executable))
    if os.environ.get("OPENFACE_EXECUTABLE"):
        candidates.append(os.environ["OPENFACE_EXECUTABLE"])
    candidates.extend(["FeatureExtraction", "OpenFaceOffline"])
    for candidate in candidates:
        found = shutil.which(candidate) if not Path(candidate).is_file() else candidate
        if found:
            return Path(found)
    return None


def _run_openface(executable: Path, source_path: Path, csv_path: Path) -> None:
    source_path = source_path.resolve()
    csv_path = csv_path.resolve()
    out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    command = _openface_command(executable, source_path, csv_path)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 and not _openface_csv_available(csv_path):
        _adopt_generated_openface_csv(source_path, csv_path)
    if completed.returncode != 0 and not _openface_csv_available(csv_path):
        message = completed.stderr.strip() or completed.stdout.strip() or "OpenFace failed"
        raise RuntimeError(message)
    if not _openface_csv_available(csv_path):
        _adopt_generated_openface_csv(source_path, csv_path)
    if not _openface_csv_available(csv_path):
        raise RuntimeError("OpenFace did not create expected CSV")


def _openface_command(executable: Path, source_path: Path, csv_path: Path) -> list[str]:
    command = [
        str(executable),
        "-f",
        str(source_path),
        "-out_dir",
        str(csv_path.parent),
        "-of",
        csv_path.stem,
    ]
    haar = _openface_haar_cascade(executable)
    if haar:
        command.extend(["-fdloc", haar])
    return command


def _openface_haar_cascade(executable: Path) -> str | None:
    env_path = os.environ.get("OPENFACE_HAAR_CASCADE")
    if env_path:
        return env_path
    if executable.name.lower() == "feature_extraction.sh":
        return OPENFACE_CONTAINER_HAAR_CASCADE
    return None


def _adopt_generated_openface_csv(source_path: Path, csv_path: Path) -> None:
    generated = csv_path.parent / f"{source_path.stem}.csv"
    if generated.is_file() and generated.stat().st_size > 0:
        generated.replace(csv_path)


def _openface_csv_available(csv_path: Path) -> bool:
    return csv_path.is_file() and csv_path.stat().st_size > 0


def _generate_opencv_face_csv(
    source_path: Path,
    csv_path: Path,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise RuntimeError(f"missing OpenCV dependency: {exc.name}") from exc

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"OpenCV Haar cascade not found: {cascade_path}")
    rows = [
        "frame,timestamp,confidence,success,face_count,face_area_ratio,gray_mean,laplacian_var,pose_Rx,pose_Ry,pose_Rz,gaze_0_x,gaze_0_y"
    ]
    frames = _sample_video_frames(source_path, start_seconds=start_seconds, end_seconds=end_seconds)
    for frame_index, timestamp, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        face_count = int(len(faces))
        success = 1 if face_count > 0 else 0
        if face_count:
            x, y, w, h = sorted(faces, key=lambda item: int(item[2] * item[3]), reverse=True)[0]
            crop = gray[y : y + h, x : x + w]
            area_ratio = float((w * h) / max(1, gray.shape[0] * gray.shape[1]))
            brightness = float(np.mean(crop) / 255.0) if crop.size else 0.0
            blur = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
            confidence = min(1.0, 0.5 + area_ratio * 10.0)
        else:
            area_ratio = 0.0
            brightness = float(np.mean(gray) / 255.0)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            confidence = 0.0
        rows.append(
            ",".join(
                [
                    str(frame_index),
                    f"{timestamp:.6f}",
                    f"{confidence:.6f}",
                    str(success),
                    str(face_count),
                    f"{area_ratio:.8f}",
                    f"{brightness:.8f}",
                    f"{blur:.8f}",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                ]
            )
        )
    if len(rows) == 1:
        raise RuntimeError(f"video produced no readable frames: {source_path}")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _sample_video_frames(
    source_path: Path,
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    fps: float = 0.5,
    max_frames: int = 5,
) -> list[tuple[int, float, np.ndarray]]:
    frames = _sample_video_frames_ffmpeg(
        source_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        fps=fps,
        max_frames=max_frames,
    )
    if frames:
        return frames
    return _sample_video_frames_cv2(
        source_path,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        fps=fps,
        max_frames=max_frames,
    )


def _sample_video_frames_ffmpeg(
    source_path: Path,
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    fps: float,
    max_frames: int,
) -> list[tuple[int, float, np.ndarray]]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    duration = None
    if start_seconds is not None and end_seconds is not None:
        duration = max(0.01, float(end_seconds) - float(start_seconds))
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_seconds is not None:
        command.extend(["-ss", f"{float(start_seconds):.6f}"])
    command.extend(["-i", str(source_path)])
    if duration is not None:
        command.extend(["-t", f"{duration:.6f}"])
    command.extend(["-vf", f"fps={fps}", "-frames:v", str(max_frames), "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0 or not completed.stdout:
        return []
    try:
        import cv2
    except ImportError:
        return []
    data = completed.stdout
    frames: list[tuple[int, float, np.ndarray]] = []
    start_marker = b"\xff\xd8"
    end_marker = b"\xff\xd9"
    cursor = 0
    while len(frames) < max_frames:
        start = data.find(start_marker, cursor)
        if start < 0:
            break
        end = data.find(end_marker, start + 2)
        if end < 0:
            break
        chunk = data[start : end + 2]
        cursor = end + 2
        image = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame_index = len(frames) + 1
        timestamp = (float(start_seconds) if start_seconds is not None else 0.0) + (frame_index - 1) / fps
        frames.append((frame_index, timestamp, image))
    return frames


def _sample_video_frames_cv2(
    source_path: Path,
    *,
    start_seconds: float | None,
    end_seconds: float | None,
    fps: float,
    max_frames: int,
) -> list[tuple[int, float, np.ndarray]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on server runtime
        raise RuntimeError(f"missing OpenCV dependency: {exc.name}") from exc
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if start_seconds is not None and start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(start_seconds) * 1000.0)
    sample_stride = max(1, int(round(source_fps / fps))) if source_fps > 0 else 1
    frames: list[tuple[int, float, np.ndarray]] = []
    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    while len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1
        timestamp = (frame_index - 1) / source_fps if source_fps > 0 else float(frame_index - 1)
        if end_seconds is not None and timestamp >= float(end_seconds):
            break
        if (frame_index - 1) % sample_stride == 0:
            frames.append((len(frames) + 1, timestamp, frame))
    capture.release()
    return frames


def _read_openface_csv(csv_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                normalized = (key or "").strip()
                if not normalized:
                    continue
                try:
                    parsed[normalized] = float(str(value).strip())
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    if not rows:
        raise ValueError("OpenFace CSV has no numeric rows")
    return rows


def _face_quality(rows: list[dict[str, float]], *, csv_path: Path, source_path: Path) -> dict[str, Any]:
    frame_count = len(rows)
    success_values = np.array([row.get("success", 0.0) for row in rows], dtype=np.float32)
    confidence = np.array([row.get("confidence", 0.0) for row in rows], dtype=np.float32)
    yaw = _first_available(rows, ("pose_Ry", "pose_Y"))
    pitch = _first_available(rows, ("pose_Rx", "pose_P"))
    roll = _first_available(rows, ("pose_Rz", "pose_R"))
    pose_bad = (
        (np.abs(yaw) > math.radians(35))
        | (np.abs(pitch) > math.radians(25))
        | (np.abs(roll) > math.radians(25))
    )
    return {
        "csv_path": str(csv_path),
        "source_path": str(source_path),
        "frame_count": frame_count,
        "face_detection_success_rate": float(np.mean(success_values > 0.5)),
        "mean_openface_confidence": float(np.mean(confidence)),
        "low_confidence_ratio": float(np.mean(confidence < LOW_CONFIDENCE_THRESHOLD)),
        "pose_bad_ratio": float(np.mean(pose_bad)),
        "dark_frame_ratio": None,
        "blur_frame_ratio": None,
        "multi_face_ratio": 0.0,
        "main_face_ambiguity_ratio": 0.0,
    }


def _openface_stats(rows: list[dict[str, float]]) -> np.ndarray:
    feature_names = [
        name
        for name in sorted(rows[0])
        if name not in {"frame", "timestamp", "success"}
        and (
            name.startswith("AU")
            or name.startswith("gaze")
            or name.startswith("pose")
            or name == "confidence"
        )
    ]
    if not feature_names:
        raise ValueError("OpenFace CSV does not contain AU/gaze/pose/confidence columns")
    features: list[float] = []
    for name in feature_names:
        values = np.array([row.get(name, math.nan) for row in rows], dtype=np.float32)
        values = values[np.isfinite(values)]
        if values.size == 0:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend(
                [
                    float(values.mean()),
                    float(values.std()),
                    float(values.min()),
                    float(values.max()),
                ]
            )
    return np.asarray(features, dtype=np.float32)


def _first_available(rows: list[dict[str, float]], names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if any(name in row for row in rows):
            return np.array([row.get(name, 0.0) for row in rows], dtype=np.float32)
    return np.zeros(len(rows), dtype=np.float32)


def _project_to_256(vector: np.ndarray, *, seed: int, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("OpenFace feature vector is empty")
    if not np.isfinite(values).all():
        raise ValueError("OpenFace feature vector contains NaN or infinite values")
    normalized = values.copy()
    std = float(normalized.std())
    if std > 0:
        normalized = (normalized - float(normalized.mean())) / std
    rng_seed = seed + int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(rng_seed)
    scale = 1.0 / max(1.0, float(np.sqrt(normalized.size)))
    weights = rng.normal(0.0, scale, size=(normalized.size, EMBEDDING_DIM)).astype(np.float32)
    return np.tanh(normalized @ weights).astype(np.float32)


def _write_face_npz(samples: list[dict[str, Any]], output_npz: Path | str) -> Path:
    out = Path(output_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([sample["sample_id"] for sample in samples], dtype=object),
        event_id=np.array([sample["event_id"] for sample in samples], dtype=object),
        subject_id=np.array([sample["subject_id"] for sample in samples], dtype=object),
        face_emb=np.stack([sample["face_emb"] for sample in samples]).astype(np.float32)
        if samples
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        modality_mask=np.stack([sample["modality_mask"] for sample in samples]).astype(np.int8)
        if samples
        else np.zeros((0, 4), dtype=np.int8),
        quality_flags=np.array(
            [json.dumps(sample["quality_flags"], ensure_ascii=False) for sample in samples],
            dtype=object,
        ),
        encoder_version=np.array([sample["encoder_version"] for sample in samples], dtype=object),
    )
    return out


def _summary(
    samples: list[dict[str, Any]],
    failures: list[EmbeddingFailure],
    encoder_profile: str,
) -> dict[str, Any]:
    usable = [sample for sample in samples if int(sample["modality_mask"][2]) == 1]
    qualities = [sample["quality_flags"] for sample in samples]
    return {
        "stage": 14,
        "modality": "face",
        "encoder_profile": encoder_profile,
        "embedded_count": len(samples),
        "success_count": len(usable),
        "failure_count": len(failures),
        "failure_types": _count_by_error_type(failures),
        "masked_count": len(samples) - len(usable),
        "mean_face_detection_success_rate": _mean_quality(qualities, "face_detection_success_rate"),
        "mean_openface_confidence": _mean_quality(qualities, "mean_openface_confidence"),
        "mean_low_confidence_ratio": _mean_quality(qualities, "low_confidence_ratio"),
        "mean_pose_bad_ratio": _mean_quality(qualities, "pose_bad_ratio"),
        "mean_dark_frame_ratio": _mean_quality(qualities, "dark_frame_ratio"),
        "mean_blur_frame_ratio": _mean_quality(qualities, "blur_frame_ratio"),
        "mean_multi_face_ratio": _mean_quality(qualities, "multi_face_ratio"),
        "nan_count": int(sum(np.isnan(sample["face_emb"]).sum() for sample in samples)),
    }


def _mean_quality(qualities: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in qualities if item.get(key) is not None]
    return None if not values else float(np.mean(values))


def _count_by_error_type(failures: list[EmbeddingFailure]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for failure in failures:
        counts[failure.error_type] = counts.get(failure.error_type, 0) + 1
    return counts


def _failure(
    window: dict[str, Any],
    encoder_profile: str,
    *,
    stage: str,
    error_type: str,
    error: str,
    source_path: str,
) -> EmbeddingFailure:
    return EmbeddingFailure(
        sample_id=str(window.get("sample_id") or "<missing-sample-id>"),
        event_id=str(window.get("event_id") or "<missing-event-id>"),
        subject_id=str(window.get("subject_id") or "<missing-subject-id>"),
        modality="face",
        encoder_profile=encoder_profile,
        stage=stage,
        error_type=error_type,
        error=error,
        source_path=source_path,
        recoverable=True,
    )
