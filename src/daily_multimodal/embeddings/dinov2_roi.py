from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape


FACE_MASK_INDEX = 2
DINO_ENCODER_VERSION = "dinov2_base_roi_v4"
FrameEncoder = Callable[..., dict[str, Any]]


def build_dinov2_roi_embeddings(
    *,
    window_index_path: Path | str,
    openface_cache_root: Path | str,
    openface_encoder_profile: str,
    out_path: Path | str,
    fps: float = 2.0,
    max_frames_per_window: int | None = 20,
    batch_size: int = 16,
    model_name: str = "facebook/dinov2-base",
    device: str | None = None,
    progress_out: Path | str | None = None,
    failures_out: Path | str | None = None,
    progress_every: int = 10,
    frame_encoder: FrameEncoder | None = None,
) -> dict[str, Any]:
    windows = _read_jsonl(Path(window_index_path))
    cache_root = Path(openface_cache_root)
    progress_path = Path(progress_out) if progress_out is not None else None
    failures_path = Path(failures_out) if failures_out is not None else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text("", encoding="utf-8")
    if failures_path is not None:
        failures_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.write_text("[]", encoding="utf-8")
    started = time.monotonic()
    total = len(windows)
    _write_progress(
        progress_path,
        "model_init",
        index=0,
        total=total,
        sample_id="-",
        written=0,
        failures=0,
        elapsed=0.0,
        force=True,
    )
    encoder = frame_encoder or Dinov2FrameEncoder(model_name=model_name, device=device)
    rows = []
    failures: list[dict[str, Any]] = []
    for index, window in enumerate(windows, start=1):
        sample_id = str(window.get("sample_id", ""))
        _write_progress(
            progress_path,
            "start",
            index=index,
            total=total,
            sample_id=sample_id,
            written=len(rows),
            failures=len(failures),
            elapsed=time.monotonic() - started,
        )
        roi_video = _roi_video_path(cache_root, sample_id, openface_encoder_profile)
        if not roi_video.is_file():
            rows.append(_masked_row(window, roi_video, missing_roi_video=True))
            failures.append({"sample_id": sample_id, "error_type": "missing_roi_video"})
            _write_failures(failures_path, failures)
            _write_progress(
                progress_path,
                "done",
                index=index,
                total=total,
                sample_id=sample_id,
                written=len(rows),
                failures=len(failures),
                elapsed=time.monotonic() - started,
                status="missing_roi_video",
                force=True,
                progress_every=progress_every,
            )
            continue
        try:
            encoded = encoder(
                roi_video,
                fps=float(fps),
                max_frames=max_frames_per_window,
                batch_size=int(batch_size),
                device=device,
            )
            raw_embedding = np.asarray(encoded["embedding"], dtype=np.float32)
            embedding = _project_to_256(raw_embedding, salt=DINO_ENCODER_VERSION)
            rows.append(_embedding_row(window, roi_video, embedding, encoded))
            _write_progress(
                progress_path,
                "done",
                index=index,
                total=total,
                sample_id=sample_id,
                written=len(rows),
                failures=len(failures),
                elapsed=time.monotonic() - started,
                status="success",
                progress_every=progress_every,
            )
        except Exception as exc:
            rows.append(_masked_row(window, roi_video, error=str(exc)))
            failures.append({"sample_id": sample_id, "error_type": type(exc).__name__, "error": str(exc)})
            _write_failures(failures_path, failures)
            _write_progress(
                progress_path,
                "done",
                index=index,
                total=total,
                sample_id=sample_id,
                written=len(rows),
                failures=len(failures),
                elapsed=time.monotonic() - started,
                status=type(exc).__name__,
                force=True,
                progress_every=progress_every,
            )
    _write_npz(rows, out_path)
    _write_failures(failures_path, failures)
    _write_progress(
        progress_path,
        "finish",
        index=total,
        total=total,
        sample_id="-",
        written=len(rows),
        failures=len(failures),
        elapsed=time.monotonic() - started,
        force=True,
        progress_every=progress_every,
    )
    return {
        "variant": DINO_ENCODER_VERSION,
        "row_count": len(rows),
        "mask_sum": int(sum(int(row["modality_mask"][FACE_MASK_INDEX]) for row in rows)),
        "failure_count": int(len(failures)),
        "out_path": str(out_path),
    }


class Dinov2FrameEncoder:
    def __init__(self, *, model_name: str = "facebook/dinov2-base", device: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise RuntimeError("DINOv2 ROI extraction requires torch and transformers") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(
        self,
        path: Path,
        *,
        fps: float,
        max_frames: int | None,
        batch_size: int,
        device: str | None = None,
    ) -> dict[str, Any]:
        frames, source_fps = _sample_video_frames(path, fps=fps, max_frames=max_frames)
        if not frames:
            raise ValueError(f"no frames sampled from ROI video: {path}")
        features = []
        with self.torch.inference_mode():
            for start in range(0, len(frames), max(1, int(batch_size))):
                batch = frames[start : start + max(1, int(batch_size))]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs = self.model(**inputs)
                cls = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
                features.append(cls.astype(np.float32, copy=False))
        stacked = np.concatenate(features, axis=0)
        return {
            "embedding": stacked.mean(axis=0).astype(np.float32),
            "sampled_frame_count": int(len(frames)),
            "usable_frame_count": int(len(frames)),
            "source_fps": None if source_fps is None else float(source_fps),
        }


def _sample_video_frames(path: Path, *, fps: float, max_frames: int | None) -> tuple[list[np.ndarray], float | None]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("DINOv2 ROI extraction requires opencv-python or opencv-python-headless") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open ROI video: {path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    step = 1 if source_fps <= 0 or fps <= 0 else max(1, int(round(source_fps / fps)))
    frames: list[np.ndarray] = []
    frame_index = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                if max_frames is not None and len(frames) >= int(max_frames):
                    break
            frame_index += 1
    finally:
        cap.release()
    return frames, source_fps if source_fps > 0 else None


def _roi_video_path(cache_root: Path, sample_id: str, profile: str) -> Path:
    return cache_root / "openface" / sample_id / profile / "window.mp4"


def _embedding_row(
    window: dict[str, Any],
    roi_video: Path,
    embedding: np.ndarray,
    encoded: dict[str, Any],
) -> dict[str, Any]:
    mask = np.zeros(4, dtype=np.int8)
    mask[FACE_MASK_INDEX] = 1
    quality = _base_quality(roi_video)
    quality.update(
        {
            "sampled_frame_count": int(encoded.get("sampled_frame_count", 0) or 0),
            "usable_frame_count": int(encoded.get("usable_frame_count", 0) or 0),
            "source_fps": _json_ready(encoded.get("source_fps")),
            "projected_from_dim": int(np.asarray(encoded["embedding"]).reshape(-1).shape[0]),
        }
    )
    return _row(window, embedding=embedding, mask=mask, quality_flags=quality)


def _masked_row(
    window: dict[str, Any],
    roi_video: Path,
    *,
    missing_roi_video: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    quality = _base_quality(roi_video)
    quality["missing_roi_video"] = bool(missing_roi_video)
    if error:
        quality["error"] = error
    return _row(
        window,
        embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        mask=np.zeros(4, dtype=np.int8),
        quality_flags=quality,
    )


def _row(
    window: dict[str, Any],
    *,
    embedding: np.ndarray,
    mask: np.ndarray,
    quality_flags: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "labels": _window_labels(window),
        "face_emb": validate_embedding_shape("face_emb", np.asarray(embedding, dtype=np.float32)),
        "modality_mask": mask.astype(np.int8),
        "quality_flags": quality_flags,
        "encoder_version": DINO_ENCODER_VERSION,
    }


def _base_quality(roi_video: Path) -> dict[str, Any]:
    return {
        "variant": DINO_ENCODER_VERSION,
        "dinov2_model": "facebook/dinov2-base",
        "frozen": True,
        "input": "openface_mainface_2x_roi_window_mp4",
        "roi_crop_scale": 2.0,
        "roi_video_path": str(roi_video),
    }


def _project_to_256(vector: np.ndarray, *, salt: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    normalized = values / norm if norm > 0 else values
    if normalized.size == EMBEDDING_DIM:
        return validate_embedding_shape("face_emb", normalized.astype(np.float32))
    digest = hashlib.sha256(f"{salt}:{normalized.size}".encode("utf-8")).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    weights = rng.normal(
        0.0,
        1.0 / math.sqrt(float(max(1, normalized.size))),
        size=(normalized.size, EMBEDDING_DIM),
    ).astype(np.float32)
    projected = normalized @ weights
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm > 0:
        projected = projected / projected_norm
    return validate_embedding_shape("face_emb", projected.astype(np.float32))


def _write_npz(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_id=np.array([row["sample_id"] for row in rows], dtype=object),
        event_id=np.array([row["event_id"] for row in rows], dtype=object),
        subject_id=np.array([row["subject_id"] for row in rows], dtype=object),
        labels=np.array([json.dumps(_json_ready(row["labels"]), ensure_ascii=False, allow_nan=False) for row in rows], dtype=object),
        face_emb=np.stack([row["face_emb"] for row in rows]).astype(np.float32)
        if rows
        else np.zeros((0, EMBEDDING_DIM), dtype=np.float32),
        modality_mask=np.stack([row["modality_mask"] for row in rows]).astype(np.int8)
        if rows
        else np.zeros((0, 4), dtype=np.int8),
        quality_flags=np.array(
            [json.dumps(_json_ready(row["quality_flags"]), ensure_ascii=False, allow_nan=False) for row in rows],
            dtype=object,
        ),
        encoder_version=np.array([row["encoder_version"] for row in rows], dtype=object),
    )
    return out


def _write_failures(path: Path | None, failures: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.write_text(json.dumps(_json_ready(failures), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_progress(
    path: Path | None,
    phase: str,
    *,
    index: int,
    total: int,
    sample_id: str,
    written: int,
    failures: int,
    elapsed: float,
    status: str | None = None,
    force: bool = False,
    progress_every: int = 1,
) -> None:
    if path is None:
        return
    if phase == "done" and not force and progress_every > 0 and index % int(progress_every) != 0 and index != total:
        return
    percent = 100.0 if total == 0 else min(100.0, (float(index) / float(total)) * 100.0)
    filled = int(round(percent / 100.0 * 24))
    bar = "#" * filled + "." * (24 - filled)
    suffix = "" if status is None else f" status={status}"
    line = (
        f"{phase} {index}/{total} [{bar}] {percent:5.1f}% "
        f"sample_id={sample_id} written={written} failures={failures} elapsed={elapsed:.1f}s{suffix}\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _window_labels(window: dict[str, Any]) -> dict[str, Any]:
    values = window.get("label_columns") or window.get("labels") or {}
    return _json_ready(values) if isinstance(values, dict) else _parse_json_object(values)


def _parse_json_object(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
