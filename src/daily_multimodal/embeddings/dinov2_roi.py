from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from daily_multimodal.embeddings.contracts import EMBEDDING_DIM, validate_embedding_shape
from daily_multimodal.embeddings.video_regions import (
    REGION_CLIP_FRAME_COUNT,
    _default_source_from_window,
    _prepare_region_frame,
    _upper_body_bbox_from_window,
)


FACE_MASK_INDEX = 2
DINO_ENCODER_VERSION = "video_v4a_dinov2_2xroi_mean_std_max"
DEFAULT_TEMPORAL_POOLING = "mean_std_max"
NO_AUGMENTATION_PROFILE = "none"
V4D_MILD_AUGMENTATION_PROFILE = "v4d_mild_color_crop_scale"
V4D_APPEARANCE_AUGMENTATION_PROFILE = "v4d_appearance_mild"
V4D_A1_COLOR_BRIGHTNESS_PROFILE = "v4d_a1_color_brightness"
V4D_A2_COLOR_BRIGHTNESS_GRAYSCALE_PROFILE = "v4d_a2_color_brightness_grayscale"
V4D_A3_COLOR_BRIGHTNESS_GRAYSCALE_CROP_SCALE_PROFILE = "v4d_a3_color_brightness_grayscale_crop_scale"
V4D_WEAK_COLOR_BRIGHTNESS_CONTRAST_PROFILE = "v4d_weak_color_brightness_contrast"
SUPPORTED_AUGMENTATION_PROFILES = {
    NO_AUGMENTATION_PROFILE,
    V4D_MILD_AUGMENTATION_PROFILE,
    V4D_APPEARANCE_AUGMENTATION_PROFILE,
    V4D_A1_COLOR_BRIGHTNESS_PROFILE,
    V4D_A2_COLOR_BRIGHTNESS_GRAYSCALE_PROFILE,
    V4D_A3_COLOR_BRIGHTNESS_GRAYSCALE_CROP_SCALE_PROFILE,
    V4D_WEAK_COLOR_BRIGHTNESS_CONTRAST_PROFILE,
}
FrameEncoder = Callable[..., dict[str, Any]]


def build_dinov2_roi_embeddings(
    *,
    window_index_path: Path | str,
    openface_cache_root: Path | str | None,
    openface_encoder_profile: str,
    out_path: Path | str,
    region_cache_root: Path | str | None = None,
    video_region: str = "2x_face_roi",
    fallback_video_region: str | None = None,
    direct_video_region_from_window: bool = False,
    frame_sequences_out: Path | str | None = None,
    fps: float = 2.0,
    num_frames: int | None = 16,
    max_frames_per_window: int | None = None,
    temporal_pooling: str = DEFAULT_TEMPORAL_POOLING,
    batch_size: int = 16,
    model_name: str = "facebook/dinov2-base",
    device: str | None = None,
    progress_out: Path | str | None = None,
    failures_out: Path | str | None = None,
    progress_every: int = 10,
    start_index: int = 0,
    max_windows: int | None = None,
    augmentation_profile: str = NO_AUGMENTATION_PROFILE,
    augmentation_views: int = 1,
    frame_encoder: FrameEncoder | None = None,
) -> dict[str, Any]:
    if temporal_pooling != DEFAULT_TEMPORAL_POOLING:
        raise ValueError(f"unsupported temporal_pooling: {temporal_pooling}")
    if video_region not in {"2x_face_roi", "upper_body", "full_frame"}:
        raise ValueError(f"unsupported video_region: {video_region}")
    if fallback_video_region is not None and fallback_video_region not in {"full_frame"}:
        raise ValueError(f"unsupported fallback_video_region: {fallback_video_region}")
    if fallback_video_region is not None and video_region != "upper_body":
        raise ValueError("fallback_video_region is only supported with video_region=upper_body")
    if fallback_video_region is not None and region_cache_root is None:
        raise ValueError("fallback_video_region requires region_cache_root")
    if fallback_video_region is not None and direct_video_region_from_window:
        raise ValueError("fallback_video_region is not supported with direct_video_region_from_window")
    if augmentation_profile not in SUPPORTED_AUGMENTATION_PROFILES:
        raise ValueError(f"unsupported augmentation_profile: {augmentation_profile}")
    if augmentation_profile != NO_AUGMENTATION_PROFILE and int(augmentation_views) < 2:
        raise ValueError("augmentation_views must be at least 2 when augmentation_profile is enabled")
    if direct_video_region_from_window and video_region == "2x_face_roi":
        raise ValueError("direct_video_region_from_window is only supported for upper_body or full_frame")
    if openface_cache_root is None and region_cache_root is None and not direct_video_region_from_window:
        raise ValueError("openface_cache_root or region_cache_root is required")
    if int(start_index) < 0:
        raise ValueError("start_index must be non-negative")
    frame_limit = max_frames_per_window if max_frames_per_window is not None else num_frames
    all_windows = _read_jsonl(Path(window_index_path))
    source_row_count = len(all_windows)
    windows = all_windows[int(start_index) :]
    if max_windows is not None:
        windows = windows[: int(max_windows)]
    cache_root = Path(openface_cache_root) if openface_cache_root is not None else None
    region_root = Path(region_cache_root) if region_cache_root is not None else None
    encoder_version = _encoder_version_for_region(
        video_region if region_root is not None or direct_video_region_from_window else "2x_face_roi",
        augmentation_profile=augmentation_profile,
        fallback_video_region=fallback_video_region,
    )
    projection_salt = _projection_salt_for_region(
        video_region if region_root is not None or direct_video_region_from_window else "2x_face_roi"
    )
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
    sequence_rows: list[dict[str, Any]] = []
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
        try:
            direct_source: dict[str, Any] | None = None
            if direct_video_region_from_window:
                direct_source = _sample_direct_region_frames(window, video_region=video_region, frame_count=int(frame_limit or REGION_CLIP_FRAME_COUNT))
                roi_video = Path(str(direct_source["source_video_path"]))
                if augmentation_profile == NO_AUGMENTATION_PROFILE:
                    encoded = _encode_pre_sampled_frames(
                        encoder,
                        direct_source["frames"],
                        source_fps=direct_source.get("source_fps"),
                        batch_size=int(batch_size),
                        device=device,
                    )
                else:
                    encoded = _encode_augmented_frame_views(
                        encoder,
                        direct_source["frames"],
                        source_fps=direct_source.get("source_fps"),
                        batch_size=int(batch_size),
                        device=device,
                        augmentation_profile=augmentation_profile,
                        augmentation_views=int(augmentation_views),
                    )
            else:
                roi_video, effective_input_region, fallback_used = _input_video_path(
                    openface_cache_root=cache_root,
                    region_cache_root=region_root,
                    sample_id=sample_id,
                    openface_encoder_profile=openface_encoder_profile,
                    video_region=video_region,
                    fallback_video_region=fallback_video_region,
                )
                if not roi_video.is_file():
                    rows.append(
                        _masked_row(
                            window,
                            roi_video,
                            encoder_version=encoder_version,
                            video_region=video_region,
                            missing_roi_video=True,
                            quality_extra=_fallback_quality(
                                requested_region=video_region,
                                effective_region=effective_input_region,
                                fallback_region=fallback_video_region,
                                fallback_used=fallback_used,
                            ),
                        )
                    )
                    sequence_rows.append(_frame_sequence_record(window, None, mask_value=0, encoder_version=encoder_version))
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
                if augmentation_profile == NO_AUGMENTATION_PROFILE:
                    encoded = encoder(
                        roi_video,
                        fps=float(fps),
                        max_frames=frame_limit,
                        batch_size=int(batch_size),
                        device=device,
                    )
                else:
                    frames, source_fps = _sample_video_frames(roi_video, fps=float(fps), max_frames=frame_limit)
                    if not frames:
                        raise ValueError(f"no frames sampled from ROI video: {roi_video}")
                    encoded = _encode_augmented_frame_views(
                        encoder,
                        frames,
                        source_fps=source_fps,
                        batch_size=int(batch_size),
                        device=device,
                        augmentation_profile=augmentation_profile,
                        augmentation_views=int(augmentation_views),
                    )
            frame_embeddings = _encoded_frame_embeddings(encoded)
            pooled_embedding = _pool_frame_embeddings(frame_embeddings, temporal_pooling=temporal_pooling)
            embedding = _project_to_256(pooled_embedding, salt=projection_salt)
            sequence_rows.append(_frame_sequence_record(window, frame_embeddings, mask_value=1, encoder_version=encoder_version))
            quality_extra = _direct_quality_flags(direct_source) if direct_source is not None else None
            rows.append(
                _embedding_row(
                    window,
                    roi_video,
                    embedding,
                    encoded,
                    encoder_version=encoder_version,
                    video_region=video_region,
                    pooled_embedding=pooled_embedding,
                    frame_embeddings=frame_embeddings,
                    temporal_pooling=temporal_pooling,
                    projection_salt=projection_salt,
                    quality_extra={
                        **_fallback_quality(
                            requested_region=video_region,
                            effective_region=video_region if direct_source is not None else effective_input_region,
                            fallback_region=fallback_video_region,
                            fallback_used=False if direct_source is not None else fallback_used,
                        ),
                        **(quality_extra or {}),
                    },
                )
            )
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
            if direct_video_region_from_window:
                roi_video = Path(str(_default_source_from_window(window)["source_video_path"] or sample_id))
            else:
                roi_video, _effective_input_region, _fallback_used = _input_video_path(
                    openface_cache_root=cache_root,
                    region_cache_root=region_root,
                    sample_id=sample_id,
                    openface_encoder_profile=openface_encoder_profile,
                    video_region=video_region,
                    fallback_video_region=fallback_video_region,
                )
            rows.append(_masked_row(window, roi_video, encoder_version=encoder_version, video_region=video_region, error=str(exc)))
            sequence_rows.append(_frame_sequence_record(window, None, mask_value=0, encoder_version=encoder_version))
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
    if frame_sequences_out is not None:
        _write_frame_sequences_npz(sequence_rows, frame_sequences_out)
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
        "variant": encoder_version,
        "row_count": len(rows),
        "mask_sum": int(sum(int(row["modality_mask"][FACE_MASK_INDEX]) for row in rows)),
        "failure_count": int(len(failures)),
        "start_index": int(start_index),
        "source_row_count": int(source_row_count),
        "out_path": str(out_path),
        "frame_sequences_out": None if frame_sequences_out is None else str(frame_sequences_out),
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
        encoded = self.encode_frames(frames, batch_size=batch_size, device=device)
        encoded.update(
            {
                "sampled_frame_count": int(len(frames)),
                "usable_frame_count": int(len(frames)),
                "source_fps": None if source_fps is None else float(source_fps),
            }
        )
        return encoded

    def encode_frames(
        self,
        frames: list[np.ndarray],
        *,
        batch_size: int,
        device: str | None = None,
    ) -> dict[str, Any]:
        if not frames:
            raise ValueError("no frames provided to DINOv2 frame encoder")
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
            "frame_embeddings": stacked.astype(np.float32, copy=False),
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
    if max_frames is not None:
        frames = _resample_frames_to_count(frames, int(max_frames))
    return frames, source_fps if source_fps > 0 else None


def _resample_frames_to_count(frames: list[np.ndarray], count: int) -> list[np.ndarray]:
    if count <= 0 or not frames:
        return []
    if len(frames) == count:
        return frames
    if len(frames) == 1:
        return [frames[0]] * count
    indices = np.linspace(0, len(frames) - 1, num=count)
    return [frames[int(round(index))] for index in indices]


def _encode_pre_sampled_frames(
    encoder: FrameEncoder,
    frames: list[np.ndarray],
    *,
    source_fps: float | None,
    batch_size: int,
    device: str | None,
) -> dict[str, Any]:
    if hasattr(encoder, "encode_frames"):
        encoded = encoder.encode_frames(frames, batch_size=batch_size, device=device)  # type: ignore[attr-defined]
    else:
        encoded = encoder(frames, fps=0.0, max_frames=len(frames), batch_size=batch_size, device=device)
    encoded = dict(encoded)
    encoded.setdefault("sampled_frame_count", int(len(frames)))
    encoded.setdefault("usable_frame_count", int(len(frames)))
    encoded.setdefault("source_fps", None if source_fps is None else float(source_fps))
    return encoded


def _encode_augmented_frame_views(
    encoder: FrameEncoder,
    frames: list[np.ndarray],
    *,
    source_fps: float | None,
    batch_size: int,
    device: str | None,
    augmentation_profile: str,
    augmentation_views: int,
) -> dict[str, Any]:
    view_count = int(augmentation_views)
    frame_count = int(len(frames))
    frame_views = [
        frames if view_index == 0 else _augment_frames(frames, profile=augmentation_profile, view_index=view_index)
        for view_index in range(view_count)
    ]
    if hasattr(encoder, "encode_frames"):
        encoded = _encode_pre_sampled_frames(
            encoder,
            [frame for view_frames in frame_views for frame in view_frames],
            source_fps=source_fps,
            batch_size=batch_size,
            device=device,
        )
        flat_embeddings = _encoded_frame_embeddings(encoded)
        expected_rows = view_count * frame_count
        if flat_embeddings.shape[0] != expected_rows:
            raise ValueError(
                f"batched augmented frame encoder returned {flat_embeddings.shape[0]} rows, expected {expected_rows}"
            )
        encoded_views = list(flat_embeddings.reshape(view_count, frame_count, flat_embeddings.shape[1]))
    else:
        encoded_views = []
        for view_frames in frame_views:
            encoded = _encode_pre_sampled_frames(
                encoder,
                view_frames,
                source_fps=source_fps,
                batch_size=batch_size,
                device=device,
            )
            encoded_views.append(_encoded_frame_embeddings(encoded))
    first_shape = encoded_views[0].shape
    if any(view.shape != first_shape for view in encoded_views):
        shapes = [tuple(view.shape) for view in encoded_views]
        raise ValueError(f"augmented frame views returned inconsistent frame embedding shapes: {shapes}")
    averaged = np.mean(np.stack(encoded_views, axis=0), axis=0).astype(np.float32, copy=False)
    return {
        "frame_embeddings": averaged,
        "sampled_frame_count": int(len(frames)),
        "usable_frame_count": int(len(frames)),
        "source_fps": None if source_fps is None else float(source_fps),
        "augmentation_profile": augmentation_profile,
        "augmentation_views": int(augmentation_views),
        "augmentation_ops": _augmentation_ops(augmentation_profile),
    }


def _augment_frames(frames: list[np.ndarray], *, profile: str, view_index: int) -> list[np.ndarray]:
    if profile not in SUPPORTED_AUGMENTATION_PROFILES or profile == NO_AUGMENTATION_PROFILE:
        raise ValueError(f"unsupported augmentation_profile: {profile}")
    policy = _augmentation_policy(profile)
    return [
        _augment_frame(
            frame,
            view_index=view_index,
            frame_index=frame_index,
            policy=policy,
        )
        for frame_index, frame in enumerate(frames)
    ]


def _augment_frame(
    frame: np.ndarray,
    *,
    view_index: int,
    frame_index: int = 0,
    policy: dict[str, bool] | None = None,
) -> np.ndarray:
    policy = policy or _augmentation_policy(V4D_MILD_AUGMENTATION_PROFILE)
    values = np.asarray(frame, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"expected RGB frame with shape [H,W,3], got {values.shape}")
    direction = -1.0 if view_index % 2 == 0 else 1.0
    strength = _augmentation_strength(policy)
    brightness = 1.0 + direction * strength["brightness"]
    contrast = 1.0 + direction * strength["contrast"]
    channel_scale = np.asarray(
        [
            1.0 + direction * strength["red"],
            1.0 - direction * strength["green"],
            1.0 + direction * strength["blue"],
        ],
        dtype=np.float32,
    ).reshape(1, 1, 3)
    jittered = values
    if policy["contrast"]:
        jittered = (jittered - 127.5) * contrast + 127.5
    jittered = jittered * brightness * channel_scale
    if policy["grayscale"] and _use_deterministic_random_op("random_grayscale", view_index, frame_index):
        jittered = _to_grayscale_rgb(jittered)
    shifted = jittered
    if policy["crop_scale"]:
        crop_fraction = 0.03 + 0.01 * float(view_index % 3)
        shifted = _crop_resize_like(jittered, crop_fraction=crop_fraction, shift_direction=direction)
    if policy["light_blur"] and _use_deterministic_random_op("light_blur", view_index, frame_index):
        shifted = _light_blur(shifted)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _augmentation_policy(profile: str) -> dict[str, bool]:
    if profile == V4D_WEAK_COLOR_BRIGHTNESS_CONTRAST_PROFILE:
        return {"contrast": True, "grayscale": False, "light_blur": False, "crop_scale": False, "weak": True}
    if profile == V4D_A1_COLOR_BRIGHTNESS_PROFILE:
        return {"contrast": False, "grayscale": False, "light_blur": False, "crop_scale": False}
    if profile == V4D_A2_COLOR_BRIGHTNESS_GRAYSCALE_PROFILE:
        return {"contrast": False, "grayscale": True, "light_blur": False, "crop_scale": False}
    if profile == V4D_A3_COLOR_BRIGHTNESS_GRAYSCALE_CROP_SCALE_PROFILE:
        return {"contrast": False, "grayscale": True, "light_blur": False, "crop_scale": True}
    if profile == V4D_MILD_AUGMENTATION_PROFILE:
        return {"contrast": True, "grayscale": False, "light_blur": False, "crop_scale": True}
    if profile == V4D_APPEARANCE_AUGMENTATION_PROFILE:
        return {"contrast": True, "grayscale": True, "light_blur": True, "crop_scale": True}
    raise ValueError(f"unsupported augmentation_profile: {profile}")


def _augmentation_strength(policy: dict[str, bool]) -> dict[str, float]:
    if bool(policy.get("weak")):
        return {"brightness": 0.015, "contrast": 0.01, "red": 0.008, "green": 0.006, "blue": 0.004}
    return {"brightness": 0.06, "contrast": 0.04, "red": 0.03, "green": 0.02, "blue": 0.01}


def _use_deterministic_random_op(op_name: str, view_index: int, frame_index: int) -> bool:
    if op_name == "random_grayscale":
        return (int(view_index) + int(frame_index)) % 2 == 1
    if op_name == "light_blur":
        return (int(view_index) + int(frame_index)) % 2 == 0
    raise ValueError(f"unsupported deterministic augmentation op: {op_name}")


def _to_grayscale_rgb(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    gray = 0.299 * values[:, :, 0] + 0.587 * values[:, :, 1] + 0.114 * values[:, :, 2]
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32, copy=False)


def _light_blur(frame: np.ndarray) -> np.ndarray:
    values = np.asarray(frame, dtype=np.float32)
    if values.shape[0] < 3 or values.shape[1] < 3:
        return values
    try:
        import cv2

        return cv2.blur(values, (3, 3)).astype(np.float32, copy=False)
    except ImportError:
        pass
    padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blurred = (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    ) / 9.0
    return blurred.astype(np.float32, copy=False)


def _crop_resize_like(frame: np.ndarray, *, crop_fraction: float, shift_direction: float) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 2 or width <= 2:
        return frame
    margin_y = max(1, int(round(float(height) * crop_fraction)))
    margin_x = max(1, int(round(float(width) * crop_fraction)))
    shift_y = int(round(shift_direction * margin_y * 0.5))
    shift_x = int(round(-shift_direction * margin_x * 0.5))
    top = max(0, min(height - 2, margin_y + shift_y))
    left = max(0, min(width - 2, margin_x + shift_x))
    bottom = max(top + 1, min(height, height - margin_y + shift_y))
    right = max(left + 1, min(width, width - margin_x + shift_x))
    cropped = frame[top:bottom, left:right]
    return _resize_nearest(cropped, height=height, width=width)


def _resize_nearest(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    if frame.shape[0] == height and frame.shape[1] == width:
        return frame
    try:
        import cv2

        return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_NEAREST)
    except ImportError:
        pass
    y_indices = np.linspace(0, frame.shape[0] - 1, num=height)
    x_indices = np.linspace(0, frame.shape[1] - 1, num=width)
    return frame[np.rint(y_indices).astype(int)][:, np.rint(x_indices).astype(int)]


def _augmentation_ops(profile: str) -> list[str]:
    if profile == NO_AUGMENTATION_PROFILE:
        return []
    if profile == V4D_MILD_AUGMENTATION_PROFILE:
        return ["brightness", "contrast", "color_jitter", "crop_jitter", "scale_jitter"]
    if profile == V4D_APPEARANCE_AUGMENTATION_PROFILE:
        return [
            "brightness",
            "contrast",
            "color_jitter",
            "random_grayscale",
            "light_blur",
            "crop_jitter",
            "scale_jitter",
        ]
    if profile == V4D_A1_COLOR_BRIGHTNESS_PROFILE:
        return ["brightness", "color_jitter"]
    if profile == V4D_A2_COLOR_BRIGHTNESS_GRAYSCALE_PROFILE:
        return ["brightness", "color_jitter", "random_grayscale"]
    if profile == V4D_A3_COLOR_BRIGHTNESS_GRAYSCALE_CROP_SCALE_PROFILE:
        return ["brightness", "color_jitter", "random_grayscale", "crop_jitter", "scale_jitter"]
    if profile == V4D_WEAK_COLOR_BRIGHTNESS_CONTRAST_PROFILE:
        return ["weak_brightness", "weak_contrast", "weak_color_jitter"]
    raise ValueError(f"unsupported augmentation_profile: {profile}")


def _sample_direct_region_frames(
    window: dict[str, Any],
    *,
    video_region: str,
    frame_count: int,
) -> dict[str, Any]:
    if video_region not in {"upper_body", "full_frame"}:
        raise ValueError(f"direct region sampling does not support {video_region}")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("direct video region extraction requires opencv-python or opencv-python-headless") from exc
    source = _default_source_from_window(window)
    source_video = Path(str(source["source_video_path"] or ""))
    if not source_video.is_file():
        raise ValueError(f"could not open source video: {source_video}")
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"could not open source video: {source_video}")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if source_fps <= 0:
            source_fps = 30.0
        start_frame = int(round(float(source["clip_start_seconds"] or 0.0) * source_fps))
        if source["clip_end_seconds"] is None:
            end_frame = max(start_frame + 1, frame_total)
        else:
            end_frame = int(round(float(source["clip_end_seconds"]) * source_fps))
        if frame_total > 0:
            start_frame = max(0, min(start_frame, frame_total - 1))
            end_frame = max(start_frame + 1, min(end_frame, frame_total))
        else:
            end_frame = max(start_frame + 1, end_frame)
        target_count = max(1, int(frame_count))
        target_indices = [int(round(float(index))) for index in np.linspace(start_frame, end_frame - 1, num=target_count)]
        crop_bbox = _upper_body_bbox_from_window(window) if video_region == "upper_body" else None
        effective_region = video_region
        fallback_full_frame = False
        if video_region == "upper_body" and crop_bbox is None:
            effective_region = "full_frame"
            fallback_full_frame = True
        frames: list[np.ndarray] = []
        target_position = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_index in range(start_frame, end_frame):
            ok, frame_bgr = cap.read()
            if not ok:
                break
            while target_position < len(target_indices) and frame_index == target_indices[target_position]:
                region_bgr = _prepare_region_frame(frame_bgr, crop_bbox=crop_bbox, cv2=cv2)
                frames.append(cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB))
                target_position += 1
            if target_position >= len(target_indices):
                break
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"no frames sampled from source video: {source_video}")
    if len(frames) != max(1, int(frame_count)):
        frames = _resample_frames_to_count(frames, max(1, int(frame_count)))
    return {
        "frames": frames,
        "source_video_path": str(source_video),
        "source_fps": None if source_fps <= 0 else float(source_fps),
        "clip_start_seconds": _json_ready(source["clip_start_seconds"]),
        "clip_end_seconds": _json_ready(source["clip_end_seconds"]),
        "crop_bbox": crop_bbox,
        "effective_region": effective_region,
        "upper_body_fallback_full_frame": fallback_full_frame,
        "direct_video_region_from_window": True,
    }


def _roi_video_path(cache_root: Path, sample_id: str, profile: str) -> Path:
    return cache_root / "openface" / sample_id / profile / "window.mp4"


def _region_video_path(cache_root: Path, sample_id: str, region: str) -> Path:
    return cache_root / region / sample_id / "window.mp4"


def _input_video_path(
    *,
    openface_cache_root: Path | None,
    region_cache_root: Path | None,
    sample_id: str,
    openface_encoder_profile: str,
    video_region: str,
    fallback_video_region: str | None = None,
) -> tuple[Path, str, bool]:
    if region_cache_root is not None:
        primary = _region_video_path(region_cache_root, sample_id, video_region)
        if primary.is_file() or fallback_video_region is None:
            return primary, video_region, False
        fallback = _region_video_path(region_cache_root, sample_id, fallback_video_region)
        if fallback.is_file():
            return fallback, fallback_video_region, True
        return primary, video_region, False
    if openface_cache_root is None:
        raise ValueError("openface_cache_root is required when region_cache_root is not set")
    return _roi_video_path(openface_cache_root, sample_id, openface_encoder_profile), "2x_face_roi", False


def _encoder_version_for_region(
    region: str,
    *,
    augmentation_profile: str = NO_AUGMENTATION_PROFILE,
    fallback_video_region: str | None = None,
) -> str:
    if region == "2x_face_roi":
        region_name = "2xroi"
    else:
        region_name = region.replace("2x_face_roi", "2xroi")
    if fallback_video_region:
        region_name = f"{region_name}_{fallback_video_region}_fallback"
    if augmentation_profile == NO_AUGMENTATION_PROFILE:
        if region == "2x_face_roi":
            return DINO_ENCODER_VERSION
        return f"video_v4a_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_MILD_AUGMENTATION_PROFILE:
        return f"video_v4d_aug_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_APPEARANCE_AUGMENTATION_PROFILE:
        return f"video_v4d_appearance_aug_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_A1_COLOR_BRIGHTNESS_PROFILE:
        return f"video_v4d_a1_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_A2_COLOR_BRIGHTNESS_GRAYSCALE_PROFILE:
        return f"video_v4d_a2_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_A3_COLOR_BRIGHTNESS_GRAYSCALE_CROP_SCALE_PROFILE:
        return f"video_v4d_a3_dinov2_{region_name}_mean_std_max"
    if augmentation_profile == V4D_WEAK_COLOR_BRIGHTNESS_CONTRAST_PROFILE:
        return f"video_v4d_weak_aug_dinov2_{region_name}_mean_std_max"
    raise ValueError(f"unsupported augmentation_profile: {augmentation_profile}")


def _projection_salt_for_region(region: str) -> str:
    if region == "2x_face_roi":
        return DINO_ENCODER_VERSION
    region_name = region.replace("2x_face_roi", "2xroi")
    return f"video_v4a_dinov2_{region_name}_mean_std_max"


def _embedding_row(
    window: dict[str, Any],
    roi_video: Path,
    embedding: np.ndarray,
    encoded: dict[str, Any],
    *,
    encoder_version: str,
    video_region: str,
    pooled_embedding: np.ndarray,
    frame_embeddings: np.ndarray,
    temporal_pooling: str,
    projection_salt: str,
    quality_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mask = np.zeros(4, dtype=np.int8)
    mask[FACE_MASK_INDEX] = 1
    quality = _base_quality(roi_video, encoder_version=encoder_version, video_region=video_region)
    quality.update(
        {
            "sampled_frame_count": int(encoded.get("sampled_frame_count", 0) or 0),
            "usable_frame_count": int(encoded.get("usable_frame_count", 0) or 0),
            "source_fps": _json_ready(encoded.get("source_fps")),
            "temporal_pooling": temporal_pooling,
            "pooled_stat_names": ["mean", "std", "max"],
            "frame_embedding_dim": int(frame_embeddings.shape[1]),
            "projected_from_dim": int(np.asarray(pooled_embedding).reshape(-1).shape[0]),
            "projection_salt": projection_salt,
            "projection_input_dim": int(np.asarray(pooled_embedding).reshape(-1).shape[0]),
            "augmentation_profile": str(encoded.get("augmentation_profile", NO_AUGMENTATION_PROFILE)),
            "augmentation_views": int(encoded.get("augmentation_views", 1) or 1),
            "augmentation_ops": _json_ready(encoded.get("augmentation_ops", [])),
        }
    )
    if quality_extra:
        quality.update(quality_extra)
    return _row(window, embedding=embedding, mask=mask, quality_flags=quality, encoder_version=encoder_version)


def _masked_row(
    window: dict[str, Any],
    roi_video: Path,
    *,
    encoder_version: str = DINO_ENCODER_VERSION,
    video_region: str = "2x_face_roi",
    missing_roi_video: bool = False,
    error: str | None = None,
    quality_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    quality = _base_quality(roi_video, encoder_version=encoder_version, video_region=video_region)
    quality["missing_roi_video"] = bool(missing_roi_video)
    if error:
        quality["error"] = error
    if quality_extra:
        quality.update(quality_extra)
    return _row(
        window,
        embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        mask=np.zeros(4, dtype=np.int8),
        quality_flags=quality,
        encoder_version=encoder_version,
    )


def _row(
    window: dict[str, Any],
    *,
    embedding: np.ndarray,
    mask: np.ndarray,
    quality_flags: dict[str, Any],
    encoder_version: str = DINO_ENCODER_VERSION,
) -> dict[str, Any]:
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "labels": _window_labels(window),
        "face_emb": validate_embedding_shape("face_emb", np.asarray(embedding, dtype=np.float32)),
        "modality_mask": mask.astype(np.int8),
        "quality_flags": quality_flags,
        "encoder_version": encoder_version,
    }


def _base_quality(roi_video: Path, *, encoder_version: str = DINO_ENCODER_VERSION, video_region: str = "2x_face_roi") -> dict[str, Any]:
    return {
        "variant": encoder_version,
        "dinov2_model": "facebook/dinov2-base",
        "frozen": True,
        "input": f"video_region_{video_region}_window_mp4" if video_region != "2x_face_roi" else "openface_mainface_2x_roi_window_mp4",
        "input_region": video_region,
        "roi_crop_scale": 2.0,
        "roi_video_path": str(roi_video),
    }


def _direct_quality_flags(source: dict[str, Any] | None) -> dict[str, Any]:
    if source is None:
        return {}
    return {
        "input": f"direct_source_video_{source.get('effective_region', '')}_frames",
        "direct_video_region_from_window": True,
        "source_video_path": str(source.get("source_video_path", "")),
        "clip_start_seconds": _json_ready(source.get("clip_start_seconds")),
        "clip_end_seconds": _json_ready(source.get("clip_end_seconds")),
        "crop_bbox": _json_ready(source.get("crop_bbox")),
        "effective_region": str(source.get("effective_region", "")),
        "upper_body_fallback_full_frame": bool(source.get("upper_body_fallback_full_frame", False)),
    }


def _fallback_quality(
    *,
    requested_region: str,
    effective_region: str,
    fallback_region: str | None,
    fallback_used: bool,
) -> dict[str, Any]:
    if fallback_region is None:
        return {}
    return {
        "requested_input_region": requested_region,
        "effective_input_region": effective_region,
        "fallback_video_region": fallback_region,
        "video_region_fallback_used": bool(fallback_used),
    }


def _frame_sequence_record(
    window: dict[str, Any],
    frame_embeddings: np.ndarray | None,
    *,
    mask_value: int,
    encoder_version: str = DINO_ENCODER_VERSION,
) -> dict[str, Any]:
    mask = np.zeros(4, dtype=np.int8)
    mask[FACE_MASK_INDEX] = int(mask_value)
    return {
        "sample_id": str(window.get("sample_id", "")),
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "labels": _window_labels(window),
        "frame_embeddings": None if frame_embeddings is None else np.asarray(frame_embeddings, dtype=np.float32),
        "modality_mask": mask,
        "encoder_version": encoder_version,
    }


def _encoded_frame_embeddings(encoded: dict[str, Any]) -> np.ndarray:
    if "frame_embeddings" in encoded:
        values = np.asarray(encoded["frame_embeddings"], dtype=np.float32)
    elif "embedding" in encoded:
        values = np.asarray(encoded["embedding"], dtype=np.float32)
    else:
        raise ValueError("DINOv2 frame encoder must return frame_embeddings or embedding")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"DINOv2 frame embeddings must be 1D or 2D, got {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"DINOv2 frame embeddings must be non-empty, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("DINOv2 frame embeddings contain non-finite values")
    return values.astype(np.float32, copy=False)


def _pool_frame_embeddings(frame_embeddings: np.ndarray, *, temporal_pooling: str) -> np.ndarray:
    if temporal_pooling != DEFAULT_TEMPORAL_POOLING:
        raise ValueError(f"unsupported temporal_pooling: {temporal_pooling}")
    values = np.asarray(frame_embeddings, dtype=np.float32)
    pooled = np.concatenate(
        [
            values.mean(axis=0),
            values.std(axis=0),
            values.max(axis=0),
        ],
        axis=0,
    )
    return pooled.astype(np.float32, copy=False)


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


def _write_frame_sequences_npz(rows: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    available = [row["frame_embeddings"] for row in rows if row["frame_embeddings"] is not None]
    if available:
        frame_count = max(int(value.shape[0]) for value in available)
        frame_dim = int(available[0].shape[1])
    else:
        frame_count = 0
        frame_dim = 0
    frame_sequences = np.zeros((len(rows), frame_count, frame_dim), dtype=np.float32)
    frame_sequence_mask = np.zeros((len(rows), frame_count), dtype=np.int8)
    for index, row in enumerate(rows):
        values = row["frame_embeddings"]
        if values is None:
            continue
        if values.ndim != 2:
            raise ValueError(f"frame sequence row {index} must be 2D, got {values.shape}")
        if int(values.shape[1]) != frame_dim:
            raise ValueError(f"frame sequence row {index} dim {values.shape[1]} does not match {frame_dim}")
        usable = min(frame_count, int(values.shape[0]))
        frame_sequences[index, :usable, :] = values[:usable]
        frame_sequence_mask[index, :usable] = 1
    np.savez_compressed(
        out,
        sample_id=np.array([row["sample_id"] for row in rows], dtype=object),
        event_id=np.array([row["event_id"] for row in rows], dtype=object),
        subject_id=np.array([row["subject_id"] for row in rows], dtype=object),
        labels=np.array([json.dumps(_json_ready(row["labels"]), ensure_ascii=False, allow_nan=False) for row in rows], dtype=object),
        frame_embeddings=frame_sequences,
        frame_sequence_mask=frame_sequence_mask,
        modality_mask=np.stack([row["modality_mask"] for row in rows]).astype(np.int8)
        if rows
        else np.zeros((0, 4), dtype=np.int8),
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
