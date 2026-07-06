from __future__ import annotations

import json
import sys

import numpy as np

import daily_multimodal.embeddings.dinov2_roi as dinov2_roi
from daily_multimodal.embeddings.dinov2_roi import build_dinov2_roi_embeddings, _resample_frames_to_count


def test_build_dinov2_roi_embeddings_freezes_v4a_pooling_contract(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "video_v4a.npz"
    sample_id = "sub-01_ses-01_win-0000"
    roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
    roi_dir.mkdir(parents=True)
    roi_video = roi_dir / "window.mp4"
    roi_video.write_bytes(b"fake-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "sub-01_ses-01_row-0001",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []
    frame_sequence = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)

    def fake_encoder(path, *, fps, max_frames, batch_size, device):
        calls.append((path, fps, max_frames, batch_size, device))
        return {
            "frame_embeddings": frame_sequence,
            "sampled_frame_count": 16,
            "usable_frame_count": 16,
            "source_fps": 30.0,
        }

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        fps=2.0,
        num_frames=16,
        temporal_pooling="mean_std_max",
        batch_size=8,
        device="cpu",
        frame_encoder=fake_encoder,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert summary["variant"] == "video_v4a_dinov2_2xroi_mean_std_max"
    assert calls == [(roi_video, 2.0, 16, 8, "cpu")]
    assert loaded["encoder_version"].astype(str).tolist() == ["video_v4a_dinov2_2xroi_mean_std_max"]
    assert loaded["face_emb"].shape == (1, 256)
    assert np.any(loaded["face_emb"][0] != 0)
    assert quality["temporal_pooling"] == "mean_std_max"
    assert quality["sampled_frame_count"] == 16
    assert quality["usable_frame_count"] == 16
    assert quality["pooled_stat_names"] == ["mean", "std", "max"]
    assert quality["frame_embedding_dim"] == 8
    assert quality["projected_from_dim"] == 24


def test_resample_frames_to_count_upsamples_low_fps_roi_to_v4a_frame_count():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(5)]

    sampled = _resample_frames_to_count(frames, 16)

    assert len(sampled) == 16
    assert int(sampled[0][0, 0, 0]) == 0
    assert int(sampled[-1][0, 0, 0]) == 4
    assert {int(frame[0, 0, 0]) for frame in sampled} == {0, 1, 2, 3, 4}


def test_build_dinov2_roi_embeddings_can_use_v4d_mild_frame_augmentation(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "video_v4d_aug.npz"
    frame_sequences_out = tmp_path / "video_v4d_aug_frame_sequences.npz"
    sample_id = "sample-aug"
    roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
    roi_dir.mkdir(parents=True)
    roi_video = roi_dir / "window.mp4"
    roi_video.write_bytes(b"fake-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "event-aug",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    base_frames = [np.full((8, 8, 3), 100, dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr(dinov2_roi, "_sample_video_frames", lambda path, fps, max_frames: (base_frames, 30.0))

    class FakeFrameEncoder:
        def __init__(self):
            self.view_means = []

        def encode_frames(self, frames, *, batch_size, device):
            del batch_size, device
            means = np.asarray([float(frame.mean()) for frame in frames], dtype=np.float32)
            self.view_means.append(means)
            return {"frame_embeddings": np.repeat(means[:, None], 8, axis=1)}

    fake_encoder = FakeFrameEncoder()

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        frame_sequences_out=frame_sequences_out,
        num_frames=4,
        batch_size=2,
        frame_encoder=fake_encoder,
        augmentation_profile="v4d_mild_color_crop_scale",
        augmentation_views=2,
    )

    loaded = np.load(out_path, allow_pickle=True)
    sequence = np.load(frame_sequences_out, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert summary["variant"] == "video_v4d_aug_dinov2_2xroi_mean_std_max"
    assert loaded["encoder_version"].astype(str).tolist() == ["video_v4d_aug_dinov2_2xroi_mean_std_max"]
    assert sequence["encoder_version"].astype(str).tolist() == ["video_v4d_aug_dinov2_2xroi_mean_std_max"]
    assert len(fake_encoder.view_means) == 1
    assert fake_encoder.view_means[0].shape == (8,)
    assert np.allclose(fake_encoder.view_means[0][:4], 100.0)
    assert not np.allclose(fake_encoder.view_means[0][4:], 100.0)
    assert quality["augmentation_profile"] == "v4d_mild_color_crop_scale"
    assert quality["augmentation_views"] == 2
    assert quality["augmentation_ops"] == ["brightness", "contrast", "color_jitter", "crop_jitter", "scale_jitter"]
    assert quality["sampled_frame_count"] == 4
    assert sequence["frame_embeddings"].shape == (1, 4, 8)


def test_v4d_upper_body_augmentation_uses_v4a_projection_salt(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    region_root = tmp_path / "regions"
    out_path = tmp_path / "video_v4d_a1.npz"
    sample_id = "sample-upper"
    video_dir = region_root / "upper_body" / sample_id
    video_dir.mkdir(parents=True)
    (video_dir / "window.mp4").write_bytes(b"fake-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "event-upper",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frames = [np.full((4, 4, 3), 100, dtype=np.uint8) for _ in range(4)]
    monkeypatch.setattr(dinov2_roi, "_sample_video_frames", lambda path, fps, max_frames: (frames, 30.0))

    class FakeFrameEncoder:
        def encode_frames(self, frame_batch, *, batch_size, device):
            del batch_size, device
            values = np.asarray([float(frame.mean()) for frame in frame_batch], dtype=np.float32)
            return {"frame_embeddings": np.repeat(values[:, None], 8, axis=1)}

    build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=None,
        openface_encoder_profile="openface_temporal_v1",
        region_cache_root=region_root,
        video_region="upper_body",
        fallback_video_region="full_frame",
        out_path=out_path,
        num_frames=4,
        frame_encoder=FakeFrameEncoder(),
        augmentation_profile="v4d_a1_color_brightness",
        augmentation_views=2,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert loaded["encoder_version"].astype(str).tolist() == [
        "video_v4d_a1_dinov2_upper_body_full_frame_fallback_mean_std_max"
    ]
    assert quality["projection_salt"] == "video_v4a_dinov2_upper_body_mean_std_max"
    assert quality["projection_input_dim"] == 24


def test_v4d_appearance_augmentation_adds_grayscale_and_light_blur_ops():
    frame = np.zeros((5, 5, 3), dtype=np.uint8)
    frame[:, :, 0] = 20
    frame[:, :, 1] = 100
    frame[:, :, 2] = 220
    frame[2, 2] = np.array([255, 0, 0], dtype=np.uint8)

    augmented = dinov2_roi._augment_frames([frame, frame], profile="v4d_appearance_mild", view_index=1)

    ops = dinov2_roi._augmentation_ops("v4d_appearance_mild")
    assert ops == [
        "brightness",
        "contrast",
        "color_jitter",
        "random_grayscale",
        "light_blur",
        "crop_jitter",
        "scale_jitter",
    ]
    assert any(np.all(candidate[..., 0] == candidate[..., 1]) and np.all(candidate[..., 1] == candidate[..., 2]) for candidate in augmented)
    assert any(0 < int(candidate[2, 2, 0]) < 255 for candidate in augmented)


def test_v4d_augmentation_batches_views_when_encoder_supports_frame_batches():
    frames = [np.full((4, 4, 3), value, dtype=np.uint8) for value in [40, 120]]

    class FakeFrameBatchEncoder:
        def __init__(self):
            self.call_sizes = []

        def encode_frames(self, frame_batch, *, batch_size, device):
            del batch_size, device
            self.call_sizes.append(len(frame_batch))
            means = np.asarray([float(frame.mean()) for frame in frame_batch], dtype=np.float32)
            return {"frame_embeddings": np.repeat(means[:, None], 3, axis=1)}

    encoder = FakeFrameBatchEncoder()

    encoded = dinov2_roi._encode_augmented_frame_views(
        encoder,
        frames,
        source_fps=30.0,
        batch_size=16,
        device="cpu",
        augmentation_profile="v4d_appearance_mild",
        augmentation_views=2,
    )

    assert encoder.call_sizes == [4]
    assert encoded["frame_embeddings"].shape == (2, 3)
    assert encoded["augmentation_ops"] == [
        "brightness",
        "contrast",
        "color_jitter",
        "random_grayscale",
        "light_blur",
        "crop_jitter",
        "scale_jitter",
    ]


def test_v4d_ablation_augmentation_profiles_match_a1_a2_a3_definitions():
    frame = np.zeros((6, 6, 3), dtype=np.uint8)
    frame[:, :, 0] = 30
    frame[:, :, 1] = 90
    frame[:, :, 2] = 210
    frame[0, 0] = np.array([255, 0, 0], dtype=np.uint8)

    a1 = dinov2_roi._augment_frames([frame], profile="v4d_a1_color_brightness", view_index=1)[0]
    a2 = dinov2_roi._augment_frames([frame], profile="v4d_a2_color_brightness_grayscale", view_index=1)[0]
    a3 = dinov2_roi._augment_frames([frame], profile="v4d_a3_color_brightness_grayscale_crop_scale", view_index=1)[0]

    assert dinov2_roi._augmentation_ops("v4d_a1_color_brightness") == ["brightness", "color_jitter"]
    assert dinov2_roi._augmentation_ops("v4d_a2_color_brightness_grayscale") == [
        "brightness",
        "color_jitter",
        "random_grayscale",
    ]
    assert dinov2_roi._augmentation_ops("v4d_a3_color_brightness_grayscale_crop_scale") == [
        "brightness",
        "color_jitter",
        "random_grayscale",
        "crop_jitter",
        "scale_jitter",
    ]
    assert not np.allclose(a1, frame)
    assert np.all(a2[..., 0] == a2[..., 1])
    assert np.all(a2[..., 1] == a2[..., 2])
    assert not np.array_equal(a3[0, 0], a2[0, 0])


def test_v4d_weak_augmentation_keeps_only_light_color_brightness_contrast():
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 0] = 40
    frame[:, :, 1] = 120
    frame[:, :, 2] = 210
    frame[0, 0] = np.array([255, 0, 0], dtype=np.uint8)

    weak = dinov2_roi._augment_frames([frame], profile="v4d_weak_color_brightness_contrast", view_index=1)[0]
    a1 = dinov2_roi._augment_frames([frame], profile="v4d_a1_color_brightness", view_index=1)[0]

    assert dinov2_roi._augmentation_ops("v4d_weak_color_brightness_contrast") == [
        "weak_brightness",
        "weak_contrast",
        "weak_color_jitter",
    ]
    assert not np.allclose(weak, frame)
    assert not np.all(weak[..., 0] == weak[..., 1])
    assert np.array_equal(weak[0, 0] > 0, a1[0, 0] > 0)
    assert float(np.mean(np.abs(weak.astype(np.float32) - frame.astype(np.float32)))) < float(
        np.mean(np.abs(a1.astype(np.float32) - frame.astype(np.float32)))
    )


def test_build_dinov2_roi_embeddings_can_write_frame_sequence_bundle_for_v4b(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "video_v4a.npz"
    frame_sequences_out = tmp_path / "video_v4a_frame_sequences.npz"
    for sample_id in ["sample-1", "sample-2"]:
        roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
        roi_dir.mkdir(parents=True)
        (roi_dir / "window.mp4").write_bytes(b"fake-video")
    window_index.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "event_id": f"{sample_id}-event",
                    "subject_id": "sub-01",
                    "label_columns": {"fatigue": float(index)},
                }
            )
            + "\n"
            for index, sample_id in enumerate(["sample-1", "sample-2"], start=1)
        ),
        encoding="utf-8",
    )

    def fake_encoder(path, *, fps, max_frames, batch_size, device):
        del fps, batch_size, device
        offset = 0.0 if path.parent.parent.name == "sample-1" else 100.0
        return {
            "frame_embeddings": np.full((int(max_frames), 8), offset + 1.0, dtype=np.float32),
            "sampled_frame_count": int(max_frames),
            "usable_frame_count": int(max_frames),
        }

    build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        frame_sequences_out=frame_sequences_out,
        num_frames=16,
        frame_encoder=fake_encoder,
    )

    loaded = np.load(frame_sequences_out, allow_pickle=True)
    assert loaded["sample_id"].astype(str).tolist() == ["sample-1", "sample-2"]
    assert loaded["event_id"].astype(str).tolist() == ["sample-1-event", "sample-2-event"]
    assert loaded["subject_id"].astype(str).tolist() == ["sub-01", "sub-01"]
    assert loaded["frame_embeddings"].shape == (2, 16, 8)
    assert loaded["frame_sequence_mask"].shape == (2, 16)
    assert loaded["frame_sequence_mask"].tolist() == [[1] * 16, [1] * 16]
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
    assert loaded["encoder_version"].astype(str).tolist() == [
        "video_v4a_dinov2_2xroi_mean_std_max",
        "video_v4a_dinov2_2xroi_mean_std_max",
    ]
    assert json.loads(loaded["labels"][0]) == {"fatigue": 1.0}


def test_build_dinov2_roi_embeddings_can_limit_windows_for_smoke_runs(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "video_v4a.npz"
    for sample_id in ["sample-1", "sample-2"]:
        roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
        roi_dir.mkdir(parents=True)
        (roi_dir / "window.mp4").write_bytes(b"fake-video")
    window_index.write_text(
        "".join(json.dumps({"sample_id": sample_id, "event_id": sample_id, "subject_id": "sub-01"}) + "\n" for sample_id in ["sample-1", "sample-2"]),
        encoding="utf-8",
    )

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        max_windows=1,
        frame_encoder=lambda *args, **kwargs: {
            "frame_embeddings": np.ones((16, 8), dtype=np.float32),
            "sampled_frame_count": 16,
            "usable_frame_count": 16,
        },
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["row_count"] == 1
    assert loaded["sample_id"].astype(str).tolist() == ["sample-1"]


def test_build_dinov2_roi_embeddings_can_start_from_window_offset_for_chunks(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "video_v4a_chunk.npz"
    sample_ids = ["sample-1", "sample-2", "sample-3", "sample-4"]
    for sample_id in sample_ids:
        roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
        roi_dir.mkdir(parents=True)
        (roi_dir / "window.mp4").write_bytes(b"fake-video")
    window_index.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "event_id": sample_id, "subject_id": "sub-01"}) + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        start_index=1,
        max_windows=2,
        frame_encoder=lambda *args, **kwargs: {
            "frame_embeddings": np.ones((16, 8), dtype=np.float32),
            "sampled_frame_count": 16,
            "usable_frame_count": 16,
        },
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["row_count"] == 2
    assert summary["start_index"] == 1
    assert summary["source_row_count"] == 4
    assert loaded["sample_id"].astype(str).tolist() == ["sample-2", "sample-3"]


def test_build_dinov2_roi_embeddings_can_read_video_region_cache(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    region_cache_root = tmp_path / "video_regions"
    out_path = tmp_path / "video_v4a_upper_body.npz"
    sample_id = "sample-upper"
    region_video = region_cache_root / "upper_body" / sample_id / "window.mp4"
    region_video.parent.mkdir(parents=True)
    region_video.write_bytes(b"fake-upper-body-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "event-upper",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 3.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_encoder(path, *, fps, max_frames, batch_size, device):
        calls.append(path)
        return {
            "frame_embeddings": np.ones((16, 8), dtype=np.float32),
            "sampled_frame_count": 16,
            "usable_frame_count": 16,
        }

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=None,
        openface_encoder_profile="openface_temporal_v1",
        region_cache_root=region_cache_root,
        video_region="upper_body",
        out_path=out_path,
        frame_encoder=fake_encoder,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert calls == [region_video]
    assert summary["variant"] == "video_v4a_dinov2_upper_body_mean_std_max"
    assert loaded["encoder_version"].astype(str).tolist() == ["video_v4a_dinov2_upper_body_mean_std_max"]
    assert quality["input_region"] == "upper_body"
    assert quality["input"] == "video_region_upper_body_window_mp4"
    assert quality["roi_video_path"] == str(region_video)


def test_build_dinov2_roi_embeddings_can_fallback_from_upper_body_to_full_frame_region_cache(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    region_cache_root = tmp_path / "video_regions"
    out_path = tmp_path / "video_v4d_upper_fallback.npz"
    sample_id = "sample-fallback"
    full_video = region_cache_root / "full_frame" / sample_id / "window.mp4"
    full_video.parent.mkdir(parents=True)
    full_video.write_bytes(b"fake-full-frame-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "event-fallback",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 3.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_encoder(path, *, fps, max_frames, batch_size, device):
        del fps, max_frames, batch_size, device
        calls.append(path)
        return {
            "frame_embeddings": np.ones((16, 8), dtype=np.float32),
            "sampled_frame_count": 16,
            "usable_frame_count": 16,
        }

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=None,
        openface_encoder_profile="openface_temporal_v1",
        region_cache_root=region_cache_root,
        video_region="upper_body",
        fallback_video_region="full_frame",
        out_path=out_path,
        frame_encoder=fake_encoder,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert calls == [full_video]
    assert summary["variant"] == "video_v4a_dinov2_upper_body_full_frame_fallback_mean_std_max"
    assert loaded["encoder_version"].astype(str).tolist() == [
        "video_v4a_dinov2_upper_body_full_frame_fallback_mean_std_max"
    ]
    assert quality["input_region"] == "upper_body"
    assert quality["requested_input_region"] == "upper_body"
    assert quality["effective_input_region"] == "full_frame"
    assert quality["fallback_video_region"] == "full_frame"
    assert quality["video_region_fallback_used"] is True
    assert quality["roi_video_path"] == str(full_video)


def test_build_dinov2_roi_embeddings_can_sample_upper_body_directly_from_window_source(tmp_path, monkeypatch):
    window_index = tmp_path / "window_index.jsonl"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    out_path = tmp_path / "video_v4a_upper_direct.npz"
    window_index.write_text(
        json.dumps(
            {
                "sample_id": "sample-direct",
                "event_id": "event-direct",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 4.0},
                "video_candidates": [
                    {
                        "mp4_path": str(source),
                        "clip_start_seconds": 1.0,
                        "clip_end_seconds": 3.0,
                    }
                ],
                "face_presence": {"main_face_bbox": [10, 10, 10, 10]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeCapture:
        def __init__(self, path):
            self.path = path
            self.position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == fake_cv2.CAP_PROP_FPS:
                return 5.0
            if prop == fake_cv2.CAP_PROP_FRAME_COUNT:
                return 100
            return 0

        def set(self, prop, value):
            if prop == fake_cv2.CAP_PROP_POS_FRAMES:
                self.position = int(value)

        def read(self):
            frame = np.full((120, 240, 3), self.position % 255, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self):
            return None

    class FakeCV2:
        CAP_PROP_FPS = 1
        CAP_PROP_FRAME_COUNT = 2
        CAP_PROP_POS_FRAMES = 3
        INTER_AREA = 4
        COLOR_BGR2RGB = 5

        @staticmethod
        def VideoCapture(path):
            return FakeCapture(path)

        @staticmethod
        def resize(frame, size, interpolation=None):
            width, height = size
            return np.zeros((height, width, 3), dtype=frame.dtype)

        @staticmethod
        def cvtColor(frame, code):
            del code
            return frame[..., ::-1]

    fake_cv2 = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def encode_frames(self, frames, *, batch_size, device):
            self.calls.append((len(frames), frames[0].shape, batch_size, device))
            return {"frame_embeddings": np.ones((len(frames), 8), dtype=np.float32)}

    encoder = FakeEncoder()
    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=None,
        openface_encoder_profile="openface_temporal_v1",
        video_region="upper_body",
        direct_video_region_from_window=True,
        out_path=out_path,
        num_frames=16,
        batch_size=4,
        device="cpu",
        frame_encoder=encoder,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert summary["variant"] == "video_v4a_dinov2_upper_body_mean_std_max"
    assert summary["mask_sum"] == 1
    assert encoder.calls == [(16, (960, 640, 3), 4, "cpu")]
    assert loaded["encoder_version"].astype(str).tolist() == ["video_v4a_dinov2_upper_body_mean_std_max"]
    assert quality["direct_video_region_from_window"] is True
    assert quality["source_video_path"] == str(source)
    assert quality["crop_bbox"] == [0, 5, 30, 50]
    assert quality["effective_region"] == "upper_body"
    assert quality["input"] == "direct_source_video_upper_body_frames"


def test_build_dinov2_roi_embeddings_uses_roi_window_video_and_writes_face_contract(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "dinov2_v4.npz"
    sample_id = "sub-01_ses-01_win-0000"
    roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
    roi_dir.mkdir(parents=True)
    roi_video = roi_dir / "window.mp4"
    roi_video.write_bytes(b"fake-video")
    window_index.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "event_id": "event-1",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 2.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    calls = []

    def fake_encoder(path, *, fps, max_frames, batch_size, device):
        calls.append((path, fps, max_frames, batch_size, device))
        return {
            "embedding": np.arange(768, dtype=np.float32),
            "sampled_frame_count": 20,
            "usable_frame_count": 20,
            "source_fps": 30.0,
        }

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        fps=2.0,
        max_frames_per_window=20,
        batch_size=8,
        device="cpu",
        frame_encoder=fake_encoder,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["row_count"] == 1
    assert summary["mask_sum"] == 1
    assert calls == [(roi_video, 2.0, 20, 8, "cpu")]
    assert loaded["face_emb"].shape == (1, 256)
    assert np.any(loaded["face_emb"][0] != 0)
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0]]
    assert loaded["encoder_version"].astype(str).tolist() == ["video_v4a_dinov2_2xroi_mean_std_max"]
    assert json.loads(loaded["labels"][0]) == {"fatigue": 2.0}
    quality = json.loads(loaded["quality_flags"][0])
    assert quality["roi_video_path"] == str(roi_video)
    assert quality["roi_crop_scale"] == 2.0
    assert quality["sampled_frame_count"] == 20
    assert quality["temporal_pooling"] == "mean_std_max"


def test_build_dinov2_roi_embeddings_does_not_inherit_openface_face_detection_mask(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "dinov2_v4.npz"
    sample_id = "face-detection-failed"
    roi_dir = cache_root / "openface" / sample_id / "openface_temporal_v1"
    roi_dir.mkdir(parents=True)
    (roi_dir / "window.mp4").write_bytes(b"fake-video")
    (roi_dir / "metadata.json").write_text(
        json.dumps(
            {
                "face_detection_success_rate": 0.0,
                "face_roi_detected_frame_count": 0,
                "masked": True,
            }
        ),
        encoding="utf-8",
    )
    window_index.write_text(
        json.dumps({"sample_id": sample_id, "event_id": "event-1", "subject_id": "sub-01"}) + "\n",
        encoding="utf-8",
    )

    build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        frame_encoder=lambda *args, **kwargs: {
            "embedding": np.ones(768, dtype=np.float32),
            "sampled_frame_count": 20,
            "usable_frame_count": 20,
        },
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0]]


def test_build_dinov2_roi_embeddings_masks_missing_roi_video(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    out_path = tmp_path / "dinov2_v4.npz"
    window_index.write_text(
        json.dumps({"sample_id": "missing", "event_id": "event-1", "subject_id": "sub-01"}) + "\n",
        encoding="utf-8",
    )

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=tmp_path / "cache",
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        frame_encoder=lambda *args, **kwargs: {"embedding": np.ones(768, dtype=np.float32)},
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["mask_sum"] == 0
    assert np.all(loaded["face_emb"][0] == 0)
    assert loaded["modality_mask"].tolist() == [[0, 0, 0, 0]]
    quality = json.loads(loaded["quality_flags"][0])
    assert quality["missing_roi_video"] is True


def test_build_dinov2_roi_embeddings_streams_progress_and_failures(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    cache_root = tmp_path / "cache"
    out_path = tmp_path / "dinov2_v4.npz"
    progress_path = tmp_path / "progress.log"
    failures_path = tmp_path / "failures.json"
    roi_dir = cache_root / "openface" / "good" / "openface_temporal_v1"
    roi_dir.mkdir(parents=True)
    (roi_dir / "window.mp4").write_bytes(b"fake-video")
    window_index.write_text(
        "\n".join(
            [
                json.dumps({"sample_id": "good", "event_id": "event-1", "subject_id": "sub-01"}),
                json.dumps({"sample_id": "missing", "event_id": "event-2", "subject_id": "sub-01"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_dinov2_roi_embeddings(
        window_index_path=window_index,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        out_path=out_path,
        progress_out=progress_path,
        failures_out=failures_path,
        progress_every=1,
        frame_encoder=lambda *args, **kwargs: {
            "embedding": np.ones(768, dtype=np.float32),
            "sampled_frame_count": 2,
            "usable_frame_count": 2,
        },
    )

    assert summary["failure_count"] == 1
    progress = progress_path.read_text(encoding="utf-8")
    assert "start 1/2" in progress
    assert "done 2/2" in progress
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    assert failures == [{"sample_id": "missing", "error_type": "missing_roi_video"}]
