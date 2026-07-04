from __future__ import annotations

import json

import numpy as np

from daily_multimodal.embeddings.dinov2_roi import build_dinov2_roi_embeddings


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
    assert loaded["encoder_version"].astype(str).tolist() == ["dinov2_base_roi_v4"]
    assert json.loads(loaded["labels"][0]) == {"fatigue": 2.0}
    quality = json.loads(loaded["quality_flags"][0])
    assert quality["roi_video_path"] == str(roi_video)
    assert quality["roi_crop_scale"] == 2.0
    assert quality["sampled_frame_count"] == 20


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
