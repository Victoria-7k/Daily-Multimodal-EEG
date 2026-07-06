from __future__ import annotations

import json

import numpy as np

from daily_multimodal.embeddings.video_temporal import build_video_temporal_embeddings


def test_v4b_tcn_and_transformer_write_face_slot_contract(tmp_path):
    sequence_path = tmp_path / "dinov2_frames.npz"
    tcn_out = tmp_path / "v4b_tcn.npz"
    transformer_out = tmp_path / "v4b_transformer.npz"
    frame_embeddings = np.arange(2 * 16 * 8, dtype=np.float32).reshape(2, 16, 8)
    np.savez_compressed(
        sequence_path,
        sample_id=np.asarray(["sample-1", "sample-2"], dtype=object),
        event_id=np.asarray(["event-1", "event-2"], dtype=object),
        subject_id=np.asarray(["sub-01", "sub-02"], dtype=object),
        labels=np.asarray([json.dumps({"fatigue": 2.0}), json.dumps({"fatigue": 3.0})], dtype=object),
        frame_embeddings=frame_embeddings,
        encoder_version=np.asarray(["video_v4a_dinov2_2xroi_mean_std_max"] * 2, dtype=object),
    )

    tcn_summary = build_video_temporal_embeddings(
        frame_sequences=sequence_path,
        out_path=tcn_out,
        temporal_encoder="tcn",
    )
    transformer_summary = build_video_temporal_embeddings(
        frame_sequences=sequence_path,
        out_path=transformer_out,
        temporal_encoder="temporal_transformer",
    )

    assert tcn_summary["encoder_version"] == "video_v4b_tcn_dinov2_2xroi"
    assert transformer_summary["encoder_version"] == "video_v4b_temporal_transformer_dinov2_2xroi"
    for out_path, version, encoder_name in [
        (tcn_out, "video_v4b_tcn_dinov2_2xroi", "tcn"),
        (transformer_out, "video_v4b_temporal_transformer_dinov2_2xroi", "temporal_transformer"),
    ]:
        loaded = np.load(out_path, allow_pickle=True)
        assert loaded["sample_id"].astype(str).tolist() == ["sample-1", "sample-2"]
        assert loaded["event_id"].astype(str).tolist() == ["event-1", "event-2"]
        assert loaded["subject_id"].astype(str).tolist() == ["sub-01", "sub-02"]
        assert "face_emb" in loaded.files
        assert loaded["face_emb"].shape == (2, 256)
        assert np.any(loaded["face_emb"][0] != 0)
        assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
        assert loaded["encoder_version"].astype(str).tolist() == [version, version]
        assert json.loads(loaded["labels"][0]) == {"fatigue": 2.0}
        quality = json.loads(loaded["quality_flags"][0])
        assert quality["temporal_encoder"] == encoder_name
        assert quality["input_frame_count"] == 16
        assert quality["frame_embedding_dim"] == 8
        assert quality["source_encoder_version"] == "video_v4a_dinov2_2xroi_mean_std_max"


def test_video_temporal_embeddings_mask_nonfinite_frame_rows(tmp_path):
    sequence_path = tmp_path / "bad_frames.npz"
    out_path = tmp_path / "v4b_tcn.npz"
    frame_embeddings = np.ones((2, 16, 8), dtype=np.float32)
    frame_embeddings[1, 0, 0] = np.nan
    np.savez_compressed(
        sequence_path,
        sample_id=np.asarray(["good", "bad"], dtype=object),
        event_id=np.asarray(["event-1", "event-2"], dtype=object),
        subject_id=np.asarray(["sub-01", "sub-02"], dtype=object),
        labels=np.asarray([json.dumps({"fatigue": 2.0}), json.dumps({"fatigue": 3.0})], dtype=object),
        frame_embeddings=frame_embeddings,
    )

    summary = build_video_temporal_embeddings(
        frame_sequences=sequence_path,
        out_path=out_path,
        temporal_encoder="tcn",
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["mask_sum"] == 1
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0]]
    assert np.any(loaded["face_emb"][0] != 0)
    assert np.all(loaded["face_emb"][1] == 0)
    quality = json.loads(loaded["quality_flags"][1])
    assert quality["masked_reason"] == "nonfinite_frame_embeddings"


def test_video_temporal_embeddings_respect_source_video_mask(tmp_path):
    sequence_path = tmp_path / "masked_frames.npz"
    out_path = tmp_path / "v4b_tcn.npz"
    frame_embeddings = np.ones((2, 16, 8), dtype=np.float32)
    np.savez_compressed(
        sequence_path,
        sample_id=np.asarray(["usable", "masked"], dtype=object),
        event_id=np.asarray(["event-1", "event-2"], dtype=object),
        subject_id=np.asarray(["sub-01", "sub-02"], dtype=object),
        labels=np.asarray([json.dumps({"fatigue": 2.0}), json.dumps({"fatigue": 3.0})], dtype=object),
        frame_embeddings=frame_embeddings,
        modality_mask=np.asarray([[0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int8),
    )

    summary = build_video_temporal_embeddings(
        frame_sequences=sequence_path,
        out_path=out_path,
        temporal_encoder="tcn",
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert summary["mask_sum"] == 1
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0]]
    assert np.any(loaded["face_emb"][0] != 0)
    assert np.all(loaded["face_emb"][1] == 0)
    quality = json.loads(loaded["quality_flags"][1])
    assert quality["masked_reason"] == "source_video_mask_zero"
