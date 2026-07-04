import json

import numpy as np

from daily_multimodal.embeddings.video_variants import build_video_variant_embeddings


def test_openface_behavior_flags_v2_preserves_face_contract(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    behavior_flags = tmp_path / "behavior_flags.jsonl"
    out_path = tmp_path / "face_openface_behavior_flags_v2_embeddings.npz"

    windows = [
        {"sample_id": "sample-1", "event_id": "event-1", "subject_id": "sub-01"},
        {"sample_id": "sample-2", "event_id": "event-2", "subject_id": "sub-02"},
    ]
    window_index.write_text(
        "".join(json.dumps(row) + "\n" for row in windows),
        encoding="utf-8",
    )

    flag_row = {
        "sampled_frame_count": 20,
        "usable_frame_count": 20,
        "face_visible_ratio": 1.0,
        "low_confidence_ratio": 0.0,
        "head_down_ratio": 0.1,
        "side_turn_ratio": 0.2,
        "hand_near_face_ratio": 0.3,
        "hand_occlusion_ratio": 0.4,
        "large_motion_ratio": 0.5,
        "offscreen_ratio": 0.0,
    }
    behavior_flags.write_text(
        "".join(
            json.dumps({**flag_row, "sample_id": row["sample_id"]}) + "\n"
            for row in windows
        ),
        encoding="utf-8",
    )

    build_video_variant_embeddings(
        variant="openface_behavior_flags_v2",
        window_index_path=window_index,
        behavior_flags_path=behavior_flags,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert "face_emb" in loaded.files
    assert "video_emb" not in loaded.files
    assert "video_emb_v2" not in loaded.files
    assert loaded["face_emb"].shape == (2, 256)
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 1, 0]]
    assert set(loaded["encoder_version"].astype(str)) == {
        "openface_behavior_flags_v2"
    }


def test_behavior_flags_only_probe_masks_missing_or_unusable_rows(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    behavior_flags = tmp_path / "behavior_flags.jsonl"
    out_path = tmp_path / "face_behavior_flags_only_v2_probe_embeddings.npz"
    _write_jsonl(
        window_index,
        [
            {"sample_id": "usable", "event_id": "event-1", "subject_id": "sub-01"},
            {"sample_id": "unusable", "event_id": "event-2", "subject_id": "sub-02"},
            {"sample_id": "missing", "event_id": "event-3", "subject_id": "sub-03"},
        ],
    )
    _write_jsonl(
        behavior_flags,
        [
            _behavior_row("usable", usable_frame_count=20, head_down_ratio=0.25),
            _behavior_row("unusable", usable_frame_count=0, head_down_ratio=0.75),
        ],
    )

    build_video_variant_embeddings(
        variant="behavior_flags_only_v2_probe",
        window_index_path=window_index,
        behavior_flags_path=behavior_flags,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["sample_id"].astype(str).tolist() == ["usable", "unusable", "missing"]
    assert loaded["face_emb"].shape == (3, 256)
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    assert set(loaded["encoder_version"].astype(str)) == {"behavior_flags_only_v2_probe"}
    assert np.any(loaded["face_emb"][0] != 0)
    assert np.all(loaded["face_emb"][1] == 0)
    assert np.all(loaded["face_emb"][2] == 0)


def test_openface_behavior_flags_v2_strict_requires_openface_mask(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    behavior_flags = tmp_path / "behavior_flags.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "strict.npz"
    _write_jsonl(
        window_index,
        [
            {"sample_id": "good", "event_id": "event-1", "subject_id": "sub-01"},
            {"sample_id": "masked", "event_id": "event-2", "subject_id": "sub-02"},
            {"sample_id": "missing-openface", "event_id": "event-3", "subject_id": "sub-03"},
        ],
    )
    _write_jsonl(
        behavior_flags,
        [
            _behavior_row("good", usable_frame_count=20),
            _behavior_row("masked", usable_frame_count=20),
            _behavior_row("missing-openface", usable_frame_count=20),
        ],
    )
    _write_openface_npz(openface_path, ["good", "masked"], masks=[1, 0])

    build_video_variant_embeddings(
        variant="openface_behavior_flags_v2",
        window_index_path=window_index,
        behavior_flags_path=behavior_flags,
        openface_embeddings_path=openface_path,
        sample_mode="strict_aligned",
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["sample_id"].astype(str).tolist() == ["good", "masked", "missing-openface"]
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


def test_openface_behavior_flags_v2_behavior_retained_keeps_low_quality_openface(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    behavior_flags = tmp_path / "behavior_flags.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "retained.npz"
    _write_jsonl(
        window_index,
        [{"sample_id": "masked", "event_id": "event-2", "subject_id": "sub-02"}],
    )
    _write_jsonl(behavior_flags, [_behavior_row("masked", usable_frame_count=20)])
    _write_openface_npz(openface_path, ["masked"], masks=[0])

    build_video_variant_embeddings(
        variant="openface_behavior_flags_v2",
        window_index_path=window_index,
        behavior_flags_path=behavior_flags,
        openface_embeddings_path=openface_path,
        sample_mode="behavior_retained",
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0]]
    assert quality["openface_mask_value"] == 0
    assert quality["behavior_retained_without_openface_mask"] is True


def test_openface_only_v1_rewrites_encoder_version_and_preserves_mask(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "v1.npz"
    _write_jsonl(
        window_index,
        [
            {"sample_id": "good", "event_id": "event-1", "subject_id": "sub-01"},
            {"sample_id": "masked", "event_id": "event-2", "subject_id": "sub-02"},
        ],
    )
    _write_openface_npz(openface_path, ["good", "masked"], masks=[1, 0])

    build_video_variant_embeddings(
        variant="openface_only_v1",
        window_index_path=window_index,
        openface_embeddings_path=openface_path,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert loaded["face_emb"].shape == (2, 256)
    assert loaded["modality_mask"].tolist() == [[0, 0, 1, 0], [0, 0, 0, 0]]
    assert set(loaded["encoder_version"].astype(str)) == {"openface_only_v1"}


def test_variant_bundle_preserves_window_labels_for_ablation(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "v1.npz"
    _write_jsonl(
        window_index,
        [
            {
                "sample_id": "sample-1",
                "event_id": "event-1",
                "subject_id": "sub-01",
                "label_columns": {"fatigue": 2.0},
            }
        ],
    )
    _write_openface_npz(openface_path, ["sample-1"], masks=[1])

    build_video_variant_embeddings(
        variant="openface_only_v1",
        window_index_path=window_index,
        openface_embeddings_path=openface_path,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert "labels" in loaded.files
    assert json.loads(loaded["labels"][0]) == {"fatigue": 2.0}


def test_variant_bundle_uses_labels_when_label_columns_is_empty(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "v1.npz"
    _write_jsonl(
        window_index,
        [
            {
                "sample_id": "sample-1",
                "event_id": "event-1",
                "subject_id": "sub-01",
                "label_columns": {},
                "labels": {"fatigue": 3.0},
            }
        ],
    )
    _write_openface_npz(openface_path, ["sample-1"], masks=[1])

    build_video_variant_embeddings(
        variant="openface_only_v1",
        window_index_path=window_index,
        openface_embeddings_path=openface_path,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    assert json.loads(loaded["labels"][0]) == {"fatigue": 3.0}


def test_duplicate_behavior_sample_ids_are_rejected(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    behavior_flags = tmp_path / "behavior_flags.jsonl"
    out_path = tmp_path / "duplicate.npz"
    _write_jsonl(window_index, [{"sample_id": "sample-1"}])
    _write_jsonl(
        behavior_flags,
        [
            _behavior_row("sample-1"),
            _behavior_row("sample-1"),
        ],
    )

    try:
        build_video_variant_embeddings(
            variant="behavior_flags_only_v2_probe",
            window_index_path=window_index,
            behavior_flags_path=behavior_flags,
            out_path=out_path,
        )
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("duplicate behavior sample_id should fail")


def test_duplicate_openface_sample_ids_are_rejected(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "duplicate.npz"
    _write_jsonl(window_index, [{"sample_id": "sample-1"}])
    _write_openface_npz(openface_path, ["sample-1", "sample-1"], masks=[1, 1])

    try:
        build_video_variant_embeddings(
            variant="openface_only_v1",
            window_index_path=window_index,
            openface_embeddings_path=openface_path,
            out_path=out_path,
        )
    except ValueError as exc:
        assert "duplicate sample_id" in str(exc)
    else:
        raise AssertionError("duplicate openface sample_id should fail")


def test_openface_quality_nan_is_sanitized_in_variant_output(tmp_path):
    window_index = tmp_path / "window_index.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_path = tmp_path / "v1.npz"
    _write_jsonl(window_index, [{"sample_id": "sample-1"}])
    np.savez_compressed(
        openface_path,
        sample_id=np.array(["sample-1"], dtype=object),
        event_id=np.array(["event-1"], dtype=object),
        subject_id=np.array(["sub-01"], dtype=object),
        face_emb=np.ones((1, 256), dtype=np.float32),
        modality_mask=np.array([[0, 0, 1, 0]], dtype=np.int8),
        quality_flags=np.array(['{"mean_openface_confidence": NaN}'], dtype=object),
        encoder_version=np.array(["openface_source"], dtype=object),
    )

    build_video_variant_embeddings(
        variant="openface_only_v1",
        window_index_path=window_index,
        openface_embeddings_path=openface_path,
        out_path=out_path,
    )

    loaded = np.load(out_path, allow_pickle=True)
    quality = json.loads(loaded["quality_flags"][0])
    assert quality["openface_quality_flags"]["mean_openface_confidence"] is None


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _behavior_row(sample_id, *, usable_frame_count=20, head_down_ratio=0.1):
    return {
        "sample_id": sample_id,
        "sampled_frame_count": 20,
        "usable_frame_count": usable_frame_count,
        "face_visible_ratio": 1.0,
        "low_confidence_ratio": 0.0,
        "head_down_ratio": head_down_ratio,
        "side_turn_ratio": 0.2,
        "hand_near_face_ratio": 0.3,
        "hand_occlusion_ratio": 0.4,
        "large_motion_ratio": 0.5,
        "offscreen_ratio": 0.0,
    }


def _write_openface_npz(path, sample_ids, *, masks):
    row_count = len(sample_ids)
    face_emb = np.arange(row_count * 256, dtype=np.float32).reshape(row_count, 256)
    modality_mask = np.zeros((row_count, 4), dtype=np.int8)
    modality_mask[:, 2] = np.array(masks, dtype=np.int8)
    np.savez_compressed(
        path,
        sample_id=np.array(sample_ids, dtype=object),
        event_id=np.array([f"event-{idx}" for idx in range(row_count)], dtype=object),
        subject_id=np.array([f"sub-{idx:02d}" for idx in range(row_count)], dtype=object),
        face_emb=face_emb,
        modality_mask=modality_mask,
        quality_flags=np.array([json.dumps({"source": "openface"}) for _ in sample_ids], dtype=object),
        encoder_version=np.array(["openface_source"] * row_count, dtype=object),
    )
