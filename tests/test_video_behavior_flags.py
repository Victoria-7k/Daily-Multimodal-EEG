from __future__ import annotations

import json
import math

import numpy as np

from daily_multimodal.embeddings.video_behavior_flags import (
    BEHAVIOR_RATIO_NAMES,
    audit_behavior_flags,
    aggregate_behavior_window,
    extract_behavior_flags,
    frame_flags_from_mediapipe_rows,
    frame_flags_from_openface_rows,
    merge_openface_and_mediapipe_flags,
    load_jsonl,
    write_json,
    write_jsonl,
)


def test_pose_thresholds_use_radians():
    rows = [{"pose_Rx": math.radians(21), "pose_Ry": math.radians(31), "confidence": 0.9, "success": 1}]
    flags = frame_flags_from_openface_rows(rows)
    assert flags[0]["head_down"] is True
    assert flags[0]["side_turn"] is True


def test_behavior_rows_do_not_require_face_presence_filter():
    window = {"sample_id": "s1", "event_id": "e1", "subject_id": "sub-01"}
    row = aggregate_behavior_window(window, frame_flags=[{"offscreen": True}] * 20, source={})
    assert row["offscreen_ratio"] == 1.0
    assert row["sampled_frame_count"] == 20


def test_unknown_mediapipe_values_do_not_make_frame_offscreen():
    rows = [
        {
            "success": 0,
            "confidence": 0.95,
            "mediapipe_face_landmarks_present": None,
            "mediapipe_pose_present": None,
        }
    ]

    flags = frame_flags_from_openface_rows(rows)

    assert flags[0]["low_confidence"] is False
    assert flags[0]["offscreen"] is False


def test_behavior_ratios_use_only_usable_frames():
    window = {"sample_id": "s3", "event_id": "e3", "subject_id": "sub-03"}
    flags = [
        {"face_visible": True, "usable": True},
        {"face_visible": False, "usable": False},
    ]

    row = aggregate_behavior_window(window, flags, source={})

    assert row["face_visible_ratio"] == 1.0
    assert row["sampled_frame_count"] == 2
    assert row["usable_frame_count"] == 1


def test_behavior_ratios_are_zero_when_no_frames_are_usable():
    window = {"sample_id": "s4", "event_id": "e4", "subject_id": "sub-04"}
    flags = [{"face_visible": True, "usable": False}]

    row = aggregate_behavior_window(window, flags, source={})

    assert row["face_visible_ratio"] == 0.0
    assert row["sampled_frame_count"] == 1
    assert row["usable_frame_count"] == 0


def test_aggregate_behavior_window_includes_schema_and_ratios():
    window = {
        "sample_id": "s2",
        "event_id": "e2",
        "subject_id": "sub-02",
    }
    source = {
        "source_mp4_path": "clip.mp4",
        "clip_start_seconds": 12.0,
        "clip_end_seconds": 22.0,
    }
    flags = [
        {"face_visible": True, "head_down": True},
        {"face_visible": False, "low_confidence": True},
    ]

    row = aggregate_behavior_window(window, flags, source, detectors={"face": "openface"})

    assert row["sample_id"] == "s2"
    assert row["source_mp4_path"] == "clip.mp4"
    assert row["clip_start_seconds"] == 12.0
    assert row["clip_end_seconds"] == 22.0
    assert row["sampled_frame_count"] == 2
    assert row["usable_frame_count"] == 2
    assert row["face_visible_ratio"] == 0.5
    assert row["head_down_ratio"] == 0.5
    assert row["low_confidence_ratio"] == 0.5
    assert row["detectors"] == {"face": "openface"}
    for ratio_name in BEHAVIOR_RATIO_NAMES:
        assert ratio_name in row


def test_openface_rows_set_visibility_confidence_and_offscreen_flags():
    rows = [
        {
            "success": 1,
            "confidence": 0.79,
            "face_bbox": [10, 10, 20, 20],
            "person_bbox": [0, 0, 100, 200],
        },
        {
            "success": 0,
            "confidence": 0.95,
            "mediapipe_face_landmarks_present": False,
            "mediapipe_pose_present": False,
        },
    ]

    flags = frame_flags_from_openface_rows(rows)

    assert flags[0]["face_visible"] is True
    assert flags[0]["low_confidence"] is True
    assert flags[0]["offscreen"] is False
    assert flags[1]["face_visible"] is False
    assert flags[1]["offscreen"] is True


def test_geometry_rules_for_hand_occlusion_near_face_and_motion():
    rows = [
        {
            "success": 1,
            "confidence": 0.9,
            "face_bbox": [10, 10, 30, 30],
            "hand_bbox": [15, 15, 25, 25],
            "hand_landmarks": [[16, 16], [35, 35], [20, 20], [24, 24]],
            "person_bbox": [0, 0, 50, 100],
        },
        {
            "success": 1,
            "confidence": 0.9,
            "face_bbox": [10, 10, 30, 30],
            "hand_bbox": [80, 80, 90, 90],
            "person_bbox": [0, 20, 50, 120],
        },
    ]

    flags = frame_flags_from_openface_rows(rows)

    assert flags[0]["hand_near_face"] is True
    assert flags[0]["hand_occlusion"] is True
    assert flags[0]["large_motion"] is False
    assert flags[1]["hand_near_face"] is False
    assert flags[1]["large_motion"] is True


def test_mediapipe_rows_add_hand_person_offscreen_and_motion_flags():
    rows = [
        {
            "mediapipe_face_landmarks_present": True,
            "mediapipe_pose_landmarks_present": True,
            "mediapipe_left_hand_landmarks_present": True,
            "face_bbox": [40, 40, 80, 80],
            "hand_bbox": [55, 55, 75, 75],
            "hand_landmarks": [[60, 60], [65, 65], [70, 70], [90, 90]],
            "person_bbox": [20, 20, 120, 220],
            "pose_visible_landmark_count": 18,
        },
        {
            "mediapipe_face_landmarks_present": False,
            "mediapipe_pose_landmarks_present": True,
            "mediapipe_left_hand_landmarks_present": False,
            "mediapipe_right_hand_landmarks_present": False,
            "person_bbox": [20, 60, 120, 260],
            "pose_visible_landmark_count": 17,
        },
        {
            "mediapipe_face_landmarks_present": False,
            "mediapipe_pose_landmarks_present": False,
            "mediapipe_left_hand_landmarks_present": False,
            "mediapipe_right_hand_landmarks_present": False,
            "pose_visible_landmark_count": 0,
        },
    ]

    flags = frame_flags_from_mediapipe_rows(rows)
    row = aggregate_behavior_window(
        {"sample_id": "s-mediapipe"},
        flags,
        source={},
        detectors={"behavior_backend": "mediapipe_holistic_v1"},
    )

    assert flags[0]["person_visible"] is True
    assert flags[0]["hand_visible"] is True
    assert flags[0]["hand_near_face"] is True
    assert flags[0]["hand_occlusion"] is True
    assert flags[1]["person_visible"] is True
    assert flags[1]["large_motion"] is True
    assert flags[1]["offscreen"] is False
    assert flags[2]["offscreen"] is True
    assert row["person_visible_ratio"] == 2 / 3
    assert row["hand_visible_ratio"] == 1 / 3
    assert row["offscreen_ratio"] == 1 / 3
    assert row["behavior_backend"] == "mediapipe_holistic_v1"


def test_openface_head_pose_overrides_mediapipe_when_flags_are_merged():
    openface_flags = [
        {
            "face_visible": True,
            "low_confidence": False,
            "head_down": True,
            "side_turn": False,
        }
    ]
    mediapipe_flags = [
        {
            "face_visible": False,
            "low_confidence": True,
            "head_down": False,
            "side_turn": True,
            "hand_visible": True,
            "person_visible": True,
            "hand_near_face": True,
            "hand_occlusion": True,
            "large_motion": False,
            "offscreen": False,
        }
    ]

    merged = merge_openface_and_mediapipe_flags(openface_flags, mediapipe_flags)

    assert merged[0]["face_visible"] is True
    assert merged[0]["low_confidence"] is False
    assert merged[0]["head_down"] is True
    assert merged[0]["side_turn"] is False
    assert merged[0]["hand_visible"] is True
    assert merged[0]["person_visible"] is True
    assert merged[0]["hand_near_face"] is True
    assert merged[0]["hand_occlusion"] is True


def test_audit_behavior_flags_summarizes_ratios_and_review_sets(tmp_path):
    flags_path = tmp_path / "flags.jsonl"
    rows = [_behavior_row(f"sample-{idx}", head_down_ratio=value) for idx, value in enumerate([0.0, 0.5, 0.5, 1.0])]
    write_jsonl(rows, flags_path)

    report = audit_behavior_flags(flags_path, top_k=2, random_seed=7)

    assert report["window_count"] == 4
    assert report["success_count"] == 4
    assert report["missing_count"] == 0
    assert "face_visible_ratio" in report["ratios"]
    assert report["ratios"]["head_down_ratio"]["mean"] == 0.5
    assert report["ratios"]["head_down_ratio"]["median"] == 0.5
    assert len(report["review_sets"]["top_head_down_ratio"]) == 2
    assert set(report["review_sets"]) == {
        "top_head_down_ratio",
        "top_hand_occlusion_ratio",
        "top_offscreen_ratio",
        "top_large_motion_ratio",
        "random_windows",
    }
    review_row = report["review_sets"]["top_head_down_ratio"][0]
    assert review_row["sample_id"] == "sample-3"
    assert review_row["source_mp4_path"] == "sample-3.mp4"
    for ratio_name in BEHAVIOR_RATIO_NAMES:
        assert ratio_name in review_row


def test_audit_behavior_flags_joins_openface_mask_and_quality_by_sample_id(tmp_path):
    flags_path = tmp_path / "flags.jsonl"
    openface_path = tmp_path / "openface.npz"
    write_jsonl([_behavior_row("sample-a"), _behavior_row("sample-b")], flags_path)
    np.savez_compressed(
        openface_path,
        sample_id=np.array(["sample-b"], dtype=object),
        modality_mask=np.array([[0, 0, 1, 0]], dtype=np.int8),
        quality_flags=np.array([json.dumps({"masked": False, "face_detection_success_rate": 0.9})], dtype=object),
    )

    report = audit_behavior_flags(flags_path, openface_embeddings=openface_path, top_k=2)

    joined = {
        row["sample_id"]: row
        for rows in report["review_sets"].values()
        for row in rows
    }
    assert joined["sample-b"]["openface_mask_value"] == 1
    assert joined["sample-b"]["openface_quality_flags"]["face_detection_success_rate"] == 0.9


def test_audit_json_writer_normalizes_non_finite_openface_quality_flags(tmp_path):
    flags_path = tmp_path / "flags.jsonl"
    openface_path = tmp_path / "openface.npz"
    out_json = tmp_path / "audit.json"
    write_jsonl([_behavior_row("sample-a")], flags_path)
    np.savez_compressed(
        openface_path,
        sample_id=np.array(["sample-a"], dtype=object),
        modality_mask=np.array([[0, 0, 1, 0]], dtype=np.int8),
        quality_flags=np.array(
            [json.dumps({"score": float("nan"), "overflow": float("inf")})],
            dtype=object,
        ),
    )

    report = audit_behavior_flags(flags_path, openface_embeddings=openface_path, top_k=1)
    write_json(report, out_json)

    text = out_json.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    loaded = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(AssertionError(value)),
    )
    row = loaded["review_sets"]["top_head_down_ratio"][0]
    assert row["openface_quality_flags"]["score"] is None
    assert row["openface_quality_flags"]["overflow"] is None


def test_extract_behavior_flags_reads_openface_cache_rows(tmp_path):
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "openface" / "sample-1" / "openface_temporal_v1"
    cache_dir.mkdir(parents=True)
    csv_path = cache_dir / "openface.csv"
    csv_path.write_text(
        "\n".join(
            [
                "frame, timestamp, confidence, success, pose_Rx, pose_Ry",
                f"1, 0.0, 0.95, 1, {math.radians(21)}, 0",
                f"2, 0.5, 0.70, 1, 0, {math.radians(31)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cache_dir / "openface_target.json").write_text(
        json.dumps(
            {
                "sample_id": "sample-1",
                "source_path": "/mnt/dataset1/sitian/video/example.MP4",
                "target_csv_path": str(csv_path),
            }
        ),
        encoding="utf-8",
    )
    window = {
        "sample_id": "sample-1",
        "event_id": "event-1",
        "subject_id": "sub-01",
        "video_candidates": [
            {
                "mp4_path": "/mnt/dataset1/sitian/video/example.MP4",
                "clip_start_seconds": 3.0,
                "clip_end_seconds": 8.0,
            }
        ],
    }
    out = tmp_path / "flags.jsonl"
    failures = tmp_path / "failures.json"

    summary = extract_behavior_flags(
        [window],
        out=out,
        failures_out=failures,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
    )

    assert summary == {"selected_count": 1, "written_count": 1, "failure_count": 0}
    rows = load_jsonl(out)
    assert rows[0]["sample_id"] == "sample-1"
    assert rows[0]["source_mp4_path"] == "/mnt/dataset1/sitian/video/example.MP4"
    assert rows[0]["clip_start_seconds"] == 3.0
    assert rows[0]["clip_end_seconds"] == 8.0
    assert rows[0]["sampled_frame_count"] == 2
    assert rows[0]["face_visible_ratio"] == 1.0
    assert rows[0]["low_confidence_ratio"] == 0.5
    assert rows[0]["head_down_ratio"] == 0.5
    assert rows[0]["side_turn_ratio"] == 0.5
    assert rows[0]["detectors"]["face"] == "openface_csv_cache"
    assert json.loads(failures.read_text(encoding="utf-8")) == []


def test_extract_behavior_flags_writes_mediapipe_backend_rows_with_openface_pose_priority(tmp_path):
    cache_root = tmp_path / "cache"
    cache_dir = cache_root / "openface" / "sample-1" / "openface_temporal_v1"
    cache_dir.mkdir(parents=True)
    csv_path = cache_dir / "openface.csv"
    csv_path.write_text(
        "\n".join(
            [
                "frame, confidence, success, pose_Rx, pose_Ry",
                f"1, 0.95, 1, {math.radians(21)}, 0",
                f"2, 0.95, 1, 0, {math.radians(31)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cache_dir / "openface_target.json").write_text(
        json.dumps({"source_path": "/videos/example.mp4", "target_csv_path": str(csv_path)}),
        encoding="utf-8",
    )
    window = {
        "sample_id": "sample-1",
        "event_id": "event-1",
        "subject_id": "sub-01",
        "video_candidates": [
            {"mp4_path": "/videos/example.mp4", "clip_start_seconds": 2.0, "clip_end_seconds": 12.0}
        ],
    }
    mediapipe_rows = [
        {
            "mediapipe_face_landmarks_present": True,
            "mediapipe_pose_landmarks_present": True,
            "mediapipe_left_hand_landmarks_present": True,
            "face_bbox": [40, 40, 80, 80],
            "hand_bbox": [55, 55, 75, 75],
            "person_bbox": [20, 20, 120, 220],
            "pose_visible_landmark_count": 18,
        },
        {
            "mediapipe_face_landmarks_present": False,
            "mediapipe_pose_landmarks_present": False,
            "mediapipe_left_hand_landmarks_present": False,
            "mediapipe_right_hand_landmarks_present": False,
            "pose_visible_landmark_count": 0,
        },
    ]
    out = tmp_path / "mediapipe_holistic_behavior_flags.jsonl"
    failures = tmp_path / "failures.json"

    summary = extract_behavior_flags(
        [window],
        out=out,
        failures_out=failures,
        openface_cache_root=cache_root,
        openface_encoder_profile="openface_temporal_v1",
        behavior_backend="mediapipe_holistic_v1",
        mediapipe_frame_rows_by_sample={"sample-1": mediapipe_rows},
    )

    assert summary == {"selected_count": 1, "written_count": 1, "failure_count": 0}
    row = load_jsonl(out)[0]
    assert row["behavior_backend"] == "mediapipe_holistic_v1"
    assert row["head_down_ratio"] == 0.5
    assert row["side_turn_ratio"] == 0.5
    assert row["hand_visible_ratio"] == 0.5
    assert row["person_visible_ratio"] == 0.5
    assert row["hand_near_face_ratio"] == 0.5
    assert row["offscreen_ratio"] == 0.5
    assert json.loads(failures.read_text(encoding="utf-8")) == []


def test_extract_behavior_flags_streams_rows_and_progress_for_monitoring(tmp_path):
    windows = [
        {"sample_id": "sample-1", "event_id": "event-1", "subject_id": "sub-01"},
        {"sample_id": "sample-2", "event_id": "event-2", "subject_id": "sub-01"},
    ]
    frame_rows = {
        "sample-1": [{"mediapipe_face_landmarks_present": False, "mediapipe_pose_landmarks_present": False}],
        "sample-2": [{"mediapipe_face_landmarks_present": True, "mediapipe_pose_landmarks_present": False}],
    }
    out = tmp_path / "flags.jsonl"
    failures = tmp_path / "failures.json"
    progress = tmp_path / "progress.log"

    summary = extract_behavior_flags(
        windows,
        out=out,
        failures_out=failures,
        behavior_backend="mediapipe_holistic_v1",
        mediapipe_frame_rows_by_sample=frame_rows,
        progress_out=progress,
        progress_every=1,
    )

    assert summary == {"selected_count": 2, "written_count": 2, "failure_count": 0}
    assert len(load_jsonl(out)) == 2
    assert json.loads(failures.read_text(encoding="utf-8")) == []
    progress_text = progress.read_text(encoding="utf-8")
    assert "start 1/2" in progress_text
    assert "done 2/2" in progress_text
    assert "sample-2" in progress_text
    assert "[" in progress_text and "]" in progress_text


def _behavior_row(sample_id: str, **ratios: float) -> dict:
    row = {
        "sample_id": sample_id,
        "event_id": sample_id.replace("sample", "event"),
        "subject_id": "sub-01",
        "source_mp4_path": f"{sample_id}.mp4",
        "clip_start_seconds": 12.0,
        "clip_end_seconds": 22.0,
    }
    for ratio_name in BEHAVIOR_RATIO_NAMES:
        row[ratio_name] = ratios.get(ratio_name, 0.0)
    return row
