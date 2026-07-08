from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from daily_multimodal.training.video_loso_diagnostics import (
    add_subject_centered_label,
    analyze_loso_failure,
)


def test_analyze_loso_failure_reports_subject_labels_and_prediction_groups(tmp_path: Path):
    repr_path = tmp_path / "repr.npz"
    report_path = tmp_path / "loso.json"
    _write_representations(repr_path)
    _write_fold_report(report_path)

    result = analyze_loso_failure(
        representations=repr_path,
        fold_report=report_path,
        variant="B1",
        out_json=tmp_path / "diagnostics.json",
        out_table=tmp_path / "diagnostics.md",
    )

    by_subject = {row["subject_id"]: row for row in result["subject_label_distribution"]}
    assert by_subject["sub-01"]["count"] == 4
    assert by_subject["sub-01"]["event_count"] == 2
    assert by_subject["sub-01"]["session_count"] == 1
    assert by_subject["sub-01"]["target_range"] == 2.0
    assert result["prediction_group_summary"]["positive"]["subject_count"] == 1
    assert result["prediction_group_summary"]["negative"]["subject_count"] == 1
    assert result["prediction_group_summary"]["negative"]["subjects"] == ["sub-02"]
    assert "subject-dependent" in result["interpretation"]


def test_add_subject_centered_label_preserves_bundle_and_adds_label(tmp_path: Path):
    in_path = tmp_path / "video.npz"
    out_path = tmp_path / "video_centered.npz"
    _write_video_bundle(in_path)

    result = add_subject_centered_label(
        embeddings=in_path,
        out=out_path,
        target_label="fatigue",
        centered_label="fatigue_subject_centered",
    )

    assert result["row_count"] == 4
    assert result["subject_means"] == {"sub-01": 3.0, "sub-02": 6.0}
    with np.load(out_path, allow_pickle=True) as loaded:
        assert loaded["face_emb"].shape == (4, 256)
        labels = [json.loads(str(value)) for value in loaded["labels"]]
        assert labels[0]["fatigue"] == 2.0
        assert labels[0]["fatigue_subject_centered"] == -1.0
        assert labels[1]["fatigue_subject_centered"] == 1.0
        assert labels[2]["fatigue_subject_centered"] == -1.0
        assert labels[3]["fatigue_subject_centered"] == 1.0


def _write_representations(path: Path) -> None:
    np.savez_compressed(
        path,
        sample_id=np.asarray(["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"], dtype=object),
        subject_id=np.asarray(["sub-01"] * 4 + ["sub-02"] * 4, dtype=object),
        event_id=np.asarray(["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4"], dtype=object),
        session_id=np.asarray(["sub-01_ses-01"] * 4 + ["sub-02_ses-01"] * 4, dtype=object),
        target=np.asarray([2.0, 2.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0], dtype=np.float32),
        repr__B1=np.zeros((8, 64), dtype=np.float32),
    )


def _write_fold_report(path: Path) -> None:
    payload = {
        "experiments": {
            "B1": {
                "folds": [
                    {
                        "test_sample_ids": ["sub-01_s1", "sub-01_s2", "sub-02_s3", "sub-02_s4"],
                        "test_predictions": [1.0, 3.0, 9.0, 7.0],
                        "test_targets": [1.0, 3.0, 7.0, 9.0],
                    }
                ]
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_video_bundle(path: Path) -> None:
    labels = [json.dumps({"fatigue": value}) for value in [2.0, 4.0, 5.0, 7.0]]
    mask = np.zeros((4, 4), dtype=np.int8)
    mask[:, 2] = 1
    np.savez_compressed(
        path,
        sample_id=np.asarray(["a", "b", "c", "d"], dtype=object),
        event_id=np.asarray(["e1", "e2", "e3", "e4"], dtype=object),
        subject_id=np.asarray(["sub-01", "sub-01", "sub-02", "sub-02"], dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=np.zeros((4, 256), dtype=np.float32),
        modality_mask=mask,
    )
