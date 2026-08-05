from __future__ import annotations

from pathlib import Path

import numpy as np

from daily_multimodal.training.video_transfer_matrix import analyze_cross_subject_transfer


def test_analyze_cross_subject_transfer_reports_pairwise_centered_mapping(tmp_path: Path):
    repr_path = tmp_path / "repr.npz"
    _write_transfer_repr(repr_path)

    result = analyze_cross_subject_transfer(
        representations=repr_path,
        variant="B1",
        out_json=tmp_path / "transfer.json",
        out_table=tmp_path / "transfer.md",
        ridge_alpha=0.01,
    )

    matrix = result["matrix"]
    assert matrix["sub-01"]["sub-01"]["protocol"] == "within_subject_session_leave_out"
    assert matrix["sub-01"]["sub-01"]["train_test_overlap"] == 0
    assert matrix["sub-02"]["sub-02"]["train_test_overlap"] == 0
    assert matrix["sub-01"]["sub-01"]["pearson_r"] > 0.9
    assert matrix["sub-02"]["sub-02"]["pearson_r"] > 0.9
    assert matrix["sub-01"]["sub-02"]["protocol"] == "cross_subject_all_to_all"
    assert matrix["sub-01"]["sub-02"]["pearson_r"] < -0.9
    assert result["sign_summary"]["positive_pairs"] == 2
    assert result["sign_summary"]["negative_pairs"] == 2


def _write_transfer_repr(path: Path) -> None:
    sample_ids = []
    subject_ids = []
    event_ids = []
    session_ids = []
    targets = []
    rows = []
    for subject, sign, baseline in [("sub-01", 1.0, 5.0), ("sub-02", -1.0, 6.0)]:
        for index, value in enumerate([-2.0, -1.0, 1.0, 2.0, -1.5, 1.5]):
            session = "ses-01" if index < 3 else "ses-02"
            sample_ids.append(f"{subject}_{session}_row-{index:04d}_win-0000")
            subject_ids.append(subject)
            event_ids.append(f"{subject}_{session}_row-{index:04d}")
            session_ids.append(f"{subject}_{session}")
            targets.append(baseline + value)
            vector = np.zeros(64, dtype=np.float32)
            vector[0] = sign * value
            rows.append(vector)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        subject_id=np.asarray(subject_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        session_id=np.asarray(session_ids, dtype=object),
        target=np.asarray(targets, dtype=np.float32),
        repr__B1=np.stack(rows).astype(np.float32),
    )
