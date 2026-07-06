from __future__ import annotations

import json

import numpy as np
import pytest

import daily_multimodal.training.video_embedding_probes as video_embedding_probes
from daily_multimodal.training.video_embedding_probes import run_video_embedding_probes


def test_run_video_embedding_probes_reports_subject_session_and_fatigue_metrics(tmp_path):
    path = tmp_path / "probe.npz"
    out_json = tmp_path / "probes.json"
    out_table = tmp_path / "probes.md"
    sample_ids = []
    subject_ids = []
    event_ids = []
    labels = []
    embeddings = []
    masks = []
    for subject_index, subject in enumerate(["sub-a", "sub-b", "sub-c"]):
        for session_index in range(3):
            for row in range(4):
                sample_ids.append(f"{subject}_ses-{session_index:02d}_win-{row:04d}")
                subject_ids.append(subject)
                event_ids.append(f"{subject}_ses-{session_index:02d}_row-{row:04d}")
                labels.append(json.dumps({"fatigue": float(subject_index + session_index)}))
                vector = np.zeros(256, dtype=np.float32)
                vector[subject_index] = 3.0
                vector[10 + session_index] = 2.0
                vector[20] = float(subject_index + session_index)
                embeddings.append(vector)
                masks.append([0, 0, 1, 0])
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        subject_id=np.asarray(subject_ids, dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=np.stack(embeddings),
        modality_mask=np.asarray(masks, dtype=np.int8),
    )

    result = run_video_embedding_probes(
        embeddings=path,
        target_label="fatigue",
        out_json=out_json,
        out_table=out_table,
        seed=3,
    )

    assert result["row_count"] == 36
    assert result["probes"]["P1_subject_logreg"]["accuracy_mean"] is not None
    assert result["probes"]["P2_within_subject_session_logreg"]["subject_count"] == 3
    assert result["probes"]["P3_fatigue_ridge"]["rmse_mean"] is not None
    loaded = json.loads(out_json.read_text(encoding="utf-8"))
    assert loaded["target_label"] == "fatigue"
    assert "P1_subject_logreg" in out_table.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("fold_strategy", "expected_min_folds"),
    [
        ("leave_one_subject_out", 3),
        ("within_subject_event_split", 3),
        ("within_subject_session_leave_out", 3),
        ("within_subject_chronological_split", 1),
    ],
)
def test_p3_fatigue_ridge_can_use_video_fold_strategy(tmp_path, fold_strategy, expected_min_folds):
    path = tmp_path / "probe.npz"
    out_json = tmp_path / "probes.json"
    out_table = tmp_path / "probes.md"
    sample_ids = []
    subject_ids = []
    event_ids = []
    labels = []
    embeddings = []
    masks = []
    for subject_index, subject in enumerate(["sub-a", "sub-b", "sub-c"]):
        for session_index in range(3):
            for row in range(3):
                sample_ids.append(f"{subject}_ses-{session_index:02d}_win-{row:04d}")
                subject_ids.append(subject)
                event_ids.append(f"{subject}_ses-{session_index:02d}_row-{row:04d}")
                labels.append(json.dumps({"fatigue": float(subject_index + session_index + row * 0.1)}))
                vector = np.zeros(256, dtype=np.float32)
                vector[subject_index] = 1.0
                vector[10 + session_index] = 1.0
                vector[20] = float(subject_index + session_index + row * 0.1)
                embeddings.append(vector)
                masks.append([0, 0, 1, 0])
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        subject_id=np.asarray(subject_ids, dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=np.stack(embeddings),
        modality_mask=np.asarray(masks, dtype=np.int8),
    )

    result = run_video_embedding_probes(
        embeddings=path,
        target_label="fatigue",
        out_json=out_json,
        out_table=out_table,
        seed=3,
        fold_strategy=fold_strategy,
    )

    p3 = result["probes"]["P3_fatigue_ridge"]
    assert "failure" not in p3
    assert p3["fold_strategy"] == fold_strategy
    assert p3["fold_count"] >= expected_min_folds


def test_probe_train_embeddings_are_used_only_for_training_splits(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.npz"
    train_path = tmp_path / "train_aug.npz"
    sample_ids = [f"{subject}{row}" for subject in ("a", "b", "c") for row in range(4)]
    _write_probe_npz(eval_path, sample_ids, offset=0.0)
    _write_probe_npz(train_path, sample_ids, offset=1000.0)
    captured = []

    def fake_ridge_predict(train_x, train_y, test_x, *, alpha):
        del train_y, alpha
        captured.append((np.asarray(train_x).copy(), np.asarray(test_x).copy()))
        return np.zeros(len(test_x), dtype=np.float32)

    monkeypatch.setattr(video_embedding_probes, "_ridge_predict", fake_ridge_predict)

    result = run_video_embedding_probes(
        embeddings=eval_path,
        train_embeddings=train_path,
        target_label="fatigue",
        out_json=tmp_path / "probes.json",
        out_table=tmp_path / "probes.md",
        seed=3,
        fold_strategy="leave_one_subject_out",
    )

    assert result["train_embeddings"] == str(train_path)
    assert captured
    assert all(float(train_x[:, 0].min()) >= 1000.0 for train_x, _test_x in captured)
    assert all(float(test_x[:, 0].max()) < 1000.0 for _train_x, test_x in captured)


def _write_probe_npz(path, sample_ids, *, offset):
    labels = [json.dumps({"fatigue": float(index % 3)}) for index, _sample_id in enumerate(sample_ids)]
    face_emb = np.zeros((len(sample_ids), 256), dtype=np.float32)
    for index in range(len(sample_ids)):
        face_emb[index, 0] = float(index) + offset
        face_emb[index, 1] = float(index % 2)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray([f"sub-{sample_id[0]}_ses-00_row-{index:04d}" for index, sample_id in enumerate(sample_ids)], dtype=object),
        subject_id=np.asarray([f"sub-{sample_id[0]}" for sample_id in sample_ids], dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=face_emb,
        modality_mask=np.asarray([[0, 0, 1, 0]] * len(sample_ids), dtype=np.int8),
    )
