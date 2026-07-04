from __future__ import annotations

import json

import numpy as np

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
