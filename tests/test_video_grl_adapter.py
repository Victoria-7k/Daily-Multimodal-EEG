from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from daily_multimodal.training.video_grl_adapter import (
    build_default_grl_variant_specs,
    load_grl_adapter_dataset,
    run_video_grl_adapter_ablation,
)
from daily_multimodal.training.video_grl_adapter_analysis import (
    audit_grl_representations,
    summarize_grl_repeat_stability,
)


def test_default_grl_variant_specs_keep_b5_on_a1_with_conservative_lambdas():
    specs = build_default_grl_variant_specs(
        lambdas=[0.0, 0.001, 0.005, 0.01, 0.05],
        b5_lambdas=[0.001, 0.005, 0.01],
    )

    by_name = {spec.name: spec for spec in specs}

    assert by_name["B0"].use_adapter is False
    assert by_name["B1"].use_adapter is True
    assert by_name["B1"].grl_lambda == 0.0
    assert by_name["B2_lam0.005"].use_subject_grl is True
    assert by_name["B2_lam0.005"].use_session_grl is False
    assert by_name["B3_lam0.005"].use_subject_grl is False
    assert by_name["B3_lam0.005"].use_session_grl is True
    assert by_name["B4_lam0.005"].use_subject_grl is True
    assert by_name["B4_lam0.005"].use_session_grl is True
    assert by_name["B5_A1_lam0.005"].train_embedding_key == "A1"
    assert "B5_A2_lam0.005" not in by_name


def test_load_grl_adapter_dataset_aligns_a1_train_embeddings_by_sample_id(tmp_path):
    eval_path = tmp_path / "eval.npz"
    train_path = tmp_path / "train.npz"
    sample_ids = ["s0", "s1", "s2", "s3"]
    _write_video_npz(eval_path, sample_ids, offset=0.0)
    _write_video_npz(train_path, list(reversed(sample_ids)), offset=100.0)

    data = load_grl_adapter_dataset(
        eval_embeddings=eval_path,
        train_embeddings={"A1": train_path},
        target_label="fatigue",
    )

    assert data["sample_id"].tolist() == sample_ids
    assert data["face_emb"][0, 0] == 0.0
    assert data["train_face_emb_by_key"]["A1"][0, 0] == 100.0
    assert data["train_face_emb_by_key"]["A1"][3, 0] == 103.0


def test_run_video_grl_adapter_ablation_writes_metrics_and_table(tmp_path):
    pytest.importorskip("torch")
    eval_path = tmp_path / "eval.npz"
    train_path = tmp_path / "train.npz"
    sample_ids = [f"s{i}" for i in range(18)]
    _write_video_npz(eval_path, sample_ids, offset=0.0)
    _write_video_npz(train_path, sample_ids, offset=10.0)

    out_json = tmp_path / "metrics.json"
    out_table = tmp_path / "metrics.md"
    result = run_video_grl_adapter_ablation(
        eval_embeddings=eval_path,
        train_embeddings={"A1": train_path},
        target_label="fatigue",
        out_json=out_json,
        out_table=out_table,
        variants=["B0", "B1", "B2_lam0.001", "B5_A1_lam0.001"],
        fold_strategy="random_window_split",
        epochs=2,
        batch_size=8,
        adapter_dim=6,
        hidden_dim=5,
        learning_rate=0.01,
        seed=7,
        compute_domain_probes=True,
    )

    assert set(result["experiments"]) == {"B0", "B1", "B2_lam0.001", "B5_A1_lam0.001"}
    assert result["experiments"]["B5_A1_lam0.001"]["train_embedding_key"] == "A1"
    assert result["experiments"]["B2_lam0.001"]["subject_grl_enabled"] is True
    assert result["experiments"]["B0"]["adapter_enabled"] is False
    assert "domain_probes" in result["experiments"]["B1"]
    assert out_json.exists()
    assert "| experiment | lambda | adapter | subject_grl | session_grl | train_emb |" in out_table.read_text(encoding="utf-8")


def test_run_video_grl_adapter_ablation_can_write_oof_representations(tmp_path):
    pytest.importorskip("torch")
    eval_path = tmp_path / "eval.npz"
    sample_ids = [f"s{i}" for i in range(18)]
    _write_video_npz(eval_path, sample_ids, offset=0.0)

    repr_path = tmp_path / "repr.npz"
    run_video_grl_adapter_ablation(
        eval_embeddings=eval_path,
        train_embeddings={},
        target_label="fatigue",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "metrics.md",
        representation_out=repr_path,
        variants=["B0", "B1"],
        fold_strategy="random_window_split",
        epochs=2,
        batch_size=8,
        adapter_dim=6,
        hidden_dim=5,
        learning_rate=0.01,
        seed=7,
    )

    with np.load(repr_path, allow_pickle=True) as loaded:
        assert loaded["sample_id"].astype(str).tolist() == sample_ids
        assert loaded["repr__B0"].shape == (18, 256)
        assert loaded["repr__B1"].shape == (18, 6)
        assert loaded["pred__B1"].shape == (18,)


def test_summarize_grl_repeat_stability_groups_fold_metrics_by_subject(tmp_path):
    root = tmp_path / "repeat"
    for seed, r_b1, r_b2 in [(41, 0.1, 0.2), (42, 0.3, 0.4)]:
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        _write_repeat_report(seed_dir / "loso_metrics.json", "leave_one_subject_out", r_b1, r_b2)
        _write_repeat_report(seed_dir / "s4_metrics.json", "within_subject_session_leave_out", r_b1 + 0.1, r_b2 + 0.1)
        _write_repeat_report(seed_dir / "s2_metrics.json", "within_subject_chronological_split", r_b1 + 0.2, r_b2 + 0.2)

    result = summarize_grl_repeat_stability(
        report_root=root,
        variants=["B1", "B2_lam0.005"],
        out_json=tmp_path / "stability.json",
        out_table=tmp_path / "stability.md",
    )

    assert result["overall"]["LOSO"]["B1"]["pearson_r_mean_mean"] == pytest.approx(0.2)
    subject_rows = result["subject_metrics"]["S2"]["B2_lam0.005"]
    assert subject_rows["sub-01"]["seed_count"] == 2
    assert "S2 subject-wise error" in (tmp_path / "stability.md").read_text(encoding="utf-8")


def test_audit_grl_representations_reports_probes_ridge_and_variance(tmp_path):
    repr_path = tmp_path / "repr.npz"
    sample_ids = [f"s{i}" for i in range(36)]
    subject_id = np.asarray([f"sub-{(idx % 3) + 1:02d}" for idx in range(36)], dtype=object)
    event_id = np.asarray([f"{subject}_ses-{(idx % 4) + 1:02d}_event-{idx:02d}" for idx, subject in enumerate(subject_id)], dtype=object)
    target = np.asarray([float(idx % 6) for idx in range(36)], dtype=np.float32)
    base = np.stack([target, np.arange(36, dtype=np.float32), np.ones(36, dtype=np.float32)], axis=1)
    np.savez_compressed(
        repr_path,
        sample_id=np.asarray(sample_ids, dtype=object),
        subject_id=subject_id,
        event_id=event_id,
        session_id=np.asarray([f"{s}_ses-{(idx % 4) + 1:02d}" for idx, s in enumerate(subject_id)], dtype=object),
        target=target,
        repr__B0=np.pad(base, ((0, 0), (0, 253))).astype(np.float32),
        pred__B0=target + 0.1,
        repr__B1=base.astype(np.float32),
        pred__B1=target + 0.2,
    )

    result = audit_grl_representations(
        representations=repr_path,
        variants=["B0", "B1"],
        out_json=tmp_path / "audit.json",
        out_table=tmp_path / "audit.md",
        ridge_strategies=["within_subject_chronological_split"],
    )

    assert result["variants"]["B1"]["embedding_dim"] == 3
    assert result["variants"]["B1"]["prediction_std"] > 0
    assert result["variants"]["B1"]["fatigue_ridge"]["within_subject_chronological_split"]["pearson_r_mean"] is not None
    assert "Fatigue Ridge S2 r" in (tmp_path / "audit.md").read_text(encoding="utf-8")


def _write_video_npz(path: Path, sample_ids: list[str], *, offset: float) -> None:
    index_by_id = {sample_id: int(sample_id[1:]) for sample_id in sample_ids}
    subject_id = np.asarray([f"sub-{(index_by_id[sample_id] % 3) + 1:02d}" for sample_id in sample_ids], dtype=object)
    event_id = np.asarray(
        [
            f"{subject}_ses-{(index_by_id[sample_id] % 3) + 1:02d}_event-{index_by_id[sample_id]:02d}"
            for sample_id, subject in zip(sample_ids, subject_id)
        ],
        dtype=object,
    )
    labels = np.asarray(
        [json.dumps({"fatigue": float((index_by_id[sample_id] % 5) + 0.1 * index_by_id[sample_id])}) for sample_id in sample_ids],
        dtype=object,
    )
    face_emb = np.zeros((len(sample_ids), 256), dtype=np.float32)
    for row, sample_id in enumerate(sample_ids):
        face_emb[row, 0] = offset + float(index_by_id[sample_id])
        face_emb[row, 1] = float(row % 4)
    modality_mask = np.zeros((len(sample_ids), 4), dtype=np.int8)
    modality_mask[:, 2] = 1
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        subject_id=subject_id,
        event_id=event_id,
        labels=labels,
        face_emb=face_emb,
        modality_mask=modality_mask,
    )


def _write_repeat_report(path: Path, strategy: str, r_b1: float, r_b2: float) -> None:
    def exp(name: str, r_value: float) -> dict:
        return {
            "row_count": 4,
            "pearson_r_mean": r_value,
            "rmse_mean": 1.0 - r_value,
            "folds": [
                {
                    "fold": "fold0",
                    "test_sample_ids": ["sub-01_ses-01_a", "sub-01_ses-01_b", "sub-02_ses-01_a", "sub-02_ses-01_b"],
                    "test_predictions": [0.0, 1.0, 1.0, 0.0],
                    "test_targets": [0.0, 1.0, 0.0, 1.0],
                    "test": {"pearson": r_value, "rmse": 1.0 - r_value},
                }
            ],
        }

    payload = {
        "fold_strategy": strategy,
        "experiments": {
            "B1": exp("B1", r_b1),
            "B2_lam0.005": exp("B2_lam0.005", r_b2),
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
