from __future__ import annotations

import json
import math

import numpy as np
import pytest

import daily_multimodal.training.video_variant_ablation as video_variant_ablation
from daily_multimodal.training.video_variant_ablation import (
    _build_video_folds,
    _load_variant_dataset,
    _write_outputs,
    run_video_variant_ablation,
)


def test_video_variant_ablation_reports_distribution_metrics_and_mean_baseline(tmp_path):
    v1_path = tmp_path / "v1.npz"
    _write_variant(v1_path, ["a1", "a2", "b1", "b2", "c1", "c2"], masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    out_json = tmp_path / "metrics.json"
    out_table = tmp_path / "table.md"

    result = run_video_variant_ablation(
        variants={"V0": "mean_baseline", "V1": v1_path},
        target_label="fatigue",
        sample_mode="strict_aligned",
        out_json=out_json,
        out_table=out_table,
        epochs=3,
        hidden_dim=4,
        seed=3,
    )

    assert set(result["experiments"]) == {"V0", "V1"}
    assert result["experiments"]["V0"]["variant_kind"] == "mean_baseline"
    for experiment in result["experiments"].values():
        assert experiment["row_count"] == 6
        for field in (
            "rmse_mean",
            "rmse_std",
            "pearson_r_mean",
            "pearson_r_std",
            "pred_std_mean",
            "pred_std_std",
            "truth_std_mean",
            "truth_std_std",
            "error_std_mean",
            "error_std_std",
        ):
            assert field in experiment
        assert len(experiment["folds"]) == 3
    assert json.loads(out_json.read_text(encoding="utf-8"))["sample_mode"] == "strict_aligned"
    assert "pred_std mean" in out_table.read_text(encoding="utf-8")


def test_strict_aligned_intersects_sample_ids_and_face_masks_across_variants(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    _write_variant(v1_path, ["a1", "a2", "b1", "b2", "c1", "c2"], masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, ["a1", "a2", "b1", "b2", "c1", "c2"], masks=[1, 0, 1, 1, 1, 1], offset=0.2)

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="strict_aligned",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
    )

    assert result["experiments"]["V1"]["row_count"] == 5
    assert result["experiments"]["V2"]["row_count"] == 5
    assert result["experiments"]["V1"]["sample_ids"] == ["a1", "b1", "b2", "c1", "c2"]
    assert result["sample_sets"]["strict_aligned"]["row_count"] == 5


def test_behavior_retained_keeps_variant_specific_row_counts(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    _write_variant(v1_path, ["a1", "a2", "b1", "b2", "c1", "c2"], masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, ["a1", "a2", "b1", "b2", "c1", "c2"], masks=[1, 0, 1, 1, 1, 1], offset=0.2)

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="behavior_retained",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
    )

    assert result["experiments"]["V1"]["row_count"] == 6
    assert result["experiments"]["V2"]["row_count"] == 5
    assert result["sample_sets"]["behavior_retained"]["V1"]["row_count"] == 6
    assert result["sample_sets"]["behavior_retained"]["V2"]["row_count"] == 5


def test_train_embedding_override_is_used_only_for_train_fold(tmp_path, monkeypatch):
    eval_path = tmp_path / "eval.npz"
    train_path = tmp_path / "train_aug.npz"
    sample_ids = ["a0", "a1", "a2", "b0", "b1", "b2", "c0", "c1", "c2"]
    _write_variant(eval_path, sample_ids, masks=[1] * len(sample_ids), offset=0.0)
    _write_variant(train_path, sample_ids, masks=[1] * len(sample_ids), offset=1000.0)
    fit_inputs = []
    predict_inputs = []

    def fake_fit_mlp(x, y, **kwargs):
        del y, kwargs
        fit_inputs.append(np.asarray(x).copy())
        return {"model": "fake"}

    def fake_predict(model, x):
        del model
        predict_inputs.append(np.asarray(x).copy())
        return np.zeros(len(x), dtype=np.float32)

    monkeypatch.setattr(video_variant_ablation, "_fit_mlp", fake_fit_mlp)
    monkeypatch.setattr(video_variant_ablation, "_predict", fake_predict)

    result = run_video_variant_ablation(
        variants={"A1": f"{eval_path}::{train_path}"},
        target_label="fatigue",
        sample_mode="strict_aligned",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
        fold_strategy="leave_one_subject_out",
    )

    assert result["variants"]["A1"]["eval_embeddings"] == str(eval_path)
    assert result["variants"]["A1"]["train_embeddings"] == str(train_path)
    assert fit_inputs
    assert all(float(x[:, 0].min()) >= 1000.0 for x in fit_inputs)
    assert predict_inputs
    assert any(float(x[:, 0].max()) < 1000.0 for x in predict_inputs)


def test_paired_v2_vs_v1_fold_deltas_are_emitted_for_matching_folds(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    sample_ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    _write_variant(v1_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.5)

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="strict_aligned",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
    )

    deltas = result["paired_fold_deltas"]["V2_vs_V1"]
    assert len(deltas) == 3
    assert {row["fold"] for row in deltas} == {"loso_sub-a", "loso_sub-b", "loso_sub-c"}
    assert {"rmse_delta", "pearson_r_delta", "pred_std_delta", "truth_std_delta", "error_std_delta"} <= set(deltas[0])


def test_grouped_k_fold_uses_same_subject_splits_for_paired_variants(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    sample_ids = ["a1", "b1", "c1", "d1", "e1", "f1"]
    _write_variant(v1_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.5)

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="strict_aligned",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
        fold_strategy="grouped_k_fold",
        n_splits=3,
        seed=1,
    )

    v1_folds = {row["fold"]: row for row in result["experiments"]["V1"]["folds"]}
    for v2_fold in result["experiments"]["V2"]["folds"]:
        v1_fold = v1_folds[v2_fold["fold"]]
        assert set(v2_fold["train_subjects"]) == set(v1_fold["train_subjects"])
        assert set(v2_fold["val_subjects"]) == set(v1_fold["val_subjects"])
        assert set(v2_fold["test_subjects"]) == set(v1_fold["test_subjects"])


def test_behavior_retained_omits_paired_deltas_when_sample_sets_differ(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    sample_ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    _write_variant(v1_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, sample_ids, masks=[1, 0, 1, 1, 1, 1], offset=0.5)

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="behavior_retained",
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
    )

    assert result["paired_fold_deltas"] == {}


def test_behavior_bucket_analysis_reports_v1_v2_bucket_metrics(tmp_path):
    v1_path = tmp_path / "v1.npz"
    v2_path = tmp_path / "v2.npz"
    sample_ids = ["a1", "a2", "b1", "b2", "c1", "c2"]
    _write_variant(v1_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.1)
    _write_variant(v2_path, sample_ids, masks=[1, 1, 1, 1, 1, 1], offset=0.5)
    flags_path = tmp_path / "behavior_flags.jsonl"
    _write_behavior_flags(
        flags_path,
        [
            ("a1", 0.0, 1.0, 0.0, 0.0, 0.0),
            ("a2", 0.2, 0.6, 0.1, 0.2, 0.0),
            ("b1", 0.8, 0.1, 0.4, 0.6, 0.3),
            ("b2", 0.0, 1.0, 0.0, 0.0, 0.0),
            ("c1", 0.4, 0.7, 0.2, 0.1, 0.0),
            ("c2", 0.9, 0.0, 0.5, 0.7, 0.2),
        ],
    )

    result = run_video_variant_ablation(
        variants={"V1": v1_path, "V2": v2_path},
        target_label="fatigue",
        sample_mode="strict_aligned",
        bucket_flags_path=flags_path,
        out_json=tmp_path / "metrics.json",
        out_table=tmp_path / "table.md",
        epochs=3,
        hidden_dim=4,
    )

    buckets = result["behavior_bucket_analysis"]["metrics"]
    assert buckets["offscreen_ratio"]["low"]["flag_sample_count"] == 2
    assert buckets["offscreen_ratio"]["mid"]["flag_sample_count"] == 2
    assert buckets["offscreen_ratio"]["high"]["flag_sample_count"] == 2
    assert buckets["hand_visible_ratio"]["zero"]["flag_sample_count"] == 2
    assert buckets["hand_occlusion_ratio"]["has_occlusion"]["flag_sample_count"] == 2
    assert "rmse_delta" in buckets["large_motion_ratio"]["high"]["V2_vs_V1"]
    assert buckets["person_visible_ratio"]["high"]["experiments"]["V1"]["count"] == 2
    assert "bucket metric" in (tmp_path / "table.md").read_text(encoding="utf-8")


def test_within_subject_event_split_keeps_events_disjoint(tmp_path):
    path = tmp_path / "v1.npz"
    sample_ids = [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
    ]
    event_ids = ["ae1", "ae1", "ae2", "ae2", "ae3", "ae3", "be1", "be1", "be2", "be2", "be3", "be3"]
    _write_variant(path, sample_ids, masks=[1] * len(sample_ids), offset=0.1, event_ids=event_ids)
    data = _load_variant_dataset(path, target_label="fatigue")

    folds = _build_video_folds(data, strategy="within_subject_event_split", n_splits=3, seed=7)

    assert len(folds) == 3
    for fold in folds:
        train_events = set(data["event_id"][fold.train])
        val_events = set(data["event_id"][fold.val])
        test_events = set(data["event_id"][fold.test])
        assert not (train_events & val_events)
        assert not (train_events & test_events)
        assert not (val_events & test_events)
        assert set(data["subject_id"][fold.train]) == {"sub-a", "sub-b"}
        assert set(data["subject_id"][fold.test]) == {"sub-a", "sub-b"}


def test_within_subject_chronological_and_random_window_splits_run(tmp_path):
    path = tmp_path / "v1.npz"
    sample_ids = [f"{subject}{index}" for subject in ("a", "b", "c") for index in range(6)]
    _write_variant(path, sample_ids, masks=[1] * len(sample_ids), offset=0.1)

    for strategy in ("within_subject_chronological_split", "random_window_split"):
        result = run_video_variant_ablation(
            variants={"V1": path},
            target_label="fatigue",
            sample_mode="strict_aligned",
            out_json=tmp_path / f"{strategy}.json",
            out_table=tmp_path / f"{strategy}.md",
            epochs=3,
            hidden_dim=4,
            fold_strategy=strategy,
            seed=3,
        )
        assert result["experiments"]["V1"]["fold_count"] == 1
        fold = result["experiments"]["V1"]["folds"][0]
        assert fold["train"]["count"] > 0
        assert fold["val"]["count"] > 0
        assert fold["test"]["count"] > 0


def test_within_subject_session_leave_out_keeps_sessions_disjoint_per_subject(tmp_path):
    path = tmp_path / "v1.npz"
    sample_ids = [f"{subject}{session}{row}" for subject in ("a", "b") for session in range(4) for row in range(2)]
    event_ids = [
        f"sub-{subject}_ses-{session:02d}_row-{row:04d}"
        for subject in ("a", "b")
        for session in range(4)
        for row in range(2)
    ]
    _write_variant(path, sample_ids, masks=[1] * len(sample_ids), offset=0.1, event_ids=event_ids)
    data = _load_variant_dataset(path, target_label="fatigue")

    folds = _build_video_folds(data, strategy="within_subject_session_leave_out", n_splits=5, seed=7)

    assert len(folds) == 4
    for fold in folds:
        for subject in {"sub-a", "sub-b"}:
            train_sessions = {
                _session_from_event(event_id)
                for event_id in data["event_id"][fold.train]
                if str(event_id).startswith(subject)
            }
            val_sessions = {
                _session_from_event(event_id)
                for event_id in data["event_id"][fold.val]
                if str(event_id).startswith(subject)
            }
            test_sessions = {
                _session_from_event(event_id)
                for event_id in data["event_id"][fold.test]
                if str(event_id).startswith(subject)
            }
            assert train_sessions
            assert len(val_sessions) == 1
            assert len(test_sessions) == 1
            assert not (train_sessions & test_sessions)
            assert not (val_sessions & test_sessions)


def test_within_subject_session_leave_out_skips_subjects_with_too_few_sessions(tmp_path):
    path = tmp_path / "v1.npz"
    sample_ids = []
    event_ids = []
    for session in range(4):
        for row in range(2):
            sample_ids.append(f"a{session}{row}")
            event_ids.append(f"sub-a_ses-{session:02d}_row-{row:04d}")
    for session in range(2):
        for row in range(2):
            sample_ids.append(f"b{session}{row}")
            event_ids.append(f"sub-b_ses-{session:02d}_row-{row:04d}")
    _write_variant(path, sample_ids, masks=[1] * len(sample_ids), offset=0.1, event_ids=event_ids)
    data = _load_variant_dataset(path, target_label="fatigue")

    folds = _build_video_folds(data, strategy="within_subject_session_leave_out", n_splits=5, seed=7)

    assert len(folds) == 4
    for fold in folds:
        assert set(data["subject_id"][fold.train]) == {"sub-a"}
        assert set(data["subject_id"][fold.test]) == {"sub-a"}


def test_loader_rejects_malformed_variant_npz(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        sample_id=np.asarray(["a1", "b1"], dtype=object),
        subject_id=np.asarray(["sub-a", "sub-b"], dtype=object),
        labels=np.asarray([json.dumps({"fatigue": 1.0})], dtype=object),
        face_emb=np.zeros((2, 8), dtype=np.float32),
        modality_mask=np.ones((2, 4), dtype=np.int8),
    )

    try:
        _load_variant_dataset(path, target_label="fatigue")
    except ValueError as exc:
        message = str(exc)
        assert str(path) in message
        assert "row count" in message or "face_emb" in message
    else:
        raise AssertionError("malformed variant NPZ should fail")


def test_loader_reports_missing_or_nonfinite_targets(tmp_path):
    missing_path = tmp_path / "missing_label.npz"
    _write_variant(missing_path, ["a1", "b1", "c1"], masks=[1, 1, 1], offset=0.1, label_name="alert")
    try:
        _load_variant_dataset(missing_path, target_label="fatigue")
    except ValueError as exc:
        assert "missing target label 'fatigue'" in str(exc)
        assert "a1" in str(exc)
    else:
        raise AssertionError("missing target label should fail")

    nonfinite_path = tmp_path / "nonfinite_label.npz"
    _write_variant(nonfinite_path, ["a1", "b1", "c1"], masks=[1, 1, 1], offset=0.1, label_value=math.inf)
    try:
        _load_variant_dataset(nonfinite_path, target_label="fatigue")
    except ValueError as exc:
        assert "non-finite target" in str(exc)
    else:
        raise AssertionError("non-finite target should fail")


def test_output_json_replaces_nonfinite_metrics_with_null(tmp_path):
    result = {
        "experiments": {
            "V1": {
                "variant_kind": "face_embedding",
                "row_count": 1,
                "rmse_mean": math.nan,
                "rmse_std": 0.0,
                "pearson_r_mean": math.inf,
                "pearson_r_std": None,
                "pred_std_mean": 1.0,
                "pred_std_std": 0.0,
                "truth_std_mean": 1.0,
                "truth_std_std": 0.0,
                "error_std_mean": 1.0,
                "error_std_std": 0.0,
            }
        },
        "paired_fold_deltas": {},
    }

    out_json = tmp_path / "metrics.json"
    _write_outputs(result, out_json=out_json, out_table=tmp_path / "table.md")

    text = out_json.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    loaded = json.loads(text)
    assert loaded["experiments"]["V1"]["rmse_mean"] is None
    assert loaded["experiments"]["V1"]["pearson_r_mean"] is None


def _write_variant(path, sample_ids, *, masks, offset, label_name="fatigue", label_value=None, event_ids=None):
    subjects = [f"sub-{sample_id[0]}" for sample_id in sample_ids]
    event_ids = event_ids if event_ids is not None else [f"event-{sample_id}" for sample_id in sample_ids]
    labels = [
        json.dumps({label_name: float(index % 3) + 0.25 * (index // 3) if label_value is None else label_value})
        for index, _sample_id in enumerate(sample_ids)
    ]
    face_emb = np.zeros((len(sample_ids), 256), dtype=np.float32)
    for index in range(len(sample_ids)):
        face_emb[index, 0] = float(index) + offset
        face_emb[index, 1] = float(index % 2) - offset
    modality_mask = np.zeros((len(sample_ids), 4), dtype=np.int8)
    modality_mask[:, 2] = np.asarray(masks, dtype=np.int8)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=object),
        event_id=np.asarray(event_ids, dtype=object),
        subject_id=np.asarray(subjects, dtype=object),
        labels=np.asarray(labels, dtype=object),
        face_emb=face_emb,
        modality_mask=modality_mask,
        encoder_version=np.asarray(["synthetic"] * len(sample_ids), dtype=object),
    )


def _write_behavior_flags(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "offscreen_ratio": offscreen,
                    "person_visible_ratio": person_visible,
                    "large_motion_ratio": large_motion,
                    "hand_visible_ratio": hand_visible,
                    "hand_occlusion_ratio": hand_occlusion,
                }
            )
            for sample_id, offscreen, person_visible, large_motion, hand_visible, hand_occlusion in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _session_from_event(event_id):
    parts = str(event_id).split("_")
    return "_".join(parts[:2])
