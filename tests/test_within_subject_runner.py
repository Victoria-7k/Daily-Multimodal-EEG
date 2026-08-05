import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from daily_multimodal.training import within_subject_runner as runner
from daily_multimodal.training.within_subject_runner import (
    FoldIndices,
    JobSpec,
    derive_job_seed,
    job_paths,
    load_backend_decision,
    run_attention_fold,
    validate_resume_state,
)


def test_job_seed_depends_on_model_seed_not_split_seed_or_worker_order():
    first = derive_job_seed(1701, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    second = derive_job_seed(1701, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    changed = derive_job_seed(1702, "event_grouped_5fold", "exp", "sub-01", "fold-00")
    assert first == second
    assert first != changed


def test_run_state_and_checkpoint_paths_include_protocol_experiment_model_and_subject(tmp_path):
    paths = job_paths(
        out_dir=tmp_path / "reports",
        model_dir=tmp_path / "models",
        protocol="session_held_out",
        experiment="fusion_exp",
        model_name="learnable_cross_attention",
        subject_id="sub-01",
        fold_id="fold-00",
    )
    assert paths.prediction_path == (
        tmp_path
        / "reports"
        / "predictions"
        / "session_held_out"
        / "fusion_exp"
        / "learnable_cross_attention"
        / "sub-01.npz"
    )
    assert paths.state_path == (
        tmp_path
        / "reports"
        / "run_state"
        / "session_held_out"
        / "fusion_exp"
        / "learnable_cross_attention"
        / "sub-01.json"
    )
    assert paths.checkpoint_path == (
        tmp_path / "models" / "session_held_out" / "fusion_exp" / "sub-01" / "fold-00.pt"
    )


def test_resume_requires_matching_prediction_and_manifest_hashes(tmp_path):
    job = _job(tmp_path)
    job.prediction_path.parent.mkdir(parents=True)
    job.prediction_path.write_bytes(b"prediction")
    state = {
        "schema_version": 1,
        "status": "completed",
        "cohort_sha256": job.cohort_sha256,
        "split_sha256": job.split_sha256,
        "model_config_sha256": job.model_config_sha256,
        "prediction_sha256": runner.sha256_file(job.prediction_path),
        "checkpoint_sha256": {},
    }
    job.state_path.parent.mkdir(parents=True)
    job.state_path.write_text(json.dumps(state), encoding="utf-8")
    assert validate_resume_state(job, job.state_path) is True
    job.prediction_path.write_bytes(b"changed")
    assert validate_resume_state(job, job.state_path) is False


def test_backend_decision_loads_device_and_workers(tmp_path):
    path = tmp_path / "backend_decision.json"
    path.write_text(json.dumps({"device": "cpu", "workers": 4}), encoding="utf-8")
    assert load_backend_decision(path) == {"device": "cpu", "workers": 4}


def test_production_attention_job_never_predicts_train_indices(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        runner,
        "predict_with_learnable_cross_attention",
        lambda model, data, indices: calls.append(np.asarray(indices).copy()) or (np.zeros(len(indices)), None),
    )
    monkeypatch.setattr(
        runner,
        "fit_learnable_cross_attention",
        lambda data, train_indices, val_indices, config: object(),
    )
    fold = FoldIndices(
        name="fold-00",
        train=np.asarray([0, 1]),
        val=np.asarray([2]),
        test=np.asarray([3]),
    )
    run_attention_fold(_job(tmp_path), dataset=object(), fold=fold, config=object(), production=True)
    assert len(calls) == 2
    assert np.array_equal(calls[0], fold.val)
    assert np.array_equal(calls[1], fold.test)


def test_matrix_cli_dry_run_consumes_backend_decision(tmp_path):
    config = tmp_path / "within_subject.json"
    backend = tmp_path / "backend_decision.json"
    out_dir = tmp_path / "reports"
    model_dir = tmp_path / "models"
    config.write_text(json.dumps({"split_seed": 17, "model_seed": 1701}), encoding="utf-8")
    backend.write_text(json.dumps({"device": "cpu", "workers": 4}), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/45_run_within_subject_fusion_matrix.py",
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--model-dir",
            str(model_dir),
            "--backend-decision",
            str(backend),
            "--dry-run",
            "--production",
            "--resume",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    manifest = json.loads((out_dir / "within_subject_fusion_runtime.json").read_text(encoding="utf-8"))
    assert completed.returncode == 0, completed.stderr
    assert manifest["device"] == "cpu"
    assert manifest["workers"] == 4
    assert manifest["backend_decision_sha256"] == runner.sha256_file(backend)
    assert manifest["production"] is True
    assert manifest["resume"] is True


def _job(tmp_path: Path) -> JobSpec:
    paths = job_paths(
        out_dir=tmp_path / "reports",
        model_dir=tmp_path / "models",
        protocol="event_grouped_5fold",
        experiment="fusion_exp",
        model_name="learnable_cross_attention",
        subject_id="sub-01",
        fold_id="fold-00",
    )
    return JobSpec(
        protocol="event_grouped_5fold",
        experiment="fusion_exp",
        subject_id="sub-01",
        model_name="learnable_cross_attention",
        model_seed=1701,
        cohort_sha256="cohort",
        split_sha256="split",
        model_config_sha256="config",
        prediction_path=paths.prediction_path,
        state_path=paths.state_path,
        checkpoint_dir=paths.checkpoint_path.parent,
    )
