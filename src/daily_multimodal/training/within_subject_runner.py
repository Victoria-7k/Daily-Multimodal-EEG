from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_multimodal.training.cross_attention_fusion import (
    LearnableAttentionConfig,
    FusionDataset,
    fit_learnable_cross_attention,
    predict_with_learnable_cross_attention,
    save_learnable_cross_attention_model,
)
from daily_multimodal.training.fusion_matrix import (
    branches_for_experiment,
    load_fusion_matrix_config,
    matrix_experiment_specs,
)
from daily_multimodal.training.within_subject_metrics import (
    PredictionRecords,
    aggregate_event_predictions,
    fit_predict_concat_ridge,
    fit_predict_train_mean,
    regression_metrics,
    save_prediction_shard,
    summarize_event_subject_macro_pearson,
    summarize_pooled_oof,
    summarize_subject_oof,
)
from daily_multimodal.training.within_subject_splits import validate_split_manifest
from daily_multimodal.training.within_subject_video_routes import (
    VideoRouteRegistry,
    build_fold_video_tokens,
    load_video_route_registry,
)


@dataclass(frozen=True)
class FoldIndices:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class JobPaths:
    prediction_path: Path
    state_path: Path
    checkpoint_path: Path


@dataclass(frozen=True)
class JobSpec:
    protocol: str
    experiment: str
    subject_id: str
    model_name: str
    model_seed: int
    cohort_sha256: str
    split_sha256: str
    model_config_sha256: str
    prediction_path: Path
    state_path: Path
    checkpoint_dir: Path

    @property
    def job_id(self) -> str:
        return "|".join([self.protocol, self.experiment, self.model_name, self.subject_id])


@dataclass(frozen=True)
class JobResult:
    job_id: str
    prediction_sha256: str
    state_path: Path
    summary: dict[str, Any] | None = None


def derive_job_seed(
    model_seed: int,
    protocol: str,
    experiment: str,
    subject: str,
    fold: str,
) -> int:
    digest = sha256(f"{model_seed}|{protocol}|{experiment}|{subject}|{fold}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31 - 1)


def job_paths(
    *,
    out_dir: Path | str,
    model_dir: Path | str,
    protocol: str,
    experiment: str,
    model_name: str,
    subject_id: str,
    fold_id: str,
) -> JobPaths:
    out_root = Path(out_dir)
    model_root = Path(model_dir)
    return JobPaths(
        prediction_path=out_root / "predictions" / protocol / experiment / model_name / f"{subject_id}.npz",
        state_path=out_root / "run_state" / protocol / experiment / model_name / f"{subject_id}.json",
        checkpoint_path=model_root / protocol / experiment / subject_id / f"{fold_id}.pt",
    )


def validate_resume_state(job: JobSpec, state_path: Path | str) -> bool:
    path = Path(state_path)
    if not path.exists() or not job.prediction_path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if state.get("schema_version") != 1 or state.get("status") != "completed":
        return False
    expected = {
        "cohort_sha256": job.cohort_sha256,
        "split_sha256": job.split_sha256,
        "model_config_sha256": job.model_config_sha256,
        "prediction_sha256": sha256_file(job.prediction_path),
    }
    return all(str(state.get(key)) == str(value) for key, value in expected.items())


def load_backend_decision(path: Path | str) -> dict:
    decision = json.loads(Path(path).read_text(encoding="utf-8"))
    if "device" not in decision or "workers" not in decision:
        raise ValueError("backend decision requires device and workers")
    return {
        "device": str(decision["device"]),
        "workers": int(decision["workers"]),
    }


def run_attention_fold(
    job: JobSpec,
    *,
    dataset: Any,
    fold: FoldIndices,
    config: Any,
    production: bool,
) -> dict:
    model = fit_learnable_cross_attention(
        dataset,
        train_indices=fold.train,
        val_indices=fold.val,
        config=config,
    )
    predictions = {}
    if not production:
        predictions["train"] = predict_with_learnable_cross_attention(model, dataset, indices=fold.train)
    predictions["val"] = predict_with_learnable_cross_attention(model, dataset, indices=fold.val)
    predictions["test"] = predict_with_learnable_cross_attention(model, dataset, indices=fold.test)
    return {
        "job_id": job.job_id,
        "fold": fold.name,
        "predictions": predictions,
        "model": model,
    }


def run_job(job: JobSpec) -> JobResult:
    return JobResult(
        job_id=job.job_id,
        prediction_sha256=sha256_file(job.prediction_path),
        state_path=job.state_path,
    )


def execute_subject_job(
    job: JobSpec,
    *,
    dataset: FusionDataset,
    folds: Sequence[FoldIndices],
    config: Mapping[str, Any],
    production: bool,
    device: str | None,
    route_registry: VideoRouteRegistry | None = None,
    video_route_name: str | None = None,
    fold_route_cache: Mapping[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
) -> JobResult:
    """Run one subject/experiment/model job and persist its independent OOF shard."""
    prediction_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    route_metadata: list[dict[str, Any]] = []
    for fold in folds:
        working = dataset
        route_meta = {"route": None, "fit_scope": "none"}
        if video_route_name is not None:
            if route_registry is None:
                raise ValueError("video route requested without a route registry")
            if fold_route_cache is not None and fold.name in fold_route_cache:
                tokens, token_mask, route_meta = fold_route_cache[fold.name]
            else:
                tokens, token_mask, route_meta = build_fold_video_tokens(
                    dataset,
                    route_registry=route_registry,
                    route_name=video_route_name,
                    train_indices=fold.train,
                    val_indices=fold.val,
                    test_indices=fold.test,
                    seed=derive_job_seed(job.model_seed, job.protocol, job.experiment, job.subject_id, fold.name),
                    epochs=int(config.get("video_adapter_epochs", 80)),
                    batch_size=int(config.get("video_adapter_batch_size", 256)),
                    adapter_dim=int(config.get("video_adapter_dim", 64)),
                    hidden_dim=int(config.get("video_adapter_hidden_dim", 64)),
                    learning_rate=float(config.get("video_adapter_learning_rate", 1e-3)),
                    weight_decay=float(config.get("video_adapter_weight_decay", 1e-4)),
                    device=device,
                )
            working = replace(dataset, tokens=tokens, token_mask=token_mask)
        route_metadata.append(route_meta)

        if job.model_name == "train_mean":
            prediction = fit_predict_train_mean(working.target, fold.train, fold.test)
            attention = None
        elif job.model_name == "concat_ridge_alpha10":
            prediction, _ridge_meta = fit_predict_concat_ridge(
                working.tokens,
                working.token_mask,
                working.target,
                fold.train,
                fold.test,
                alpha=10.0,
            )
            attention = None
        elif job.model_name == "learnable_cross_attention":
            model_config = _attention_config(config, job=job, fold=fold, device=device, production=production)
            fold_result = run_attention_fold(
                job,
                dataset=working,
                fold=fold,
                config=model_config,
                production=production,
            )
            prediction, attention = fold_result["predictions"]["test"]
            checkpoint = job.checkpoint_dir / f"{fold.name}.pt"
            save_learnable_cross_attention_model(fold_result["model"], checkpoint)
            checkpoint_hashes[fold.name] = sha256_file(checkpoint)
        else:
            raise ValueError(f"unsupported within-subject model: {job.model_name}")

        session_id = (
            dataset.session_id.astype(str)
            if dataset.session_id is not None
            else np.asarray(["session-unknown"] * len(dataset.sample_id), dtype=str)
        )
        for offset, row_index in enumerate(fold.test.tolist()):
            prediction_rows.append(
                {
                    "sample_id": dataset.sample_id[row_index],
                    "event_id": dataset.event_id[row_index],
                    "subject_id": dataset.subject_id[row_index],
                    "session_id": session_id[row_index],
                    "fold_id": fold.name,
                    "target": dataset.target[row_index],
                    "prediction": prediction[offset],
                    "attention": None if attention is None else attention[offset],
                }
            )

    records = _records_from_rows(
        prediction_rows,
        model_name=job.model_name,
        experiment=job.experiment,
        protocol=job.protocol,
    )
    expected = dataset.sample_id[dataset.subject_id.astype(str) == str(job.subject_id)]
    summary = summarize_subject_oof(records, expected)
    summary["pooled"] = summarize_pooled_oof(records)
    summary["event_subject_macro"] = summarize_event_subject_macro_pearson(records)
    metadata = {
        "job_id": job.job_id,
        "model_name": job.model_name,
        "experiment": job.experiment,
        "protocol": job.protocol,
        "subject_id": job.subject_id,
        "cohort_sha256": job.cohort_sha256,
        "split_sha256": job.split_sha256,
        "model_config_sha256": job.model_config_sha256,
        "production": bool(production),
        "route_metadata": route_metadata,
        "summary": summary,
    }
    save_prediction_shard(job.prediction_path, records, metadata)
    state = {
        "schema_version": 1,
        "status": "completed",
        "job_id": job.job_id,
        "cohort_sha256": job.cohort_sha256,
        "split_sha256": job.split_sha256,
        "model_config_sha256": job.model_config_sha256,
        "prediction_sha256": sha256_file(job.prediction_path),
        "checkpoint_sha256": checkpoint_hashes,
        "summary": summary,
    }
    _write_json_atomic(job.state_path, state)
    return JobResult(
        job_id=job.job_id,
        prediction_sha256=state["prediction_sha256"],
        state_path=job.state_path,
        summary=summary,
    )


def run_within_subject_matrix(
    *,
    config_path: Path | str,
    out_dir: Path | str,
    model_dir: Path | str,
    protocol: str,
    device: str | None = None,
    workers: int = 1,
    screen_subjects: Sequence[str] | None = None,
    screen_experiments: Sequence[str] | None = None,
    epochs: int | None = None,
    hidden_dim: int | None = None,
    video_adapter_epochs: int | None = None,
    resume: bool = False,
    production: bool = False,
) -> dict[str, Any]:
    """Execute the frozen within-subject matrix serially with resumable jobs."""
    config_file = Path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    matrix_config = load_fusion_matrix_config(config["fusion_config"])
    specs = matrix_experiment_specs(matrix_config)
    if screen_experiments is not None:
        selected_experiments = {str(value) for value in screen_experiments}
        specs = [spec for spec in specs if spec.name in selected_experiments]
        if not specs:
            raise ValueError("screen_experiments selected no matrix experiment")
    cohort_path = Path(config["cohort_manifest"])
    split_path = Path(config["split_manifest"])
    cohort_manifest = json.loads(cohort_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split_manifest(
        split_manifest,
        cohort_hash=cohort_manifest["sample_id_sha256"],
        window_index_hash=cohort_manifest["source_hashes"]["window_index"],
    )
    cohort = np.asarray(cohort_manifest["sample_ids"], dtype=str)
    route_registry = None
    route_config_path = config.get("video_route_config")
    if route_config_path:
        route_registry = load_video_route_registry(route_config_path)
    datasets = {
        spec.name: _load_strict_dataset(matrix_config, spec, cohort)
        for spec in specs
    }
    subjects = _subject_manifests(split_manifest, protocol, screen_subjects)
    model_names = [str(name) for name in config.get("models", [])]
    if not model_names:
        raise ValueError("within-subject config has no models")
    model_config = dict(config)
    model_config["epochs"] = int(epochs if epochs is not None else config.get("production", {}).get("epochs", 200))
    model_config["hidden_dim"] = int(hidden_dim if hidden_dim is not None else config.get("production", {}).get("hidden_dim", 128))
    model_config["video_adapter_epochs"] = int(
        video_adapter_epochs
        if video_adapter_epochs is not None
        else config.get("video_adapter_epochs", 80)
    )
    out_root = Path(out_dir)
    model_root = Path(model_dir)
    results: list[dict[str, Any]] = []
    for spec in specs:
        dataset = datasets[spec.name]
        video_route_name = _video_route_name(spec.name, dataset, matrix_config)
        for subject in subjects:
            folds = _folds_for_subject(subject, protocol, dataset)
            if not folds:
                continue
            fold_route_cache = _prepare_fold_route_cache(
                dataset,
                folds=folds,
                route_registry=route_registry,
                video_route_name=video_route_name,
                config=model_config,
                job_seed=int(config.get("model_seed", 1701)),
                protocol=protocol,
                experiment=spec.name,
                subject_id=str(subject["subject_id"]),
                device=device,
            )
            for model_name in model_names:
                paths = job_paths(
                    out_dir=out_root,
                    model_dir=model_root,
                    protocol=protocol,
                    experiment=spec.name,
                    model_name=model_name,
                    subject_id=str(subject["subject_id"]),
                    fold_id="fold-00",
                )
                job = JobSpec(
                    protocol=protocol,
                    experiment=spec.name,
                    subject_id=str(subject["subject_id"]),
                    model_name=model_name,
                    model_seed=int(config.get("model_seed", 1701)),
                    cohort_sha256=str(cohort_manifest["sample_id_sha256"]),
                    split_sha256=sha256_file(split_path),
                    model_config_sha256=_sha256_json({**model_config, "device": device, "model": model_name}),
                    prediction_path=paths.prediction_path,
                    state_path=paths.state_path,
                    checkpoint_dir=paths.checkpoint_path.parent,
                )
                if resume and validate_resume_state(job, job.state_path):
                    results.append({"job_id": job.job_id, "status": "resumed", "prediction_path": str(job.prediction_path)})
                    continue
                result = execute_subject_job(
                    job,
                    dataset=dataset,
                    folds=folds,
                    config=model_config,
                    production=production,
                    device=device,
                    route_registry=route_registry,
                    video_route_name=video_route_name,
                    fold_route_cache=fold_route_cache,
                )
                results.append({"job_id": result.job_id, "status": "completed", "prediction_path": str(job.prediction_path)})
    summary = _aggregate_matrix_predictions(results, out_root)
    summary.update(
        {
            "protocol": protocol,
            "experiment_count": len(specs),
            "subject_count": len(subjects),
            "model_names": model_names,
            "workers_requested": int(workers),
            "execution_mode": "serial_job_runner",
            "production": bool(production),
        }
    )
    _write_json_atomic(out_root / "within_subject_fusion_summary.json", summary)
    return summary


def _prepare_fold_route_cache(
    dataset: FusionDataset,
    *,
    folds: Sequence[FoldIndices],
    route_registry: VideoRouteRegistry | None,
    video_route_name: str | None,
    config: Mapping[str, Any],
    job_seed: int,
    protocol: str,
    experiment: str,
    subject_id: str,
    device: str | None,
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    if video_route_name is None:
        return {}
    if route_registry is None:
        raise ValueError("video route requested without a route registry")
    return {
        fold.name: build_fold_video_tokens(
            dataset,
            route_registry=route_registry,
            route_name=video_route_name,
            train_indices=fold.train,
            val_indices=fold.val,
            test_indices=fold.test,
            seed=derive_job_seed(job_seed, protocol, experiment, subject_id, fold.name),
            epochs=int(config.get("video_adapter_epochs", 80)),
            batch_size=int(config.get("video_adapter_batch_size", 256)),
            adapter_dim=int(config.get("video_adapter_dim", 64)),
            hidden_dim=int(config.get("video_adapter_hidden_dim", 64)),
            learning_rate=float(config.get("video_adapter_learning_rate", 1e-3)),
            weight_decay=float(config.get("video_adapter_weight_decay", 1e-4)),
            device=device,
        )
        for fold in folds
    }


def _load_strict_dataset(matrix_config: Any, spec: Any, cohort: np.ndarray) -> FusionDataset:
    from daily_multimodal.training.cross_attention_fusion import build_fusion_dataset

    dataset = build_fusion_dataset(
        branches=branches_for_experiment(matrix_config, spec.name),
        experiment=spec,
        base_sample_ids=cohort,
        metadata_source=matrix_config.metadata_source,
    )
    if dataset.sample_id.astype(str).tolist() != cohort.astype(str).tolist():
        raise ValueError(f"{spec.name} strict dataset does not preserve cohort order")
    return dataset


def _subject_manifests(split_manifest: Mapping[str, Any], protocol: str, selected: Sequence[str] | None) -> list[dict[str, Any]]:
    subjects = split_manifest["protocols"][protocol]["subjects"]
    selected_set = None if selected is None else {str(value) for value in selected}
    return [row for row in subjects if selected_set is None or str(row["subject_id"]) in selected_set]


def _folds_for_subject(subject: Mapping[str, Any], protocol: str, dataset: FusionDataset) -> list[FoldIndices]:
    if subject.get("status") != "eligible":
        return []
    index = {sample_id: idx for idx, sample_id in enumerate(dataset.sample_id.astype(str).tolist())}
    subject_mask = dataset.subject_id.astype(str) == str(subject["subject_id"])
    folds = []
    for raw in subject.get("folds", []):
        train = _indices_for_manifest(raw["train_sample_ids"], index)
        val = _indices_for_manifest(raw["val_sample_ids"], index)
        test = _indices_for_manifest(raw["test_sample_ids"], index)
        if not np.all(subject_mask[np.concatenate([train, val, test])]):
            raise ValueError(f"{protocol} fold {raw['fold_id']} crosses subject boundary")
        if set(train.tolist()) & set(val.tolist()) or set(train.tolist()) & set(test.tolist()) or set(val.tolist()) & set(test.tolist()):
            raise ValueError(f"{protocol} fold {raw['fold_id']} has overlapping samples")
        folds.append(FoldIndices(name=str(raw["fold_id"]), train=train, val=val, test=test))
    return folds


def _indices_for_manifest(sample_ids: Sequence[str], index: Mapping[str, int]) -> np.ndarray:
    missing = [str(value) for value in sample_ids if str(value) not in index]
    if missing:
        raise ValueError(f"split manifest sample missing from fusion dataset: {missing[:5]}")
    return np.asarray([index[str(value)] for value in sample_ids], dtype=np.int64)


def _video_route_name(experiment_name: str, dataset: FusionDataset, matrix_config: Any) -> str | None:
    if "video" not in dataset.modalities:
        return None
    parts = experiment_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot derive video route from {experiment_name}")
    route = parts[2]
    if route not in matrix_config.video:
        raise ValueError(f"video route {route} missing from matrix config")
    return route


def _attention_config(config: Mapping[str, Any], *, job: JobSpec, fold: FoldIndices, device: str | None, production: bool) -> LearnableAttentionConfig:
    return LearnableAttentionConfig(
        token_dim=int(config.get("hidden_dim", 128)),
        epochs=int(config.get("epochs", 200)),
        batch_size=int(config.get("batch_size", 64)),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        dropout=float(config.get("dropout", 0.1)),
        patience=int(config.get("patience", 25)),
        seed=derive_job_seed(job.model_seed, job.protocol, job.experiment, job.subject_id, fold.name),
        device=device,
    )


def _records_from_rows(rows: Sequence[Mapping[str, Any]], *, model_name: str, experiment: str, protocol: str) -> PredictionRecords:
    if not rows:
        raise ValueError("cannot build prediction records from empty rows")
    attention = None if rows[0]["attention"] is None else np.asarray([row["attention"] for row in rows], dtype=np.float32)
    return PredictionRecords(
        sample_id=np.asarray([row["sample_id"] for row in rows], dtype=str),
        event_id=np.asarray([row["event_id"] for row in rows], dtype=str),
        subject_id=np.asarray([row["subject_id"] for row in rows], dtype=str),
        session_id=np.asarray([row["session_id"] for row in rows], dtype=str),
        fold_id=np.asarray([row["fold_id"] for row in rows], dtype=str),
        target=np.asarray([row["target"] for row in rows], dtype=np.float32),
        prediction=np.asarray([row["prediction"] for row in rows], dtype=np.float32),
        attention=attention,
        model_name=model_name,
        experiment=experiment,
        protocol=protocol,
    )


def _aggregate_matrix_predictions(results: Sequence[Mapping[str, Any]], out_root: Path) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[PredictionRecords]] = {}
    for row in results:
        path = Path(row["prediction_path"])
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as loaded:
            metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
            records = PredictionRecords(
                sample_id=loaded["sample_id"].astype(str),
                event_id=loaded["event_id"].astype(str),
                subject_id=loaded["subject_id"].astype(str),
                session_id=loaded["session_id"].astype(str),
                fold_id=loaded["fold_id"].astype(str),
                target=loaded["target"].astype(np.float32),
                prediction=loaded["prediction"].astype(np.float32),
                attention=None if "attention" not in loaded.files else loaded["attention"].astype(np.float32),
                model_name=str(metadata["model_name"]),
                experiment=str(metadata["experiment"]),
                protocol=str(metadata["protocol"]),
            )
        key = (records.protocol, records.experiment, records.model_name)
        groups.setdefault(key, []).append(records)
    summaries = []
    for key, rows in sorted(groups.items()):
        records = _concat_records(rows)
        event_records = aggregate_event_predictions(records)
        summaries.append(
            {
                "protocol": key[0],
                "experiment": key[1],
                "model": key[2],
                "window": regression_metrics(records.prediction, records.target),
                "event": regression_metrics(event_records.prediction, event_records.target),
                "pooled": summarize_pooled_oof(records),
                "event_subject_macro": summarize_event_subject_macro_pearson(records),
                "primary_metric": "event_subject_macro.pearson",
                "primary_value": summarize_event_subject_macro_pearson(records)["pearson"],
            }
        )
    return {
        "primary_metric": "event-level subject-macro Pearson",
        "decision_rule": "select by event_subject_macro.pearson; inspect event macro RMSE and centered pooled Pearson",
        "groups": summaries,
        "job_count": len(results),
    }


def _concat_records(rows: Sequence[PredictionRecords]) -> PredictionRecords:
    first = rows[0]
    return PredictionRecords(
        sample_id=np.concatenate([row.sample_id for row in rows]),
        event_id=np.concatenate([row.event_id for row in rows]),
        subject_id=np.concatenate([row.subject_id for row in rows]),
        session_id=np.concatenate([row.session_id for row in rows]),
        fold_id=np.concatenate([row.fold_id for row in rows]),
        target=np.concatenate([row.target for row in rows]),
        prediction=np.concatenate([row.prediction for row in rows]),
        attention=None if first.attention is None else np.concatenate([row.attention for row in rows]),
        model_name=first.model_name,
        experiment=first.experiment,
        protocol=first.protocol,
    )


def _sha256_json(value: Mapping[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path | str, value: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(out)


def run_jobs(jobs: Sequence[JobSpec], *, workers: int = 1) -> list[JobResult]:
    results = [run_job(job) for job in jobs]
    return sorted(results, key=lambda row: row.job_id)


def sha256_file(path: Path | str) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
