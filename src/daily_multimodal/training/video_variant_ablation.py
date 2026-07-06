from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from daily_multimodal.embeddings.contracts import validate_embedding_shape
from daily_multimodal.training.baseline_mlp import _fit_mlp, _predict
from daily_multimodal.training.subject_cv import SubjectFold, build_subject_folds


FACE_MASK_INDEX = 2
SAMPLE_MODES = {"strict_aligned", "behavior_retained"}
MEAN_BASELINE_NAMES = {"mean_baseline", "V0=mean_baseline"}


def run_video_variant_ablation(
    *,
    variants: Mapping[str, str | Path],
    target_label: str,
    sample_mode: str,
    out_json: Path | str,
    out_table: Path | str,
    bucket_flags_path: Path | str | None = None,
    mode: str = "video_only",
    epochs: int = 200,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 41,
    fold_strategy: str = "leave_one_subject_out",
    n_splits: int = 5,
) -> dict[str, Any]:
    if mode != "video_only":
        raise ValueError("video variant ablation currently supports only mode='video_only'")
    if sample_mode not in SAMPLE_MODES:
        raise ValueError(f"unsupported sample_mode: {sample_mode}")
    normalized = _normalize_variants(variants)
    sources = {
        name: _parse_variant_source(path)
        for name, path in normalized.items()
        if not _is_mean_baseline(path)
    }
    datasets = {
        name: _load_variant_dataset(source["eval_embeddings"], target_label=target_label)
        for name, source in sources.items()
    }
    train_datasets = {
        name: _load_variant_dataset(source["train_embeddings"], target_label=target_label)
        for name, source in sources.items()
        if source["train_embeddings"] is not None
    }
    if not datasets:
        raise ValueError("at least one non-baseline variant .npz is required")

    selected = _select_sample_sets(datasets, sample_mode=sample_mode)
    selected = {
        name: _attach_train_embeddings(data, train_datasets.get(name))
        for name, data in selected.items()
    }
    experiments: dict[str, Any] = {}
    reference_name = next(iter(datasets))
    for offset, (name, value) in enumerate(normalized.items()):
        if _is_mean_baseline(value):
            data = selected[reference_name]
            experiments[name] = _run_mean_baseline(
                name,
                data,
                fold_strategy=fold_strategy,
                n_splits=n_splits,
                seed=seed,
            )
        else:
            data = selected[name]
            experiments[name] = _run_embedding_experiment(
                name,
                data,
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                fold_seed=seed,
                model_seed=seed + offset,
                fold_strategy=fold_strategy,
                n_splits=n_splits,
            )

    result = {
        "stage": 26,
        "mode": mode,
        "target_label": target_label,
        "sample_mode": sample_mode,
        "variants": {
            name: "mean_baseline" if _is_mean_baseline(value) else _json_ready(sources[name])
            for name, value in normalized.items()
        },
        "sample_sets": _sample_set_summary(selected, sample_mode=sample_mode),
        "experiments": experiments,
        "paired_fold_deltas": _paired_fold_deltas(experiments),
    }
    if bucket_flags_path is not None:
        result["behavior_bucket_analysis"] = _behavior_bucket_analysis(
            experiments,
            bucket_flags_path=bucket_flags_path,
        )
    _write_outputs(result, out_json=out_json, out_table=out_table)
    return result


def _normalize_variants(variants: Mapping[str, str | Path] | list[str] | tuple[str, ...]) -> dict[str, str | Path]:
    if isinstance(variants, Mapping):
        out = {str(name): value for name, value in variants.items()}
    else:
        out = {}
        for item in variants:
            if "=" not in str(item):
                raise ValueError(f"variant must use NAME=path form: {item}")
            name, value = str(item).split("=", 1)
            out[name] = value
    if not out:
        raise ValueError("at least one variant is required")
    return out


def _is_mean_baseline(value: str | Path) -> bool:
    return str(value) in MEAN_BASELINE_NAMES


def _parse_variant_source(value: str | Path) -> dict[str, str | None]:
    text = str(value)
    if "::" in text:
        eval_path, train_path = text.split("::", 1)
        if not eval_path or not train_path:
            raise ValueError(f"train-override variant must use eval.npz::train.npz: {text}")
        return {"eval_embeddings": eval_path, "train_embeddings": train_path}
    return {"eval_embeddings": text, "train_embeddings": None}


def _load_variant_dataset(path: Path | str, *, target_label: str) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        required = {"sample_id", "subject_id", "labels", "face_emb", "modality_mask"}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing required arrays: {', '.join(missing)}")
        sample_id = loaded["sample_id"].astype(str)
        if len(set(sample_id.tolist())) != len(sample_id):
            raise ValueError(f"duplicate sample_id in {path}")
        subject_id = loaded["subject_id"].astype(str)
        event_id = loaded["event_id"].astype(str) if "event_id" in loaded.files else sample_id.copy()
        raw_labels = loaded["labels"].tolist()
        try:
            face_emb = validate_embedding_shape("face_emb", loaded["face_emb"]).astype(np.float32)
        except ValueError as exc:
            raise ValueError(f"{path} face_emb invalid: {exc}") from exc
        modality_mask = loaded["modality_mask"].astype(np.int8)
        _validate_row_count(path, "subject_id", len(subject_id), len(sample_id))
        _validate_row_count(path, "event_id", len(event_id), len(sample_id))
        _validate_row_count(path, "labels", len(raw_labels), len(sample_id))
        _validate_row_count(path, "face_emb", face_emb.shape[0], len(sample_id))
        _validate_row_count(path, "modality_mask", modality_mask.shape[0], len(sample_id))
        if modality_mask.ndim != 2 or modality_mask.shape[1] <= FACE_MASK_INDEX:
            raise ValueError(f"{path} modality_mask must include the face slot")
        labels = [_parse_json_object(value) for value in raw_labels]
        target = _target_values(path, sample_id, labels, target_label)
        return {
            "sample_id": sample_id,
            "subject_id": subject_id,
            "event_id": event_id,
            "target": target,
            "face_emb": face_emb,
            "modality_mask": modality_mask,
            "usable": modality_mask[:, FACE_MASK_INDEX].astype(bool),
        }


def _validate_row_count(path: Path, name: str, actual: int, expected: int) -> None:
    if int(actual) != int(expected):
        raise ValueError(f"{path} row count mismatch for {name}: expected {expected}, got {actual}")


def _target_values(
    path: Path,
    sample_id: np.ndarray,
    labels: list[dict[str, Any]],
    target_label: str,
) -> np.ndarray:
    values: list[float] = []
    for idx, row in enumerate(labels):
        sid = str(sample_id[idx])
        if target_label not in row:
            raise ValueError(f"{path} sample_id {sid!r} missing target label {target_label!r}")
        try:
            value = float(row[target_label])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} sample_id {sid!r} has non-numeric target label {target_label!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{path} sample_id {sid!r} has non-finite target label {target_label!r}")
        values.append(value)
    return np.asarray(values, dtype=np.float32)


def _parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("label JSON must decode to an object")
        return parsed
    if isinstance(value, dict):
        return value
    raise ValueError("labels must be JSON strings or objects")


def _select_sample_sets(datasets: dict[str, dict[str, Any]], *, sample_mode: str) -> dict[str, dict[str, Any]]:
    if sample_mode == "behavior_retained":
        return {name: _subset_dataset(data, np.flatnonzero(data["usable"])) for name, data in datasets.items()}

    common: set[str] | None = None
    for data in datasets.values():
        usable_ids = set(data["sample_id"][data["usable"]].tolist())
        common = usable_ids if common is None else common & usable_ids
    aligned_ids = common or set()
    first = next(iter(datasets.values()))
    ordered_ids = [sample_id for sample_id in first["sample_id"].tolist() if sample_id in aligned_ids]
    return {name: _subset_by_sample_ids(data, ordered_ids) for name, data in datasets.items()}


def _subset_by_sample_ids(data: dict[str, Any], sample_ids: list[str]) -> dict[str, Any]:
    index = {sample_id: idx for idx, sample_id in enumerate(data["sample_id"].tolist())}
    return _subset_dataset(data, np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64))


def _subset_dataset(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    subset = {
        "sample_id": data["sample_id"][indices],
        "subject_id": data["subject_id"][indices],
        "event_id": data["event_id"][indices],
        "target": data["target"][indices],
        "face_emb": data["face_emb"][indices],
        "modality_mask": data["modality_mask"][indices],
        "usable": data["usable"][indices],
    }
    if "train_face_emb" in data:
        subset["train_face_emb"] = data["train_face_emb"][indices]
    if "train_embeddings_source" in data:
        subset["train_embeddings_source"] = data["train_embeddings_source"]
    return subset


def _attach_train_embeddings(data: dict[str, Any], train_data: dict[str, Any] | None) -> dict[str, Any]:
    if train_data is None:
        return {**data, "train_face_emb": data["face_emb"], "train_embeddings_source": None}
    aligned = _subset_by_sample_ids(train_data, data["sample_id"].tolist())
    for key in ("sample_id", "subject_id", "event_id"):
        if aligned[key].astype(str).tolist() != data[key].astype(str).tolist():
            raise ValueError(f"train embeddings metadata mismatch for {key}")
    if not np.allclose(aligned["target"], data["target"]):
        raise ValueError("train embeddings target values do not match eval embeddings")
    if not bool(np.all(aligned["usable"])):
        raise ValueError("train embeddings must be usable for every selected eval row")
    return {
        **data,
        "train_face_emb": aligned["face_emb"],
        "train_embeddings_source": "train_override",
    }


def _run_embedding_experiment(
    name: str,
    data: dict[str, Any],
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    fold_seed: int,
    model_seed: int,
    fold_strategy: str,
    n_splits: int,
) -> dict[str, Any]:
    base = _experiment_base(name, data, variant_kind="face_embedding")
    if len(data["target"]) == 0:
        return _failed_experiment(base, "no usable face rows")
    try:
        folds = _build_video_folds(data, strategy=fold_strategy, n_splits=n_splits, seed=fold_seed)
    except ValueError as exc:
        return _failed_experiment(base, str(exc))
    fold_results = []
    for offset, fold in enumerate(folds):
        if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
            return _failed_experiment(base, f"{fold.name} has empty train/val/test split")
        train_face_emb = data.get("train_face_emb", data["face_emb"])
        model = _fit_mlp(
            train_face_emb[fold.train],
            data["target"][fold.train],
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=model_seed + offset,
        )
        fold_results.append(
            {
                "fold": fold.name,
                "train_subjects": fold.train_subjects,
                "val_subjects": fold.val_subjects,
                "test_subjects": fold.test_subjects,
                "train_sample_ids": data["sample_id"][fold.train].tolist(),
                "val_sample_ids": data["sample_id"][fold.val].tolist(),
                "test_sample_ids": data["sample_id"][fold.test].tolist(),
                "train": _evaluate_predictions(_predict(model, train_face_emb[fold.train]), data["target"][fold.train]),
                "val": _evaluate_predictions(_predict(model, data["face_emb"][fold.val]), data["target"][fold.val]),
                "test": _evaluate_predictions(_predict(model, data["face_emb"][fold.test]), data["target"][fold.test]),
                "test_predictions": _predict(model, data["face_emb"][fold.test]).astype(float).tolist(),
                "test_targets": data["target"][fold.test].astype(float).tolist(),
            }
        )
    return {**base, "fold_count": len(fold_results), "folds": fold_results, **_summarize_folds(fold_results)}


def _build_video_folds(
    data: dict[str, Any],
    *,
    strategy: str,
    n_splits: int,
    seed: int,
) -> list[SubjectFold]:
    if strategy in {"leave_one_subject_out", "grouped_k_fold"}:
        return build_subject_folds(data["subject_id"], strategy=strategy, n_splits=n_splits, seed=seed)
    if strategy == "within_subject_event_split":
        return _within_subject_event_folds(data, n_splits=n_splits, seed=seed)
    if strategy == "within_subject_chronological_split":
        return [_within_subject_chronological_fold(data)]
    if strategy == "within_subject_session_leave_out":
        return _within_subject_session_leave_out_folds(data)
    if strategy == "random_window_split":
        return [_random_window_fold(data, seed=seed)]
    raise ValueError(f"unsupported video fold strategy: {strategy}")


def _within_subject_event_folds(data: dict[str, Any], *, n_splits: int, seed: int) -> list[SubjectFold]:
    subjects = data["subject_id"].astype(str)
    events = data["event_id"].astype(str)
    unique_subjects = list(dict.fromkeys(subjects.tolist()))
    split_count = max(2, int(n_splits))
    rng = np.random.default_rng(seed)
    per_subject_groups: dict[str, list[np.ndarray]] = {}
    for subject in unique_subjects:
        subject_indices = np.flatnonzero(subjects == subject)
        unique_events = list(dict.fromkeys(events[subject_indices].tolist()))
        if len(unique_events) < 3:
            raise ValueError(f"within_subject_event_split requires at least three events for {subject}")
        shuffled = list(unique_events)
        rng.shuffle(shuffled)
        subject_split_count = min(split_count, len(shuffled))
        groups = [
            np.asarray(group, dtype=str)
            for group in np.array_split(np.asarray(shuffled, dtype=str), subject_split_count)
            if len(group)
        ]
        if len(groups) < 3:
            raise ValueError(f"within_subject_event_split could not form train/val/test event groups for {subject}")
        per_subject_groups[subject] = groups

    fold_count = min(len(groups) for groups in per_subject_groups.values())
    folds: list[SubjectFold] = []
    for index in range(fold_count):
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for subject, groups in per_subject_groups.items():
            test_events = set(groups[index % len(groups)].tolist())
            val_events = set(groups[(index + 1) % len(groups)].tolist())
            train_mask = (subjects == subject) & ~np.isin(events, list(test_events | val_events))
            val_mask = (subjects == subject) & np.isin(events, list(val_events))
            test_mask = (subjects == subject) & np.isin(events, list(test_events))
            train_parts.append(np.flatnonzero(train_mask))
            val_parts.append(np.flatnonzero(val_mask))
            test_parts.append(np.flatnonzero(test_mask))
        folds.append(
            _index_fold(
                f"within_event_{index:02d}",
                train=np.concatenate(train_parts),
                val=np.concatenate(val_parts),
                test=np.concatenate(test_parts),
                subjects=subjects,
            )
        )
    return folds


def _within_subject_chronological_fold(data: dict[str, Any]) -> SubjectFold:
    subjects = data["subject_id"].astype(str)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for subject in dict.fromkeys(subjects.tolist()):
        indices = np.flatnonzero(subjects == subject)
        train, val, test = _split_ordered_indices(indices)
        train_parts.append(train)
        val_parts.append(val)
        test_parts.append(test)
    return _index_fold(
        "within_chronological_60_20_20",
        train=np.concatenate(train_parts),
        val=np.concatenate(val_parts),
        test=np.concatenate(test_parts),
        subjects=subjects,
    )


def _within_subject_session_leave_out_folds(data: dict[str, Any]) -> list[SubjectFold]:
    subjects = data["subject_id"].astype(str)
    session_ids = np.asarray(
        [
            _session_id(str(subject), str(event_id), str(sample_id))
            for subject, event_id, sample_id in zip(subjects, data["event_id"], data["sample_id"])
        ],
        dtype=str,
    )
    unique_subjects = list(dict.fromkeys(subjects.tolist()))
    sessions_by_subject: dict[str, list[str]] = {}
    for subject in unique_subjects:
        sessions = list(dict.fromkeys(session_ids[subjects == subject].tolist()))
        if len(sessions) < 3:
            continue
        sessions_by_subject[subject] = sessions
    if not sessions_by_subject:
        raise ValueError("within_subject_session_leave_out requires at least one subject with three sessions")
    fold_count = min(len(sessions) for sessions in sessions_by_subject.values())
    folds: list[SubjectFold] = []
    for index in range(fold_count):
        train_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        test_parts: list[np.ndarray] = []
        for subject, sessions in sessions_by_subject.items():
            test_session = sessions[index % len(sessions)]
            val_session = sessions[(index + 1) % len(sessions)]
            subject_mask = subjects == subject
            train_parts.append(np.flatnonzero(subject_mask & (session_ids != test_session) & (session_ids != val_session)))
            val_parts.append(np.flatnonzero(subject_mask & (session_ids == val_session)))
            test_parts.append(np.flatnonzero(subject_mask & (session_ids == test_session)))
        folds.append(
            _index_fold(
                f"within_session_{index:02d}",
                train=np.concatenate(train_parts),
                val=np.concatenate(val_parts),
                test=np.concatenate(test_parts),
                subjects=subjects,
            )
        )
    return folds


def _session_id(subject_id: str, event_id: str, sample_id: str) -> str:
    for value in (event_id, sample_id):
        match = re.search(r"(sub-[^_]+)_+(ses-[^_]+)", value)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
    return f"{subject_id}_unknown-session"


def _random_window_fold(data: dict[str, Any], *, seed: int) -> SubjectFold:
    subjects = data["subject_id"].astype(str)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(subjects), dtype=np.int64)
    rng.shuffle(indices)
    train, val, test = _split_ordered_indices(indices)
    return _index_fold(
        "random_window_60_20_20",
        train=train,
        val=val,
        test=test,
        subjects=subjects,
    )


def _split_ordered_indices(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(indices) < 3:
        raise ValueError("within-subject splits require at least three rows per split unit")
    train_end = max(1, int(round(len(indices) * 0.6)))
    val_end = max(train_end + 1, int(round(len(indices) * 0.8)))
    if val_end >= len(indices):
        val_end = len(indices) - 1
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _index_fold(
    name: str,
    *,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    subjects: np.ndarray,
) -> SubjectFold:
    return SubjectFold(
        name=name,
        train=np.asarray(train, dtype=np.int64),
        val=np.asarray(val, dtype=np.int64),
        test=np.asarray(test, dtype=np.int64),
        train_subjects=list(dict.fromkeys(subjects[train].astype(str).tolist())),
        val_subjects=list(dict.fromkeys(subjects[val].astype(str).tolist())),
        test_subjects=list(dict.fromkeys(subjects[test].astype(str).tolist())),
    )


def _run_mean_baseline(
    name: str,
    data: dict[str, Any],
    *,
    fold_strategy: str,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    base = _experiment_base(name, data, variant_kind="mean_baseline")
    if len(data["target"]) == 0:
        return _failed_experiment(base, "no usable rows for mean baseline")
    try:
        folds = _build_video_folds(data, strategy=fold_strategy, n_splits=n_splits, seed=seed)
    except ValueError as exc:
        return _failed_experiment(base, str(exc))
    fold_results = []
    for fold in folds:
        if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
            return _failed_experiment(base, f"{fold.name} has empty train/val/test split")
        mean_value = float(np.mean(data["target"][fold.train]))
        fold_results.append(
            {
                "fold": fold.name,
                "train_subjects": fold.train_subjects,
                "val_subjects": fold.val_subjects,
                "test_subjects": fold.test_subjects,
                "train_sample_ids": data["sample_id"][fold.train].tolist(),
                "val_sample_ids": data["sample_id"][fold.val].tolist(),
                "test_sample_ids": data["sample_id"][fold.test].tolist(),
                "train": _evaluate_predictions(np.full(len(fold.train), mean_value, dtype=np.float32), data["target"][fold.train]),
                "val": _evaluate_predictions(np.full(len(fold.val), mean_value, dtype=np.float32), data["target"][fold.val]),
                "test": _evaluate_predictions(np.full(len(fold.test), mean_value, dtype=np.float32), data["target"][fold.test]),
                "test_predictions": np.full(len(fold.test), mean_value, dtype=np.float32).astype(float).tolist(),
                "test_targets": data["target"][fold.test].astype(float).tolist(),
            }
        )
    return {**base, "fold_count": len(fold_results), "folds": fold_results, **_summarize_folds(fold_results)}


def _experiment_base(name: str, data: dict[str, Any], *, variant_kind: str) -> dict[str, Any]:
    return {
        "experiment": name,
        "variant_kind": variant_kind,
        "row_count": int(len(data["target"])),
        "sample_ids": data["sample_id"].tolist(),
        "fold_count": 0,
        "folds": [],
    }


def _failed_experiment(base: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "failure": reason,
        "rmse_mean": None,
        "rmse_std": None,
        "pearson_r_mean": None,
        "pearson_r_std": None,
        "pred_std_mean": None,
        "pred_std_std": None,
        "truth_std_mean": None,
        "truth_std_std": None,
        "error_std_mean": None,
        "error_std_std": None,
    }


def _evaluate_predictions(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    if len(truth) == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "pearson": None,
            "pred_std": None,
            "truth_std": None,
            "error_std": None,
        }
    error = pred - truth
    return {
        "count": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(math.sqrt(np.mean(np.square(error)))),
        "pearson": _pearson(pred, truth),
        "pred_std": float(np.std(pred)),
        "truth_std": float(np.std(truth)),
        "error_std": float(np.std(error)),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _summarize_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    metric_map = {
        "rmse": "rmse",
        "pearson_r": "pearson",
        "pred_std": "pred_std",
        "truth_std": "truth_std",
        "error_std": "error_std",
    }
    out: dict[str, Any] = {}
    for output_name, fold_key in metric_map.items():
        values = np.asarray(
            [fold["test"][fold_key] for fold in folds if fold["test"].get(fold_key) is not None],
            dtype=np.float32,
        )
        out[f"{output_name}_mean"] = None if values.size == 0 else float(np.mean(values))
        out[f"{output_name}_std"] = None if values.size == 0 else float(np.std(values))
    return out


def _paired_fold_deltas(experiments: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if "V1" not in experiments or "V2" not in experiments:
        return {}
    v1 = {fold["fold"]: fold for fold in experiments["V1"].get("folds", [])}
    deltas = []
    for fold in experiments["V2"].get("folds", []):
        match = v1.get(fold["fold"])
        if match is None:
            continue
        if not _same_fold_population(fold, match):
            continue
        deltas.append(
            {
                "fold": fold["fold"],
                "rmse_delta": _delta(fold["test"].get("rmse"), match["test"].get("rmse")),
                "pearson_r_delta": _delta(fold["test"].get("pearson"), match["test"].get("pearson")),
                "pred_std_delta": _delta(fold["test"].get("pred_std"), match["test"].get("pred_std")),
                "truth_std_delta": _delta(fold["test"].get("truth_std"), match["test"].get("truth_std")),
                "error_std_delta": _delta(fold["test"].get("error_std"), match["test"].get("error_std")),
                "v2_test_count": fold["test"].get("count"),
                "v1_test_count": match["test"].get("count"),
            }
        )
    return {"V2_vs_V1": deltas} if deltas else {}


def _same_fold_population(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in ("train_subjects", "val_subjects", "test_subjects"):
        if set(left.get(key, [])) != set(right.get(key, [])):
            return False
    for key in ("train_sample_ids", "val_sample_ids", "test_sample_ids"):
        if left.get(key, []) != right.get(key, []):
            return False
    return True


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _behavior_bucket_analysis(
    experiments: dict[str, Any],
    *,
    bucket_flags_path: Path | str,
) -> dict[str, Any]:
    flags = _load_bucket_flags(bucket_flags_path)
    bucket_specs = _bucket_specs()
    analysis: dict[str, Any] = {
        "bucket_flags_path": str(bucket_flags_path),
        "metrics": {},
    }
    for metric_name, buckets in bucket_specs.items():
        metric_result: dict[str, Any] = {}
        for bucket_name, predicate in buckets.items():
            bucket_samples = {
                sample_id
                for sample_id, row in flags.items()
                if predicate(_safe_float(row.get(metric_name, 0.0)))
            }
            experiments_result = {
                name: _evaluate_bucket_experiment(experiment, bucket_samples)
                for name, experiment in experiments.items()
            }
            metric_result[bucket_name] = {
                "rule": _bucket_rule_text(metric_name, bucket_name),
                "flag_sample_count": int(len(bucket_samples)),
                "experiments": experiments_result,
            }
            if "V1" in experiments_result and "V2" in experiments_result:
                metric_result[bucket_name]["V2_vs_V1"] = {
                    "rmse_delta": _delta(
                        experiments_result["V2"].get("rmse"),
                        experiments_result["V1"].get("rmse"),
                    ),
                    "pearson_r_delta": _delta(
                        experiments_result["V2"].get("pearson"),
                        experiments_result["V1"].get("pearson"),
                    ),
                    "count_delta": int(experiments_result["V2"].get("count", 0))
                    - int(experiments_result["V1"].get("count", 0)),
                }
        analysis["metrics"][metric_name] = metric_result
    return analysis


def _load_bucket_flags(path: Path | str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                records[sample_id] = row
    return records


def _bucket_specs() -> dict[str, dict[str, Any]]:
    return {
        "offscreen_ratio": {
            "low": lambda value: value < 0.1,
            "mid": lambda value: 0.1 <= value <= 0.5,
            "high": lambda value: value > 0.5,
        },
        "person_visible_ratio": {
            "low": lambda value: value < 0.3,
            "mid": lambda value: 0.3 <= value <= 0.8,
            "high": lambda value: value > 0.8,
        },
        "large_motion_ratio": {
            "low": lambda value: value < 0.05,
            "mid": lambda value: 0.05 <= value <= 0.3,
            "high": lambda value: value > 0.3,
        },
        "hand_visible_ratio": {
            "zero": lambda value: value == 0.0,
            "low": lambda value: 0.0 < value <= 0.3,
            "high": lambda value: value > 0.3,
        },
        "hand_occlusion_ratio": {
            "no_occlusion": lambda value: value == 0.0,
            "has_occlusion": lambda value: value > 0.0,
        },
    }


def _bucket_rule_text(metric_name: str, bucket_name: str) -> str:
    rules = {
        ("offscreen_ratio", "low"): "< 0.1",
        ("offscreen_ratio", "mid"): "0.1 <= value <= 0.5",
        ("offscreen_ratio", "high"): "> 0.5",
        ("person_visible_ratio", "low"): "< 0.3",
        ("person_visible_ratio", "mid"): "0.3 <= value <= 0.8",
        ("person_visible_ratio", "high"): "> 0.8",
        ("large_motion_ratio", "low"): "< 0.05",
        ("large_motion_ratio", "mid"): "0.05 <= value <= 0.3",
        ("large_motion_ratio", "high"): "> 0.3",
        ("hand_visible_ratio", "zero"): "= 0",
        ("hand_visible_ratio", "low"): "0 < value <= 0.3",
        ("hand_visible_ratio", "high"): "> 0.3",
        ("hand_occlusion_ratio", "no_occlusion"): "= 0",
        ("hand_occlusion_ratio", "has_occlusion"): "> 0",
    }
    return rules[(metric_name, bucket_name)]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _evaluate_bucket_experiment(experiment: dict[str, Any], bucket_samples: set[str]) -> dict[str, Any]:
    pred: list[float] = []
    truth: list[float] = []
    for fold in experiment.get("folds", []):
        sample_ids = fold.get("test_sample_ids", [])
        predictions = fold.get("test_predictions", [])
        targets = fold.get("test_targets", [])
        for sample_id, y_hat, y_true in zip(sample_ids, predictions, targets):
            if sample_id in bucket_samples:
                pred.append(float(y_hat))
                truth.append(float(y_true))
    metrics = _evaluate_predictions(
        np.asarray(pred, dtype=np.float32),
        np.asarray(truth, dtype=np.float32),
    )
    return {
        "count": metrics["count"],
        "rmse": metrics["rmse"],
        "pearson": metrics["pearson"],
        "mae": metrics["mae"],
        "pred_std": metrics["pred_std"],
        "truth_std": metrics["truth_std"],
        "error_std": metrics["error_std"],
    }


def _sample_set_summary(selected: dict[str, dict[str, Any]], *, sample_mode: str) -> dict[str, Any]:
    if sample_mode == "strict_aligned":
        first = next(iter(selected.values()))
        return {
            "strict_aligned": {
                "row_count": int(len(first["sample_id"])),
                "sample_ids": first["sample_id"].tolist(),
            }
        }
    return {
        "behavior_retained": {
            name: {"row_count": int(len(data["sample_id"])), "sample_ids": data["sample_id"].tolist()}
            for name, data in selected.items()
        }
    }


def _write_outputs(result: dict[str, Any], *, out_json: Path | str, out_table: Path | str) -> None:
    result = _json_ready(result)
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_table(result, out_table)


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| experiment | kind | rows | RMSE mean +/- std | Pearson r mean +/- std | pred_std mean +/- std | truth_std mean +/- std | error_std mean +/- std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, experiment in result["experiments"].items():
        rows.append(
            "| {name} | {kind} | {rows} | {rmse} | {r} | {pred_std} | {truth_std} | {error_std} |".format(
                name=name,
                kind=experiment.get("variant_kind"),
                rows=experiment.get("row_count", 0),
                rmse=_format_pair(experiment.get("rmse_mean"), experiment.get("rmse_std")),
                r=_format_pair(experiment.get("pearson_r_mean"), experiment.get("pearson_r_std")),
                pred_std=_format_pair(experiment.get("pred_std_mean"), experiment.get("pred_std_std")),
                truth_std=_format_pair(experiment.get("truth_std_mean"), experiment.get("truth_std_std")),
                error_std=_format_pair(experiment.get("error_std_mean"), experiment.get("error_std_std")),
            )
        )
    deltas = result.get("paired_fold_deltas", {}).get("V2_vs_V1", [])
    if deltas:
        rows.extend(["", "| paired fold | rmse_delta | pearson_r_delta |", "| --- | ---: | ---: |"])
        for row in deltas:
            rows.append(
                "| {fold} | {rmse} | {r} |".format(
                    fold=row["fold"],
                    rmse=_format_metric(row.get("rmse_delta")),
                    r=_format_metric(row.get("pearson_r_delta")),
                )
            )
    bucket_analysis = result.get("behavior_bucket_analysis", {}).get("metrics", {})
    if bucket_analysis:
        rows.extend(
            [
                "",
                "| bucket metric | bucket | rule | flag samples | V1 count | V1 RMSE | V1 r | V2 count | V2 RMSE | V2 r | RMSE delta | r delta |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric_name, buckets in bucket_analysis.items():
            for bucket_name, bucket in buckets.items():
                v1 = bucket.get("experiments", {}).get("V1", {})
                v2 = bucket.get("experiments", {}).get("V2", {})
                delta = bucket.get("V2_vs_V1", {})
                rows.append(
                    "| {metric} | {bucket_name} | {rule} | {flag_count} | {v1_count} | {v1_rmse} | {v1_r} | {v2_count} | {v2_rmse} | {v2_r} | {rmse_delta} | {r_delta} |".format(
                        metric=metric_name,
                        bucket_name=bucket_name,
                        rule=bucket.get("rule", ""),
                        flag_count=bucket.get("flag_sample_count", 0),
                        v1_count=v1.get("count", 0),
                        v1_rmse=_format_metric(v1.get("rmse")),
                        v1_r=_format_metric(v1.get("pearson")),
                        v2_count=v2.get("count", 0),
                        v2_rmse=_format_metric(v2.get("rmse")),
                        v2_r=_format_metric(v2.get("pearson")),
                        rmse_delta=_format_metric(delta.get("rmse_delta")),
                        r_delta=_format_metric(delta.get("pearson_r_delta")),
                    )
                )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def _format_pair(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "NA"
    mean = float(mean)
    std = float(std)
    if not math.isfinite(mean) or not math.isfinite(std):
        return "NA"
    return f"{mean:.4f} +/- {std:.4f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)
