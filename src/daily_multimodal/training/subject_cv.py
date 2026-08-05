from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.baseline_mlp import (
    MODALITY_RUNS,
    _available_modality_mask,
    _evaluate_split,
    _features_for_modalities,
    _fit_mlp,
    _load_embedding_dataset,
)
from daily_multimodal.training.cross_attention_fusion import (
    FusionDataset,
    FusionExperimentSpec,
    LearnableAttentionConfig,
    build_fusion_dataset,
    fit_learnable_cross_attention,
    predict_with_learnable_cross_attention,
)
from daily_multimodal.training.fusion_matrix import (
    branches_for_experiment,
    load_fusion_matrix_config,
    matrix_experiment_specs,
)


DEFAULT_MODALITIES = tuple(MODALITY_RUNS["full"])


@dataclass(frozen=True)
class SubjectFold:
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_subjects: list[str]
    val_subjects: list[str]
    test_subjects: list[str]


def build_subject_folds(
    subjects: np.ndarray,
    *,
    strategy: str = "leave_one_subject_out",
    n_splits: int = 5,
    seed: int = 17,
) -> list[SubjectFold]:
    values = np.asarray(subjects).astype(str)
    unique_subjects = list(dict.fromkeys(values.tolist()))
    if len(unique_subjects) < 3:
        raise ValueError("subject CV requires at least three subjects for non-empty train/val/test")
    if strategy == "leave_one_subject_out":
        return _leave_one_subject_out(values, unique_subjects)
    if strategy == "grouped_k_fold":
        return _grouped_k_fold(values, unique_subjects, n_splits=n_splits, seed=seed)
    raise ValueError(f"unsupported subject CV strategy: {strategy}")


def run_subject_cv(
    *,
    embeddings: Path | str,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    strategy: str = "leave_one_subject_out",
    n_splits: int = 5,
    epochs: int = 200,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 17,
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    model: str = "concat_mlp",
    min_available_modalities: int = 2,
    device: str | None = None,
    fusion_spec: Path | str | None = None,
    fusion_experiment: str | None = None,
) -> dict[str, Any]:
    modalities = _normalize_modalities(modalities)
    if model not in {"concat_mlp", "learnable_cross_attention"}:
        raise ValueError(f"unsupported subject CV model: {model}")
    if fusion_spec is not None:
        if model != "learnable_cross_attention":
            raise ValueError("--fusion-spec is only supported with learnable_cross_attention")
        fusion_dataset = _fusion_dataset_from_spec(
            fusion_spec,
            target_label=target_label,
            fusion_experiment=fusion_experiment,
            min_available_modalities=min_available_modalities,
        )
        data = _embedding_data_from_fusion_dataset(fusion_dataset)
        features = None
        valid = np.ones(len(fusion_dataset.sample_id), dtype=bool)
        modalities = fusion_dataset.modalities
    else:
        data = _load_embedding_dataset(embeddings, target_label=target_label)
        features = _features_for_modalities(data, modalities) if model == "concat_mlp" else None
        valid = (
            _available_modality_mask(data["modality_mask"], modalities)
            if model == "concat_mlp"
            else _fusion_available_mask(data["modality_mask"], modalities, min_available_modalities=min_available_modalities)
        )
        fusion_dataset = _fusion_dataset_from_embedding_data(data, modalities=modalities) if model == "learnable_cross_attention" else None
    folds = build_subject_folds(data["subject_id"], strategy=strategy, n_splits=n_splits, seed=seed)
    fold_results = []
    for offset, fold in enumerate(folds):
        train = fold.train[valid[fold.train]]
        val = fold.val[valid[fold.val]]
        test = fold.test[valid[fold.test]]
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError(f"fold {fold.name} has an empty train/val/test split after modality mask filtering")
        if model == "concat_mlp":
            trained = _fit_mlp(
                features[train],
                data["target"][train],
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed + offset,
            )
            train_metrics = _evaluate_split(trained, features, data["target"], train)
            val_metrics = _evaluate_split(trained, features, data["target"], val)
            test_metrics = _evaluate_split(trained, features, data["target"], test)
            attention_summary = None
        else:
            assert fusion_dataset is not None
            trained_attention = fit_learnable_cross_attention(
                fusion_dataset,
                train_indices=train,
                val_indices=val,
                config=LearnableAttentionConfig(
                    token_dim=hidden_dim,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    seed=seed + offset,
                    device=device,
                ),
            )
            train_pred, _ = predict_with_learnable_cross_attention(trained_attention, fusion_dataset, indices=train)
            val_pred, _ = predict_with_learnable_cross_attention(trained_attention, fusion_dataset, indices=val)
            test_pred, attention = predict_with_learnable_cross_attention(trained_attention, fusion_dataset, indices=test)
            train_metrics = _evaluate_arrays(train_pred, data["target"][train])
            val_metrics = _evaluate_arrays(val_pred, data["target"][val])
            test_metrics = _evaluate_arrays(test_pred, data["target"][test])
            attention_summary = _attention_summary(attention, fusion_dataset.modalities)
        fold_results.append(
            {
                "fold": fold.name,
                "train_subjects": fold.train_subjects,
                "val_subjects": fold.val_subjects,
                "test_subjects": fold.test_subjects,
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
                "sample_counts": {
                    "train": int(len(train)),
                    "val": int(len(val)),
                    "test": int(len(test)),
                },
                **({} if attention_summary is None else {"attention_summary": attention_summary}),
            }
        )
    rmses = np.asarray([row["test"]["rmse"] for row in fold_results if row["test"]["rmse"] is not None], dtype=np.float32)
    pearsons = np.asarray(
        [row["test"]["pearson"] for row in fold_results if row["test"].get("pearson") is not None],
        dtype=np.float32,
    )
    result = {
        "stage": 20,
        "target_label": target_label,
        "embeddings": str(embeddings),
        "fusion_spec": None if fusion_spec is None else str(fusion_spec),
        "fusion_experiment": fusion_experiment,
        "model": model,
        "modalities": list(modalities),
        "min_available_modalities": int(min_available_modalities),
        "strategy": strategy,
        "fold_count": len(fold_results),
        "subject_leakage": _has_subject_leakage(folds, data["subject_id"]),
        "rmse_mean": None if rmses.size == 0 else float(np.mean(rmses)),
        "rmse_std": None if rmses.size == 0 else float(np.std(rmses)),
        "pearson_r_mean": None if pearsons.size == 0 else float(np.mean(pearsons)),
        "pearson_r_std": None if pearsons.size == 0 else float(np.std(pearsons)),
        "folds": fold_results,
    }
    _write_json(result, out_json)
    _write_table(result, out_table)
    return result


def _leave_one_subject_out(subjects: np.ndarray, unique_subjects: list[str]) -> list[SubjectFold]:
    folds: list[SubjectFold] = []
    for index, test_subject in enumerate(unique_subjects):
        val_subject = unique_subjects[(index + 1) % len(unique_subjects)]
        train_subjects = [subject for subject in unique_subjects if subject not in {test_subject, val_subject}]
        folds.append(_fold(subjects, f"loso_{test_subject}", train_subjects, [val_subject], [test_subject]))
    return folds


def _grouped_k_fold(
    subjects: np.ndarray,
    unique_subjects: list[str],
    *,
    n_splits: int,
    seed: int,
) -> list[SubjectFold]:
    split_count = max(2, min(int(n_splits), len(unique_subjects)))
    rng = np.random.default_rng(seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)
    groups = [group.tolist() for group in np.array_split(np.asarray(shuffled, dtype=object), split_count) if len(group)]
    if len(groups) < 2:
        raise ValueError("grouped_k_fold requires at least two non-empty groups")
    folds: list[SubjectFold] = []
    for index, test_subjects in enumerate(groups):
        val_subjects = groups[(index + 1) % len(groups)]
        held_out = set(test_subjects) | set(val_subjects)
        train_subjects = [subject for group in groups for subject in group if subject not in held_out]
        if not train_subjects:
            raise ValueError("grouped_k_fold requires enough subjects for a non-empty train split")
        folds.append(_fold(subjects, f"grouped_{index:02d}", train_subjects, val_subjects, test_subjects))
    return folds


def _fold(
    subjects: np.ndarray,
    name: str,
    train_subjects: list[str],
    val_subjects: list[str],
    test_subjects: list[str],
) -> SubjectFold:
    return SubjectFold(
        name=name,
        train=np.flatnonzero(np.isin(subjects, train_subjects)),
        val=np.flatnonzero(np.isin(subjects, val_subjects)),
        test=np.flatnonzero(np.isin(subjects, test_subjects)),
        train_subjects=list(train_subjects),
        val_subjects=list(val_subjects),
        test_subjects=list(test_subjects),
    )


def _has_subject_leakage(folds: list[SubjectFold], subjects: np.ndarray) -> bool:
    values = np.asarray(subjects).astype(str)
    for fold in folds:
        train_subjects = set(values[fold.train])
        val_subjects = set(values[fold.val])
        test_subjects = set(values[fold.test])
        if train_subjects & test_subjects or train_subjects & val_subjects or val_subjects & test_subjects:
            return True
    return False


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| fold | test_subjects | train_count | val_count | test_count | test_rmse | test_mae | test_r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in result["folds"]:
        rows.append(
            "| {fold} | {subjects} | {train_count} | {val_count} | {test_count} | {rmse} | {mae} | {r} |".format(
                fold=fold["fold"],
                subjects=",".join(fold["test_subjects"]),
                train_count=fold["train"]["count"],
                val_count=fold["val"]["count"],
                test_count=fold["test"]["count"],
                rmse=_format_metric(fold["test"]["rmse"]),
                mae=_format_metric(fold["test"]["mae"]),
                r=_format_metric(fold["test"].get("pearson")),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _normalize_modalities(modalities: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(str(item).strip() for item in modalities if str(item).strip())
    if not normalized:
        raise ValueError("at least one modality is required")
    valid = set(MODALITY_RUNS["full"])
    unknown = sorted(set(normalized) - valid)
    if unknown:
        raise ValueError(f"unsupported modalities: {', '.join(unknown)}")
    return normalized


def _fusion_available_mask(mask: np.ndarray, modalities: tuple[str, ...], *, min_available_modalities: int) -> np.ndarray:
    columns = [_mask_index_for_fusion_modality(modality) for modality in modalities]
    return mask[:, columns].sum(axis=1) >= int(min_available_modalities)


def _fusion_dataset_from_embedding_data(data: dict[str, Any], *, modalities: tuple[str, ...]) -> FusionDataset:
    token_modalities = tuple("video" if modality == "face" else modality for modality in modalities)
    arrays = {
        "eeg": data["eeg_emb"],
        "wear": data["wear_emb"],
        "video": data["face_emb"],
        "audio": data["audio_emb"],
    }
    tokens = np.stack([arrays[modality] for modality in token_modalities], axis=1).astype(np.float32)
    mask = np.stack(
        [data["modality_mask"][:, _mask_index_for_fusion_modality(modality)] for modality in modalities],
        axis=1,
    ).astype(bool)
    return FusionDataset(
        name="subject_cv",
        modalities=token_modalities,
        sample_id=data["sample_id"].astype(str),
        event_id=np.asarray([f"event-{idx}" for idx in range(len(data["sample_id"]))], dtype=str),
        subject_id=data["subject_id"].astype(str),
        target=data["target"].astype(np.float32),
        tokens=tokens,
        token_mask=mask,
        branch_profiles={modality: "packed_embedding" for modality in token_modalities},
        target_label=data["target_label"],
    )


def _fusion_dataset_from_spec(
    fusion_spec: Path | str,
    *,
    target_label: str,
    fusion_experiment: str | None,
    min_available_modalities: int,
) -> FusionDataset:
    config = load_fusion_matrix_config(fusion_spec)
    specs = matrix_experiment_specs(config)
    if fusion_experiment is None:
        if len(specs) != 1:
            raise ValueError("--fusion-experiment is required when fusion spec expands to multiple experiments")
        experiment = specs[0]
    else:
        matches = [spec for spec in specs if spec.name == fusion_experiment]
        if not matches:
            raise ValueError(f"unknown fusion experiment in spec: {fusion_experiment}")
        experiment = matches[0]
    experiment = FusionExperimentSpec(
        name=experiment.name,
        enabled_modalities=experiment.enabled_modalities,
        target_label=target_label or experiment.target_label,
        min_available_modalities=min_available_modalities,
    )
    return build_fusion_dataset(
        branches=branches_for_experiment(config, experiment.name),
        experiment=experiment,
        metadata_source=config.metadata_source,
    )


def _embedding_data_from_fusion_dataset(dataset: FusionDataset) -> dict[str, Any]:
    return {
        "sample_id": dataset.sample_id.astype(str),
        "subject_id": dataset.subject_id.astype(str),
        "target": dataset.target.astype(np.float32),
        "target_label": dataset.target_label,
    }


def _mask_index_for_fusion_modality(modality: str) -> int:
    if modality == "face":
        return 2
    return {"eeg": 0, "wear": 1, "video": 2, "audio": 3}[modality]


def _evaluate_arrays(pred: np.ndarray, truth: np.ndarray) -> dict[str, float | int | None]:
    if len(truth) == 0:
        return {"count": 0, "mae": None, "rmse": None, "pearson": None}
    error = pred - truth
    return {
        "count": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson": _pearson(pred, truth),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _attention_summary(attention: np.ndarray, modalities: tuple[str, ...]) -> dict[str, float]:
    return {
        modality: float(np.mean(attention[:, index])) if attention.size else 0.0
        for index, modality in enumerate(modalities)
    }
