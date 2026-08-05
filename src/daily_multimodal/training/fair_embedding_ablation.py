from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from daily_multimodal.training.baseline_mlp import (
    MODALITY_TO_EMB,
    MODALITY_TO_MASK_INDEX,
    _available_modality_mask,
    _evaluate_split,
    _features_for_modalities,
    _fit_mlp,
    _load_embedding_dataset,
    _run_overfit_check,
    _subject_split,
)
from daily_multimodal.training.cross_attention_fusion import (
    FusionBranchSpec,
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


FULL_MODALITIES = ("eeg", "wear", "face", "audio")
PATH_NEUTRAL_MODALITIES = ("eeg", "face", "audio")


def run_fair_embedding_ablation(
    *,
    basic_embeddings: Path | str,
    real_embeddings: Path | str,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    epochs: int = 200,
    overfit_limit: int = 128,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 23,
    modalities: tuple[str, ...] = FULL_MODALITIES,
    model: str = "concat_mlp",
    min_available_modalities: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    modalities = _normalize_modalities(modalities)
    if model not in {"concat_mlp", "learnable_cross_attention"}:
        raise ValueError(f"unsupported fair ablation model: {model}")
    basic = _load_fair_dataset(basic_embeddings, target_label=target_label)
    real = _load_fair_dataset(real_embeddings, target_label=target_label)
    aligned = _sample_ids_aligned(basic, real)
    failures = _validate_alignment(basic, real)
    if failures:
        result = _result_shell(
            basic_embeddings=basic_embeddings,
            real_embeddings=real_embeddings,
            target_label=target_label,
            modalities=modalities,
            row_count=min(len(basic["sample_id"]), len(real["sample_id"])),
            sample_id_aligned=aligned,
            failures=failures,
            experiments={},
        )
        _write_outputs(result, out_json=out_json, out_table=out_table)
        return result

    variants = {
        "basic_aligned": basic,
        "basic_no_path": _basic_no_path_variant(basic),
        "path_only": _path_only_variant(basic),
        "real": _strip_metadata(real),
    }
    experiments: dict[str, Any] = {}
    for offset, (name, data) in enumerate(variants.items()):
        experiments[name] = _run_variant(
            name,
            data,
            modalities=modalities,
            epochs=epochs,
            overfit_limit=overfit_limit,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed + offset,
            model=model,
            min_available_modalities=min_available_modalities,
            device=device,
        )
    _add_decisions(experiments)

    result = _result_shell(
        basic_embeddings=basic_embeddings,
        real_embeddings=real_embeddings,
        target_label=target_label,
        modalities=modalities,
        model=model,
        min_available_modalities=min_available_modalities,
        row_count=len(basic["sample_id"]),
        sample_id_aligned=True,
        failures=[],
        experiments=experiments,
    )
    _write_outputs(result, out_json=out_json, out_table=out_table)
    return result


def run_fusion_spec_fair_ablation(
    *,
    fusion_spec: Path | str,
    fusion_experiment: str,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    basic_embeddings: Path | str | None = None,
    epochs: int = 200,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
    seed: int = 23,
    min_available_modalities: int = 2,
    device: str | None = None,
) -> dict[str, Any]:
    real_dataset = _load_fusion_dataset_from_spec(
        fusion_spec,
        fusion_experiment=fusion_experiment,
        target_label=target_label,
        min_available_modalities=min_available_modalities,
    )
    variants: dict[str, FusionDataset] = {}
    if basic_embeddings is not None:
        basic = _load_fair_dataset(basic_embeddings, target_label=target_label)
        variants["basic_aligned"] = _fusion_dataset_from_basic_embedding_data(
            basic,
            reference=real_dataset,
            name="basic_aligned",
            neutralize_path=False,
        )
        variants["basic_no_path"] = _fusion_dataset_from_basic_embedding_data(
            _basic_no_path_variant(basic),
            reference=real_dataset,
            name="basic_no_path",
            neutralize_path=False,
        )
    variants["path_only"] = _path_only_fusion_dataset(real_dataset)
    variants["real"] = real_dataset
    experiments: dict[str, Any] = {}
    for offset, (name, dataset) in enumerate(variants.items()):
        experiments[name] = _run_fusion_dataset_variant(
            name,
            dataset,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed + offset,
            min_available_modalities=min_available_modalities,
            device=device,
        )
    _add_fusion_fair_decisions(experiments)
    result = {
        "stage": 18,
        "audit": "fusion_spec_fair_ablation",
        "target_label": target_label,
        "model": "learnable_cross_attention",
        "fusion_spec": str(fusion_spec),
        "fusion_experiment": fusion_experiment,
        "basic_embeddings": None if basic_embeddings is None else str(basic_embeddings),
        "row_count": int(len(real_dataset.sample_id)),
        "modalities": list(real_dataset.modalities),
        "min_available_modalities": int(min_available_modalities),
        "sample_id_aligned": True,
        "experiments": experiments,
        "failure_count": 0,
        "failures": [],
    }
    _write_outputs(result, out_json=out_json, out_table=out_table)
    return result


def _load_fair_dataset(path: Path | str, *, target_label: str) -> dict[str, Any]:
    data = _load_embedding_dataset(path, target_label=target_label)
    with np.load(path, allow_pickle=True) as loaded:
        row_count = len(data["sample_id"])
        data["event_id"] = _optional_string_array(loaded, "event_id", row_count)
        data["session_id"] = _optional_string_array(loaded, "session_id", row_count)
        data["source_paths"] = _optional_string_array(loaded, "source_paths", row_count)
    return data


def _optional_string_array(loaded: Any, key: str, row_count: int) -> np.ndarray:
    if key in loaded:
        return loaded[key].astype(str)
    return np.array([""] * row_count, dtype=str)


def _sample_ids_aligned(basic: dict[str, Any], real: dict[str, Any]) -> bool:
    return len(basic["sample_id"]) == len(real["sample_id"]) and np.array_equal(
        basic["sample_id"],
        real["sample_id"],
    )


def _validate_alignment(basic: dict[str, Any], real: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if len(basic["sample_id"]) != len(real["sample_id"]):
        failures.append(
            {
                "error_type": "row_count_mismatch",
                "basic_row_count": int(len(basic["sample_id"])),
                "real_row_count": int(len(real["sample_id"])),
            }
        )
        return failures
    if not np.array_equal(basic["sample_id"], real["sample_id"]):
        failures.append(
            {
                "error_type": "sample_id_mismatch",
                "message": "basic and real embeddings must have identical sample_id order",
            }
        )
    return failures


def _basic_no_path_variant(data: dict[str, Any]) -> dict[str, Any]:
    variant = _strip_metadata(data)
    row_count = len(variant["sample_id"])
    for index, modality in enumerate(PATH_NEUTRAL_MODALITIES, start=1):
        emb = np.zeros((row_count, 256), dtype=np.float32)
        emb[:, 0] = float(index)
        variant[MODALITY_TO_EMB[modality]] = emb
    return variant


def _path_only_variant(data: dict[str, Any]) -> dict[str, Any]:
    variant = _strip_metadata(data)
    row_count = len(variant["sample_id"])
    for modality in FULL_MODALITIES:
        values = []
        for row in range(row_count):
            metadata = "|".join(
                [
                    modality,
                    str(data["sample_id"][row]),
                    str(data["event_id"][row]),
                    str(data["subject_id"][row]),
                    str(data["session_id"][row]),
                    str(data["source_paths"][row]),
                ]
            )
            values.append(_metadata_vector(metadata))
        variant[MODALITY_TO_EMB[modality]] = np.vstack(values).astype(np.float32)
    return variant


def _metadata_vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=256).astype(np.float32)


def _strip_metadata(data: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in data.items()
        if key not in {"event_id", "session_id", "source_paths"}
    }


def _run_variant(
    name: str,
    data: dict[str, Any],
    *,
    modalities: tuple[str, ...],
    epochs: int,
    overfit_limit: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    model: str,
    min_available_modalities: int,
    device: str | None,
) -> dict[str, Any]:
    if model == "learnable_cross_attention":
        return _run_learnable_attention_variant(
            name,
            data,
            modalities=modalities,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
            min_available_modalities=min_available_modalities,
            device=device,
        )
    split = _subject_split(data["subject_id"])
    features = _features_for_modalities(data, modalities)
    valid = _available_modality_mask(data["modality_mask"], modalities)
    run_split = {split_name: indices[valid[indices]] for split_name, indices in split["indices"].items()}
    model = _fit_mlp(
        features[run_split["train"]],
        data["target"][run_split["train"]],
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    return {
        "experiment": name,
        "modalities": list(modalities),
        "seed": int(seed),
        "overfit_check": _run_overfit_check(
            features[valid],
            data["target"][valid],
            limit=overfit_limit,
            epochs=epochs,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
        ),
        "train": _evaluate_split(model, features, data["target"], run_split["train"]),
        "val": _evaluate_split(model, features, data["target"], run_split["val"]),
        "test": _evaluate_split(model, features, data["target"], run_split["test"]),
        "sample_counts": {split_name: int(len(indices)) for split_name, indices in run_split.items()},
    }


def _add_decisions(experiments: dict[str, Any]) -> None:
    basic_no_path_rmse = experiments["basic_no_path"]["test"]["rmse"]
    basic_aligned_rmse = experiments["basic_aligned"]["test"]["rmse"]
    for name, experiment in experiments.items():
        rmse = experiment["test"]["rmse"]
        experiment["test_rmse"] = rmse
        experiment["test_pearson_r"] = experiment["test"].get("pearson")
        if name == "path_only":
            experiment["decision"] = "leakage_control"
            experiment["reason"] = "path/sample/session metadata only"
        elif name == "basic_aligned":
            experiment["decision"] = "reference"
            experiment["reason"] = "aligned basic reference"
        elif rmse is None or basic_no_path_rmse is None:
            experiment["decision"] = "needs_review"
            experiment["reason"] = "missing comparable rmse"
        elif float(rmse) < float(basic_no_path_rmse):
            experiment["decision"] = "accepted_candidate"
            experiment["reason"] = "beats basic_no_path on aligned rows"
        else:
            experiment["decision"] = "rollback"
            experiment["reason"] = "does not beat basic_no_path on aligned rows"
        experiment["competitive_with_basic_aligned"] = (
            None
            if rmse is None or basic_aligned_rmse is None
            else bool(float(rmse) <= float(basic_aligned_rmse) * 1.05)
        )


def _result_shell(
    *,
    basic_embeddings: Path | str,
    real_embeddings: Path | str,
    target_label: str,
    modalities: tuple[str, ...],
    model: str = "concat_mlp",
    min_available_modalities: int = 2,
    row_count: int,
    sample_id_aligned: bool,
    failures: list[dict[str, Any]],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": 18,
        "audit": "fair_embedding_ablation",
        "target_label": target_label,
        "model": model,
        "modalities": list(modalities),
        "min_available_modalities": int(min_available_modalities),
        "basic_embeddings": str(basic_embeddings),
        "real_embeddings": str(real_embeddings),
        "row_count": int(row_count),
        "sample_id_aligned": bool(sample_id_aligned),
        "experiments": experiments,
        "failure_count": int(len(failures)),
        "failures": failures,
    }


def _write_outputs(result: dict[str, Any], *, out_json: Path | str, out_table: Path | str) -> None:
    json_path = Path(out_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_table(result, out_table)


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| experiment | test_rmse | test_r | decision | reason | train | val | test |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for name, experiment in result["experiments"].items():
        counts = experiment.get("sample_counts", {})
        rows.append(
            "| {name} | {rmse} | {r} | {decision} | {reason} | {train} | {val} | {test} |".format(
                name=name,
                rmse=_format_metric(experiment.get("test_rmse")),
                r=_format_metric(experiment.get("test_pearson_r")),
                decision=experiment.get("decision", ""),
                reason=experiment.get("reason", ""),
                train=counts.get("train", 0),
                val=counts.get("val", 0),
                test=counts.get("test", 0),
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
    unknown = sorted(set(normalized) - set(FULL_MODALITIES))
    if unknown:
        raise ValueError(f"unsupported modalities: {', '.join(unknown)}")
    return normalized


def _run_learnable_attention_variant(
    name: str,
    data: dict[str, Any],
    *,
    modalities: tuple[str, ...],
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    min_available_modalities: int,
    device: str | None,
) -> dict[str, Any]:
    split = _subject_split(data["subject_id"])
    dataset = _fusion_dataset_from_fair_data(data, modalities=modalities, name=name)
    valid = dataset.token_mask.sum(axis=1) >= int(min_available_modalities)
    run_split = {split_name: indices[valid[indices]] for split_name, indices in split["indices"].items()}
    trained = fit_learnable_cross_attention(
        dataset,
        train_indices=run_split["train"],
        val_indices=run_split["val"],
        config=LearnableAttentionConfig(
            token_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        ),
    )
    train_pred, _ = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["train"])
    val_pred, _ = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["val"])
    test_pred, attention = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["test"])
    return {
        "experiment": name,
        "model": "learnable_cross_attention",
        "modalities": list(dataset.modalities),
        "seed": int(seed),
        "overfit_check": {"skipped": True, "reason": "not defined for learnable_cross_attention fair controls"},
        "train": _evaluate_arrays(train_pred, dataset.target[run_split["train"]]),
        "val": _evaluate_arrays(val_pred, dataset.target[run_split["val"]]),
        "test": _evaluate_arrays(test_pred, dataset.target[run_split["test"]]),
        "sample_counts": {split_name: int(len(indices)) for split_name, indices in run_split.items()},
        "attention_summary": _attention_summary(attention, dataset.modalities),
    }


def _fusion_dataset_from_fair_data(data: dict[str, Any], *, modalities: tuple[str, ...], name: str) -> FusionDataset:
    token_modalities = tuple("video" if modality == "face" else modality for modality in modalities)
    arrays = {
        "eeg": data["eeg_emb"],
        "wear": data["wear_emb"],
        "video": data["face_emb"],
        "audio": data["audio_emb"],
    }
    tokens = np.stack([arrays[modality] for modality in token_modalities], axis=1).astype(np.float32)
    token_mask = np.stack(
        [data["modality_mask"][:, MODALITY_TO_MASK_INDEX[modality]] for modality in modalities],
        axis=1,
    ).astype(bool)
    return FusionDataset(
        name=name,
        modalities=token_modalities,
        sample_id=data["sample_id"].astype(str),
        event_id=_optional_fair_array(data, "event_id", default_prefix="event"),
        subject_id=data["subject_id"].astype(str),
        target=data["target"].astype(np.float32),
        tokens=tokens,
        token_mask=token_mask,
        branch_profiles={modality: "fair_ablation_variant" for modality in token_modalities},
        target_label=data["target_label"],
    )


def _optional_fair_array(data: dict[str, Any], key: str, *, default_prefix: str) -> np.ndarray:
    if key in data:
        return data[key].astype(str)
    return np.asarray([f"{default_prefix}-{idx}" for idx in range(len(data["sample_id"]))], dtype=str)


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
    if attention.size == 0:
        return {modality: 0.0 for modality in modalities}
    return {
        modality: float(np.mean(attention[:, index]))
        for index, modality in enumerate(modalities)
    }


def _load_fusion_dataset_from_spec(
    fusion_spec: Path | str,
    *,
    fusion_experiment: str,
    target_label: str,
    min_available_modalities: int,
) -> FusionDataset:
    config = load_fusion_matrix_config(fusion_spec)
    matches = [spec for spec in matrix_experiment_specs(config) if spec.name == fusion_experiment]
    if not matches:
        raise ValueError(f"unknown fusion experiment in spec: {fusion_experiment}")
    spec = matches[0]
    experiment = FusionExperimentSpec(
        name=spec.name,
        enabled_modalities=spec.enabled_modalities,
        target_label=target_label or spec.target_label,
        min_available_modalities=min_available_modalities,
    )
    return build_fusion_dataset(
        branches=branches_for_experiment(config, experiment.name),
        experiment=experiment,
        metadata_source=config.metadata_source,
    )


def _fusion_dataset_from_basic_embedding_data(
    data: dict[str, Any],
    *,
    reference: FusionDataset,
    name: str,
    neutralize_path: bool,
) -> FusionDataset:
    if neutralize_path:
        data = _basic_no_path_variant(data)
    index = {sample_id: idx for idx, sample_id in enumerate(data["sample_id"].astype(str).tolist())}
    missing = [sample_id for sample_id in reference.sample_id.astype(str).tolist() if sample_id not in index]
    if missing:
        raise ValueError(f"basic embeddings missing fusion sample_id values: {missing[:5]}")
    rows = np.asarray([index[sample_id] for sample_id in reference.sample_id.astype(str).tolist()], dtype=np.int64)
    tokens = []
    masks = []
    for modality in reference.modalities:
        packed_modality = "face" if modality == "video" else modality
        tokens.append(data[MODALITY_TO_EMB[packed_modality]][rows])
        masks.append(data["modality_mask"][rows, MODALITY_TO_MASK_INDEX[packed_modality]])
    return FusionDataset(
        name=name,
        modalities=reference.modalities,
        sample_id=reference.sample_id.copy(),
        event_id=reference.event_id.copy(),
        subject_id=reference.subject_id.copy(),
        target=reference.target.copy(),
        tokens=np.stack(tokens, axis=1).astype(np.float32),
        token_mask=np.stack(masks, axis=1).astype(bool),
        branch_profiles={modality: "basic_aligned" for modality in reference.modalities},
        target_label=reference.target_label,
    )


def _path_only_fusion_dataset(reference: FusionDataset) -> FusionDataset:
    rows = []
    for row_idx, sample_id in enumerate(reference.sample_id.astype(str).tolist()):
        token_vectors = []
        for modality in reference.modalities:
            metadata = "|".join(
                [
                    modality,
                    str(sample_id),
                    str(reference.event_id[row_idx]),
                    str(reference.subject_id[row_idx]),
                    str(reference.branch_profiles.get(modality, "")),
                ]
            )
            token_vectors.append(_metadata_vector(metadata))
        rows.append(np.stack(token_vectors, axis=0))
    return FusionDataset(
        name="path_only",
        modalities=reference.modalities,
        sample_id=reference.sample_id.copy(),
        event_id=reference.event_id.copy(),
        subject_id=reference.subject_id.copy(),
        target=reference.target.copy(),
        tokens=np.stack(rows, axis=0).astype(np.float32),
        token_mask=reference.token_mask.copy(),
        branch_profiles={modality: "path_only" for modality in reference.modalities},
        target_label=reference.target_label,
    )


def _run_fusion_dataset_variant(
    name: str,
    dataset: FusionDataset,
    *,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    min_available_modalities: int,
    device: str | None,
) -> dict[str, Any]:
    split = _subject_split(dataset.subject_id)
    valid = dataset.token_mask.sum(axis=1) >= int(min_available_modalities)
    run_split = {split_name: indices[valid[indices]] for split_name, indices in split["indices"].items()}
    trained = fit_learnable_cross_attention(
        dataset,
        train_indices=run_split["train"],
        val_indices=run_split["val"],
        config=LearnableAttentionConfig(
            token_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        ),
    )
    train_pred, _ = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["train"])
    val_pred, _ = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["val"])
    test_pred, attention = predict_with_learnable_cross_attention(trained, dataset, indices=run_split["test"])
    return {
        "experiment": name,
        "model": "learnable_cross_attention",
        "modalities": list(dataset.modalities),
        "seed": int(seed),
        "train": _evaluate_arrays(train_pred, dataset.target[run_split["train"]]),
        "val": _evaluate_arrays(val_pred, dataset.target[run_split["val"]]),
        "test": _evaluate_arrays(test_pred, dataset.target[run_split["test"]]),
        "sample_counts": {split_name: int(len(indices)) for split_name, indices in run_split.items()},
        "attention_summary": _attention_summary(attention, dataset.modalities),
        "initial_train_loss": trained.initial_train_loss,
        "final_train_loss": trained.final_train_loss,
    }


def _add_fusion_fair_decisions(experiments: dict[str, Any]) -> None:
    basic_no_path_rmse = experiments.get("basic_no_path", {}).get("test", {}).get("rmse")
    path_only_rmse = experiments.get("path_only", {}).get("test", {}).get("rmse")
    real_rmse = experiments.get("real", {}).get("test", {}).get("rmse")
    for name, experiment in experiments.items():
        rmse = experiment["test"].get("rmse")
        experiment["test_rmse"] = rmse
        experiment["test_pearson_r"] = experiment["test"].get("pearson")
        if name == "basic_aligned":
            experiment["decision"] = "reference"
            experiment["reason"] = "aligned basic reference on fusion rows"
        elif name == "path_only":
            experiment["decision"] = "leakage_control"
            experiment["reason"] = "sample/event/subject/profile metadata only"
        elif name == "basic_no_path":
            experiment["decision"] = "reference_control"
            experiment["reason"] = "basic embedding with path-like modalities neutralized"
        elif name == "real":
            if rmse is None or basic_no_path_rmse is None:
                experiment["decision"] = "needs_review"
                experiment["reason"] = "missing comparable basic_no_path rmse"
            elif float(rmse) >= float(basic_no_path_rmse):
                experiment["decision"] = "rollback"
                experiment["reason"] = "does not beat basic_no_path on identical fusion rows"
            elif path_only_rmse is not None and float(path_only_rmse) <= float(rmse):
                experiment["decision"] = "needs_review"
                experiment["reason"] = "path_only is not clearly below the real fusion run"
            else:
                experiment["decision"] = "accepted_candidate"
                experiment["reason"] = "beats basic_no_path and path_only controls on identical fusion rows"
        else:
            experiment["decision"] = "needs_review"
            experiment["reason"] = "unrecognized fusion fair variant"
