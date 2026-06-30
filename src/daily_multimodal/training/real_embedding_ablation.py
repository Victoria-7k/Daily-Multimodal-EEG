from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from daily_multimodal.training.baseline_mlp import (
    MODALITY_TO_MASK_INDEX,
    _available_modality_mask,
    _evaluate_split,
    _features_for_modalities,
    _fit_mlp,
    _load_embedding_dataset,
    _predict,
    _run_overfit_check,
    _subject_split,
)
from daily_multimodal.training.upgrade_ablation import _modality_attention_features


FULL_MODALITIES = ("eeg", "wear", "face", "audio")


def run_real_embedding_ablation(
    *,
    basic_embeddings: Path | str,
    real_embeddings: Path | str,
    baseline_metrics: Path | str,
    stage10_metrics: Path | str | None = None,
    target_label: str | None = None,
    out_table: Path | str,
    metrics_out: Path | str,
    failures_out: Path | str,
    epochs: int = 200,
    overfit_limit: int = 128,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seeds: Iterable[int] | None = None,
    bootstrap_iterations: int = 1000,
) -> dict[str, Any]:
    baseline = _read_json(baseline_metrics)
    label = target_label or baseline.get("target_label")
    if not label:
        raise ValueError("target_label is required when baseline metrics do not declare one.")
    seed_values = list(seeds) if seeds is not None else [17, 29, 41, 53, 67]

    basic = _load_embedding_dataset(basic_embeddings, target_label=label)
    real = _load_embedding_dataset(real_embeddings, target_label=label)
    failures = _validate_inputs(basic, real)
    if failures:
        result = _empty_result(label, baseline, failures)
        _write_outputs(result, out_table=out_table, metrics_out=metrics_out, failures_out=failures_out)
        return result

    baseline_rmse = _baseline_full_rmse(baseline)
    stage10 = _read_json(stage10_metrics) if stage10_metrics else None
    experiments: list[dict[str, Any]] = [
        _reference_experiment("baseline_reference_full_concat_mlp", baseline, embedding_source="basic", model="concat_mlp"),
    ]
    if stage10 is not None:
        experiments.append(_stage10_experiment(stage10, baseline_rmse))

    training_specs = [
        ("audio_real_only_replaced", _blend_modalities(basic, real, ("audio",)), FULL_MODALITIES, "concat_mlp"),
        ("face_real_only_replaced", _blend_modalities(basic, real, ("face",)), FULL_MODALITIES, "concat_mlp"),
        ("face_raw_openface_stats_v1", _blend_modalities(basic, real, ("face",)), FULL_MODALITIES, "concat_mlp"),
        ("face_preprocessed_openface_stats_v1", _blend_modalities(basic, real, ("face",)), FULL_MODALITIES, "concat_mlp"),
        ("eeg_real_only_replaced", _blend_modalities(basic, real, ("eeg",)), FULL_MODALITIES, "concat_mlp"),
        ("wear_real_only_replaced", _blend_modalities(basic, real, ("wear",)), FULL_MODALITIES, "concat_mlp"),
        ("all_real_concat_mlp", real, FULL_MODALITIES, "concat_mlp"),
        ("all_real_modality_token_attention", real, FULL_MODALITIES, "modality_token_attention"),
        ("all_real_without_face", real, ("eeg", "wear", "audio"), "concat_mlp"),
        ("all_real_with_raw_face", real, FULL_MODALITIES, "concat_mlp"),
        ("all_real_with_preprocessed_face", real, FULL_MODALITIES, "concat_mlp"),
    ]

    trained_by_name: dict[str, dict[str, Any]] = {}
    for offset, (name, data, modalities, model) in enumerate(training_specs):
        trained = _run_training_experiment(
            name,
            data,
            modalities=modalities,
            model=model,
            baseline_rmse=baseline_rmse,
            epochs=epochs,
            overfit_limit=overfit_limit,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed_values[0] + offset,
        )
        experiments.append(trained)
        trained_by_name[name] = trained

    face_seed_summary = _face_seed_summary(
        basic=basic,
        real=real,
        baseline_rmse=baseline_rmse,
        seeds=seed_values,
        epochs=epochs,
        overfit_limit=overfit_limit,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        bootstrap_iterations=bootstrap_iterations,
    )
    result = {
        "stage": 18,
        "target_label": label,
        "split": _split_summary(real),
        "basic_embeddings": str(basic_embeddings),
        "real_embeddings": str(real_embeddings),
        "baseline_metrics": str(baseline_metrics),
        "stage10_metrics": None if stage10_metrics is None else str(stage10_metrics),
        "experiments": experiments,
        "face_seed_summary": face_seed_summary,
        "failures": failures,
    }
    _write_outputs(result, out_table=out_table, metrics_out=metrics_out, failures_out=failures_out)
    return result


def _validate_inputs(basic: dict[str, Any], real: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for name, data in (("basic_embeddings", basic), ("real_embeddings", real)):
        split = _subject_split(data["subject_id"])
        missing = [split_name for split_name, indices in split["indices"].items() if len(indices) == 0]
        if missing:
            failures.append(
                {
                    "experiment": "real_embedding_ablation",
                    "source": name,
                    "error_type": "subject_split_incomplete",
                    "error": f"Missing samples for split(s): {', '.join(missing)}",
                    "missing_splits": missing,
                }
            )
    if len(basic["sample_id"]) == len(real["sample_id"]) and not np.array_equal(basic["sample_id"], real["sample_id"]):
        failures.append(
            {
                "experiment": "real_embedding_ablation",
                "source": "sample_id",
                "error_type": "shape_mismatch",
                "error": "basic and real embeddings have the same row count but different sample_id order",
            }
        )
    if len(basic["sample_id"]) != len(real["sample_id"]):
        failures.append(
            {
                "experiment": "real_embedding_ablation",
                "source": "sample_id",
                "error_type": "shape_mismatch",
                "error": f"basic row count {len(basic['sample_id'])} != real row count {len(real['sample_id'])}",
            }
        )
    return failures


def _empty_result(label: str, baseline: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage": 18,
        "target_label": label,
        "split": baseline.get("split", {}),
        "experiments": [],
        "face_seed_summary": {},
        "failures": failures,
    }


def _reference_experiment(
    name: str,
    metrics: dict[str, Any],
    *,
    embedding_source: str,
    model: str,
) -> dict[str, Any]:
    test = metrics.get("runs", {}).get("full", {}).get("test") or metrics.get("test") or {}
    return {
        "experiment": name,
        "embedding_source": embedding_source,
        "model": model,
        "modalities": list(FULL_MODALITIES),
        "seed": None,
        "overfit_check": metrics.get("overfit_check", {}),
        "train": {},
        "val": {},
        "test": test,
        "test_rmse": test.get("rmse"),
        "decision": "reference",
        "reason": "reference metric",
    }


def _stage10_experiment(metrics: dict[str, Any], baseline_rmse: float) -> dict[str, Any]:
    test = metrics.get("test", {})
    rmse = test.get("rmse")
    return {
        "experiment": "stage10_modality_token_attention",
        "embedding_source": "basic",
        "model": "modality_token_attention",
        "modalities": list(FULL_MODALITIES),
        "seed": None,
        "overfit_check": metrics.get("overfit_check", {}),
        "train": metrics.get("train", {}),
        "val": metrics.get("val", {}),
        "test": test,
        "test_rmse": rmse,
        "decision": _decision(rmse, baseline_rmse),
        "reason": _decision_reason(rmse, baseline_rmse),
    }


def _run_training_experiment(
    name: str,
    data: dict[str, Any],
    *,
    modalities: tuple[str, ...],
    model: str,
    baseline_rmse: float,
    epochs: int,
    overfit_limit: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    split = _subject_split(data["subject_id"])
    features = (
        _modality_attention_features(data)
        if model == "modality_token_attention"
        else _features_for_modalities(data, modalities)
    )
    valid = _available_modality_mask(data["modality_mask"], modalities)
    run_split = {name_: indices[valid[indices]] for name_, indices in split["indices"].items()}
    overfit = _run_overfit_check(
        features[valid],
        data["target"][valid],
        limit=overfit_limit,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    trained = _fit_mlp(
        features[run_split["train"]],
        data["target"][run_split["train"]],
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    test = _evaluate_split(trained, features, data["target"], run_split["test"])
    rmse = test.get("rmse")
    return {
        "experiment": name,
        "embedding_source": _embedding_source(name),
        "model": model,
        "modalities": list(modalities),
        "seed": int(seed),
        "overfit_check": overfit,
        "train": _evaluate_split(trained, features, data["target"], run_split["train"]),
        "val": _evaluate_split(trained, features, data["target"], run_split["val"]),
        "test": test,
        "test_rmse": rmse,
        "decision": _decision(rmse, baseline_rmse),
        "reason": _decision_reason(rmse, baseline_rmse),
        "sample_counts": {split_name: int(len(indices)) for split_name, indices in run_split.items()},
    }


def _face_seed_summary(
    *,
    basic: dict[str, Any],
    real: dict[str, Any],
    baseline_rmse: float,
    seeds: list[int],
    epochs: int,
    overfit_limit: int,
    hidden_dim: int,
    learning_rate: float,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    face_data = _blend_modalities(basic, real, ("face",))
    runs = [
        _run_training_experiment(
            "face_raw_openface_stats_v1",
            face_data,
            modalities=FULL_MODALITIES,
            model="concat_mlp",
            baseline_rmse=baseline_rmse,
            epochs=epochs,
            overfit_limit=overfit_limit,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed,
        )
        for seed in seeds
    ]
    rmses = np.array([run["test_rmse"] for run in runs if run["test_rmse"] is not None], dtype=np.float32)
    deltas = rmses - float(baseline_rmse)
    return {
        "experiment": "face_raw_openface_stats_v1",
        "seed_count": len(seeds),
        "seeds": [int(seed) for seed in seeds],
        "test_rmse_median": _safe_stat(rmses, np.median),
        "test_rmse_mean": _safe_stat(rmses, np.mean),
        "test_rmse_std": _safe_stat(rmses, np.std),
        "test_rmse_best": _safe_stat(rmses, np.min),
        "test_rmse_worst": _safe_stat(rmses, np.max),
        "bootstrap_ci95_delta_rmse": _bootstrap_ci(deltas, iterations=bootstrap_iterations, seed=seeds[0] if seeds else 0),
    }


def _blend_modalities(basic: dict[str, Any], real: dict[str, Any], real_modalities: tuple[str, ...]) -> dict[str, Any]:
    blended = {
        key: (value.copy() if isinstance(value, np.ndarray) else value)
        for key, value in basic.items()
    }
    mask = basic["modality_mask"].copy()
    for modality in real_modalities:
        blended[f"{modality}_emb"] = real[f"{modality}_emb"].copy()
        mask[:, MODALITY_TO_MASK_INDEX[modality]] = real["modality_mask"][:, MODALITY_TO_MASK_INDEX[modality]]
    blended["modality_mask"] = mask
    return blended


def _split_summary(data: dict[str, Any]) -> dict[str, Any]:
    split = _subject_split(data["subject_id"])
    return {
        "train_subjects": split["train_subjects"],
        "val_subjects": split["val_subjects"],
        "test_subjects": split["test_subjects"],
        "sample_counts": {name: int(len(indices)) for name, indices in split["indices"].items()},
    }


def _write_outputs(
    result: dict[str, Any],
    *,
    out_table: Path | str,
    metrics_out: Path | str,
    failures_out: Path | str,
) -> None:
    _write_metrics(result, metrics_out)
    _write_table(result["experiments"], out_table)
    out = Path(failures_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.get("failures", []), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_metrics(result: dict[str, Any], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_table(experiments: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| experiment | embedding_source | model | test_rmse | decision | reason |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for experiment in experiments:
        rows.append(
            "| {experiment} | {source} | {model} | {rmse} | {decision} | {reason} |".format(
                experiment=experiment["experiment"],
                source=experiment["embedding_source"],
                model=experiment["model"],
                rmse=_format_metric(experiment.get("test_rmse")),
                decision=experiment["decision"],
                reason=experiment["reason"],
            )
        )
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def _read_json(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _baseline_full_rmse(metrics: dict[str, Any]) -> float:
    return float(metrics["runs"]["full"]["test"]["rmse"])


def _decision(rmse: float | None, baseline_rmse: float) -> str:
    return "accepted" if rmse is not None and float(rmse) < float(baseline_rmse) else "rollback"


def _decision_reason(rmse: float | None, baseline_rmse: float) -> str:
    if rmse is None:
        return "missing test rmse"
    return "test_rmse improved baseline" if float(rmse) < float(baseline_rmse) else "test_rmse did not improve baseline"


def _embedding_source(experiment_name: str) -> str:
    if experiment_name.startswith("all_real"):
        return "real"
    if "_real_only_replaced" in experiment_name or experiment_name.startswith("face_"):
        return "basic_plus_real"
    return "basic"


def _safe_stat(values: np.ndarray, fn: Any) -> float | None:
    if values.size == 0:
        return None
    return float(fn(values))


def _bootstrap_ci(values: np.ndarray, *, iterations: int, seed: int) -> dict[str, float | None]:
    if values.size == 0:
        return {"low": None, "high": None, "iterations": int(iterations)}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(np.mean(sample)))
    low, high = np.percentile(np.array(means, dtype=np.float32), [2.5, 97.5])
    return {"low": float(low), "high": float(high), "iterations": int(iterations)}


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"
