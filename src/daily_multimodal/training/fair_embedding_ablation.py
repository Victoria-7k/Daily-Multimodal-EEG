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
) -> dict[str, Any]:
    basic = _load_fair_dataset(basic_embeddings, target_label=target_label)
    real = _load_fair_dataset(real_embeddings, target_label=target_label)
    aligned = _sample_ids_aligned(basic, real)
    failures = _validate_alignment(basic, real)
    if failures:
        result = _result_shell(
            basic_embeddings=basic_embeddings,
            real_embeddings=real_embeddings,
            target_label=target_label,
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
            epochs=epochs,
            overfit_limit=overfit_limit,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            seed=seed + offset,
        )
    _add_decisions(experiments)

    result = _result_shell(
        basic_embeddings=basic_embeddings,
        real_embeddings=real_embeddings,
        target_label=target_label,
        row_count=len(basic["sample_id"]),
        sample_id_aligned=True,
        failures=[],
        experiments=experiments,
    )
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
    epochs: int,
    overfit_limit: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    split = _subject_split(data["subject_id"])
    features = _features_for_modalities(data, FULL_MODALITIES)
    valid = _available_modality_mask(data["modality_mask"], FULL_MODALITIES)
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
        "modalities": list(FULL_MODALITIES),
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
    row_count: int,
    sample_id_aligned: bool,
    failures: list[dict[str, Any]],
    experiments: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": 18,
        "audit": "fair_embedding_ablation",
        "target_label": target_label,
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
        "| experiment | test_rmse | decision | reason | train | val | test |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for name, experiment in result["experiments"].items():
        counts = experiment.get("sample_counts", {})
        rows.append(
            "| {name} | {rmse} | {decision} | {reason} | {train} | {val} | {test} |".format(
                name=name,
                rmse=_format_metric(experiment.get("test_rmse")),
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
