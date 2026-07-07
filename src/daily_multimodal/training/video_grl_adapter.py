from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from daily_multimodal.embeddings.contracts import validate_embedding_shape
from daily_multimodal.embeddings.video_domain_robust import IGNORE_INDEX, gradient_reverse
from daily_multimodal.training.video_variant_ablation import (
    FACE_MASK_INDEX,
    _build_video_folds,
    _evaluate_predictions,
    _json_ready,
    _session_id,
    _summarize_folds,
)


DEFAULT_GRL_LAMBDAS = (0.0, 0.001, 0.005, 0.01, 0.05)
DEFAULT_B5_LAMBDAS = (0.001, 0.005, 0.01)


@dataclass(frozen=True)
class GrlVariantSpec:
    name: str
    use_adapter: bool
    use_subject_grl: bool = False
    use_session_grl: bool = False
    grl_lambda: float = 0.0
    train_embedding_key: str | None = None


def build_default_grl_variant_specs(
    *,
    lambdas: Sequence[float] = DEFAULT_GRL_LAMBDAS,
    b5_lambdas: Sequence[float] = DEFAULT_B5_LAMBDAS,
) -> list[GrlVariantSpec]:
    specs = [
        GrlVariantSpec("B0", use_adapter=False),
        GrlVariantSpec("B1", use_adapter=True),
    ]
    for value in lambdas:
        suffix = _lambda_suffix(value)
        specs.extend(
            [
                GrlVariantSpec(f"B2_lam{suffix}", use_adapter=True, use_subject_grl=True, grl_lambda=float(value)),
                GrlVariantSpec(f"B3_lam{suffix}", use_adapter=True, use_session_grl=True, grl_lambda=float(value)),
                GrlVariantSpec(
                    f"B4_lam{suffix}",
                    use_adapter=True,
                    use_subject_grl=True,
                    use_session_grl=True,
                    grl_lambda=float(value),
                ),
            ]
        )
    for value in b5_lambdas:
        suffix = _lambda_suffix(value)
        specs.append(
            GrlVariantSpec(
                f"B5_A1_lam{suffix}",
                use_adapter=True,
                use_subject_grl=True,
                use_session_grl=True,
                grl_lambda=float(value),
                train_embedding_key="A1",
            )
        )
    return specs


def load_grl_adapter_dataset(
    *,
    eval_embeddings: Path | str,
    train_embeddings: Mapping[str, Path | str] | None = None,
    target_label: str,
) -> dict[str, Any]:
    data = _load_one_bundle(eval_embeddings, target_label=target_label)
    train_by_key: dict[str, np.ndarray] = {}
    for key, path in (train_embeddings or {}).items():
        train_data = _load_one_bundle(path, target_label=target_label)
        aligned = _align_by_sample_id(train_data, data["sample_id"].astype(str).tolist())
        for meta_key in ("sample_id", "subject_id", "event_id", "session_id"):
            if aligned[meta_key].astype(str).tolist() != data[meta_key].astype(str).tolist():
                raise ValueError(f"train embeddings {key} metadata mismatch for {meta_key}")
        if not np.allclose(aligned["target"], data["target"]):
            raise ValueError(f"train embeddings {key} target values do not match eval embeddings")
        train_by_key[str(key)] = aligned["face_emb"]
    return {**data, "train_face_emb_by_key": train_by_key}


def run_video_grl_adapter_ablation(
    *,
    eval_embeddings: Path | str,
    train_embeddings: Mapping[str, Path | str] | None,
    target_label: str,
    out_json: Path | str,
    out_table: Path | str,
    representation_out: Path | str | None = None,
    variants: Sequence[str] | None = None,
    lambdas: Sequence[float] = DEFAULT_GRL_LAMBDAS,
    b5_lambdas: Sequence[float] = DEFAULT_B5_LAMBDAS,
    fold_strategy: str = "leave_one_subject_out",
    n_splits: int = 5,
    epochs: int = 80,
    batch_size: int = 256,
    adapter_dim: int = 128,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 41,
    device: str | None = None,
    compute_domain_probes: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    data = load_grl_adapter_dataset(
        eval_embeddings=eval_embeddings,
        train_embeddings=train_embeddings,
        target_label=target_label,
    )
    all_specs = build_default_grl_variant_specs(lambdas=lambdas, b5_lambdas=b5_lambdas)
    selected_specs = _select_specs(all_specs, variants)
    experiments = {
        spec.name: _run_one_experiment(
            data,
            spec=spec,
            fold_strategy=fold_strategy,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            adapter_dim=adapter_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + index,
            device=device,
            torch=torch,
            compute_domain_probes=compute_domain_probes,
        )
        for index, spec in enumerate(selected_specs)
    }
    result = {
        "stage": 35,
        "eval_embeddings": str(eval_embeddings),
        "train_embeddings": {key: str(value) for key, value in (train_embeddings or {}).items()},
        "target_label": target_label,
        "fold_strategy": fold_strategy,
        "n_splits": int(n_splits),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "adapter_dim": int(adapter_dim),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "row_count": int(len(data["target"])),
        "experiments": experiments,
    }
    _write_outputs(result, out_json=out_json, out_table=out_table)
    if representation_out is not None:
        _write_representations(
            data,
            experiments=experiments,
            output=representation_out,
            target_label=target_label,
            fold_strategy=fold_strategy,
        )
    return _json_ready(result)


def _run_one_experiment(
    data: dict[str, Any],
    *,
    spec: GrlVariantSpec,
    fold_strategy: str,
    n_splits: int,
    epochs: int,
    batch_size: int,
    adapter_dim: int,
    hidden_dim: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str | None,
    torch: Any,
    compute_domain_probes: bool,
) -> dict[str, Any]:
    base = {
        "variant_kind": "video_grl_adapter",
        "row_count": int(len(data["target"])),
        "adapter_enabled": bool(spec.use_adapter),
        "subject_grl_enabled": bool(spec.use_subject_grl),
        "session_grl_enabled": bool(spec.use_session_grl),
        "grl_lambda": float(spec.grl_lambda),
        "train_embedding_key": spec.train_embedding_key,
    }
    if spec.train_embedding_key is not None and spec.train_embedding_key not in data["train_face_emb_by_key"]:
        return {**base, **_failed_metrics(f"missing train embeddings for {spec.train_embedding_key}")}
    try:
        folds = _build_video_folds(data, strategy=fold_strategy, n_splits=n_splits, seed=seed)
    except ValueError as exc:
        return {**base, **_failed_metrics(str(exc))}

    subject_lookup = _class_lookup(data["subject_id"])
    session_lookup = _class_lookup(data["session_id"])
    fold_results = []
    oof_repr = np.full((len(data["target"]), _representation_dim(spec, data["face_emb"].shape[1], adapter_dim)), np.nan, dtype=np.float32)
    oof_pred = np.full(len(data["target"]), np.nan, dtype=np.float32)
    for offset, fold in enumerate(folds):
        if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
            return {**base, **_failed_metrics(f"{fold.name} has empty train/val/test split")}
        train_face = _train_face_matrix(data, spec)
        model = _fit_torch_model(
            train_x=train_face[fold.train],
            train_y=data["target"][fold.train],
            train_subject=_targets_for(data["subject_id"][fold.train], subject_lookup),
            train_session=_targets_for(data["session_id"][fold.train], session_lookup),
            eval_x=data["face_emb"],
            spec=spec,
            input_dim=data["face_emb"].shape[1],
            adapter_dim=adapter_dim,
            hidden_dim=hidden_dim,
            subject_count=len(subject_lookup),
            session_count=len(session_lookup),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed + offset,
            device=device,
            torch=torch,
        )
        train_pred, _ = _predict_with_model(model, train_face[fold.train], torch=torch, device=device)
        val_pred, _ = _predict_with_model(model, data["face_emb"][fold.val], torch=torch, device=device)
        test_pred, test_repr = _predict_with_model(model, data["face_emb"][fold.test], torch=torch, device=device)
        oof_repr[fold.test] = test_repr
        oof_pred[fold.test] = test_pred
        fold_results.append(
            {
                "fold": fold.name,
                "train_subjects": fold.train_subjects,
                "val_subjects": fold.val_subjects,
                "test_subjects": fold.test_subjects,
                "train_sample_ids": data["sample_id"][fold.train].tolist(),
                "val_sample_ids": data["sample_id"][fold.val].tolist(),
                "test_sample_ids": data["sample_id"][fold.test].tolist(),
                "train": _evaluate_predictions(train_pred, data["target"][fold.train]),
                "val": _evaluate_predictions(val_pred, data["target"][fold.val]),
                "test": _evaluate_predictions(test_pred, data["target"][fold.test]),
                "test_predictions": test_pred.astype(float).tolist(),
                "test_targets": data["target"][fold.test].astype(float).tolist(),
            }
        )
    result = {**base, "fold_count": len(fold_results), "folds": fold_results, **_summarize_folds(fold_results)}
    result["_oof_representation"] = oof_repr
    result["_oof_prediction"] = oof_pred
    if compute_domain_probes:
        valid = np.isfinite(oof_repr).all(axis=1)
        if np.any(valid):
            result["domain_probes"] = _domain_probes(oof_repr[valid], data["subject_id"][valid], data["session_id"][valid])
    return result


def _fit_torch_model(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_subject: np.ndarray,
    train_session: np.ndarray,
    eval_x: np.ndarray,
    spec: GrlVariantSpec,
    input_dim: int,
    adapter_dim: int,
    hidden_dim: int,
    subject_count: int,
    session_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str | None,
    torch: Any,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x_mean = train_x.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = train_x.std(axis=0, keepdims=True).astype(np.float32)
    x_std[x_std < 1e-6] = 1.0
    y_mean = float(train_y.mean())
    y_std = float(train_y.std()) or 1.0
    model = _AdapterRegressor(
        torch=torch,
        input_dim=input_dim,
        adapter_dim=adapter_dim,
        hidden_dim=hidden_dim,
        subject_count=subject_count,
        session_count=session_count,
        spec=spec,
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    x_train = torch.as_tensor((train_x - x_mean) / x_std, dtype=torch.float32, device=target_device)
    y_train = torch.as_tensor(((train_y - y_mean) / y_std).reshape(-1, 1), dtype=torch.float32, device=target_device)
    subject_train = torch.as_tensor(train_subject, dtype=torch.long, device=target_device)
    session_train = torch.as_tensor(train_session, dtype=torch.long, device=target_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    row_count = int(len(train_y))
    batch_size = max(1, min(int(batch_size), row_count))
    model.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(row_count, generator=generator).to(target_device)
        for start in range(0, row_count, batch_size):
            batch = order[start : start + batch_size]
            outputs = model(x_train[batch])
            loss = mse(outputs["fatigue"], y_train[batch])
            if spec.use_subject_grl:
                loss = loss + ce(outputs["subject_logits"], subject_train[batch])
            if spec.use_session_grl:
                loss = loss + ce(outputs["session_logits"], session_train[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        "module": model.eval(),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "eval_shape": tuple(eval_x.shape),
    }


def _predict_with_model(model: dict[str, Any], x: np.ndarray, *, torch: Any, device: str | None) -> tuple[np.ndarray, np.ndarray]:
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module = model["module"]
    x_norm = (x - model["x_mean"]) / model["x_std"]
    with torch.no_grad():
        tensor = torch.as_tensor(x_norm, dtype=torch.float32, device=target_device)
        outputs = module(tensor)
        pred = outputs["fatigue"].detach().cpu().numpy().reshape(-1).astype(np.float32)
        representation = outputs["representation"].detach().cpu().numpy().astype(np.float32)
    return pred * float(model["y_std"]) + float(model["y_mean"]), representation


class _AdapterRegressor:
    def __new__(
        cls,
        *,
        torch: Any,
        input_dim: int,
        adapter_dim: int,
        hidden_dim: int,
        subject_count: int,
        session_count: int,
        spec: GrlVariantSpec,
    ) -> Any:
        class _Module(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.spec = spec
                if spec.use_adapter:
                    self.adapter = torch.nn.Sequential(
                        torch.nn.Linear(input_dim, adapter_dim),
                        torch.nn.ReLU(),
                        torch.nn.LayerNorm(adapter_dim),
                    )
                    rep_dim = adapter_dim
                else:
                    self.adapter = None
                    rep_dim = input_dim
                self.fatigue_head = torch.nn.Sequential(
                    torch.nn.Linear(rep_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                self.subject_head = torch.nn.Linear(rep_dim, subject_count)
                self.session_head = torch.nn.Linear(rep_dim, session_count)

            def forward(self, values: Any) -> dict[str, Any]:
                representation = values if self.adapter is None else self.adapter(values)
                reversed_representation = gradient_reverse(representation, lambda_=self.spec.grl_lambda)
                return {
                    "representation": representation,
                    "fatigue": self.fatigue_head(representation),
                    "subject_logits": self.subject_head(reversed_representation),
                    "session_logits": self.session_head(reversed_representation),
                }

        return _Module()


def _load_one_bundle(path: Path | str, *, target_label: str) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        sample_id = loaded["sample_id"].astype(str)
        subject_id = loaded["subject_id"].astype(str)
        event_id = loaded["event_id"].astype(str) if "event_id" in loaded.files else sample_id.copy()
        labels = [_parse_json_object(value) for value in loaded["labels"].tolist()]
        face_emb = validate_embedding_shape("face_emb", loaded["face_emb"]).astype(np.float32)
        mask = loaded["modality_mask"].astype(np.int8)[:, FACE_MASK_INDEX].astype(bool)
    target = np.asarray([float(row[target_label]) for row in labels], dtype=np.float32)
    session_id = np.asarray(
        [_session_id(str(subject), str(event), str(sample)) for subject, event, sample in zip(subject_id, event_id, sample_id)],
        dtype=str,
    )
    return {
        "sample_id": sample_id[mask],
        "subject_id": subject_id[mask],
        "event_id": event_id[mask],
        "session_id": session_id[mask],
        "target": target[mask],
        "face_emb": face_emb[mask],
    }


def _align_by_sample_id(data: dict[str, Any], sample_ids: list[str]) -> dict[str, Any]:
    index = {sample_id: idx for idx, sample_id in enumerate(data["sample_id"].astype(str).tolist())}
    missing = [sample_id for sample_id in sample_ids if sample_id not in index]
    if missing:
        raise ValueError(f"train embeddings missing sample_id values: {missing[:5]}")
    indices = np.asarray([index[sample_id] for sample_id in sample_ids], dtype=np.int64)
    return {key: value[indices] if isinstance(value, np.ndarray) and len(value) == len(data["sample_id"]) else value for key, value in data.items()}


def _select_specs(all_specs: list[GrlVariantSpec], variants: Sequence[str] | None) -> list[GrlVariantSpec]:
    if not variants:
        return all_specs
    by_name = {spec.name: spec for spec in all_specs}
    missing = [name for name in variants if name not in by_name]
    if missing:
        raise ValueError(f"unknown GRL adapter variants: {', '.join(missing)}")
    return [by_name[name] for name in variants]


def _class_lookup(values: np.ndarray) -> dict[str, int]:
    return {value: idx for idx, value in enumerate(sorted(set(values.astype(str).tolist())))}


def _targets_for(values: np.ndarray, lookup: Mapping[str, int]) -> np.ndarray:
    return np.asarray([lookup.get(str(value), IGNORE_INDEX) for value in values], dtype=np.int64)


def _train_face_matrix(data: dict[str, Any], spec: GrlVariantSpec) -> np.ndarray:
    if spec.train_embedding_key is None:
        return data["face_emb"]
    return data["train_face_emb_by_key"][spec.train_embedding_key]


def _representation_dim(spec: GrlVariantSpec, input_dim: int, adapter_dim: int) -> int:
    return int(adapter_dim) if spec.use_adapter else int(input_dim)


def _domain_probes(x: np.ndarray, subject_id: np.ndarray, session_id: np.ndarray) -> dict[str, Any]:
    try:
        from daily_multimodal.training.video_embedding_probes import _classification_probe, _within_subject_session_probe
    except ImportError as exc:  # pragma: no cover
        return {"failure": str(exc)}
    data = {"embedding": x, "subject_id": subject_id.astype(str), "session_id": session_id.astype(str)}
    return {
        "subject_logreg": _classification_probe(x, subject_id, seed=41, n_splits=5),
        "within_subject_session_logreg": _within_subject_session_probe(data, seed=41, n_splits=5),
    }


def _failed_metrics(reason: str) -> dict[str, Any]:
    return {
        "failure": reason,
        "fold_count": 0,
        "folds": [],
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


def _write_outputs(result: dict[str, Any], *, out_json: Path | str, out_table: Path | str) -> None:
    ready = _json_ready(_strip_private_arrays(result))
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ready, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    _write_table(ready, out_table)


def _write_representations(
    data: dict[str, Any],
    *,
    experiments: Mapping[str, Any],
    output: Path | str,
    target_label: str,
    fold_strategy: str,
) -> None:
    payload: dict[str, Any] = {
        "sample_id": data["sample_id"].astype(object),
        "subject_id": data["subject_id"].astype(object),
        "event_id": data["event_id"].astype(object),
        "session_id": data["session_id"].astype(object),
        "target": data["target"].astype(np.float32),
        "metadata": np.asarray(
            [
                json.dumps(
                    {
                        "target_label": target_label,
                        "fold_strategy": fold_strategy,
                        "representation_source": "out_of_fold_test_rows",
                    },
                    ensure_ascii=False,
                )
            ],
            dtype=object,
        ),
    }
    for name, experiment in experiments.items():
        if "_oof_representation" in experiment:
            payload[f"repr__{name}"] = experiment["_oof_representation"].astype(np.float32)
        if "_oof_prediction" in experiment:
            payload[f"pred__{name}"] = experiment["_oof_prediction"].astype(np.float32)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)


def _strip_private_arrays(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_private_arrays(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, list):
        return [_strip_private_arrays(item) for item in value]
    return value


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| experiment | lambda | adapter | subject_grl | session_grl | train_emb | rows | RMSE mean +/- std | Pearson r mean +/- std | pred_std mean +/- std |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, experiment in result["experiments"].items():
        rows.append(
            "| {name} | {lam} | {adapter} | {subject} | {session} | {train} | {rows} | {rmse} | {r} | {pred_std} |".format(
                name=name,
                lam=_format_metric(experiment.get("grl_lambda")),
                adapter="yes" if experiment.get("adapter_enabled") else "no",
                subject="yes" if experiment.get("subject_grl_enabled") else "no",
                session="yes" if experiment.get("session_grl_enabled") else "no",
                train=experiment.get("train_embedding_key") or "A0",
                rows=experiment.get("row_count", 0),
                rmse=_format_pair(experiment.get("rmse_mean"), experiment.get("rmse_std")),
                r=_format_pair(experiment.get("pearson_r_mean"), experiment.get("pearson_r_std")),
                pred_std=_format_pair(experiment.get("pred_std_mean"), experiment.get("pred_std_std")),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _format_pair(mean: Any, std: Any) -> str:
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} +/- {0.0 if std is None else float(std):.4f}"


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def _lambda_suffix(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


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


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("video GRL adapter ablation requires PyTorch") from exc
    return torch
