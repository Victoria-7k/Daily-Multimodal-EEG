from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from daily_multimodal.training.video_embedding_probes import (
    _classification_probe,
    _ridge_probe,
    _within_subject_session_probe,
)
from daily_multimodal.training.video_variant_ablation import _evaluate_predictions, _json_ready


DEFAULT_STABILITY_SPLITS = ("LOSO", "S4", "S2")


def summarize_grl_repeat_stability(
    *,
    report_root: Path | str,
    variants: Sequence[str],
    out_json: Path | str,
    out_table: Path | str,
    splits: Sequence[str] = DEFAULT_STABILITY_SPLITS,
) -> dict[str, Any]:
    root = Path(report_root)
    seed_dirs = sorted(path for path in root.glob("seed_*") if path.is_dir())
    if not seed_dirs:
        raise ValueError(f"no seed_* report directories found under {root}")

    overall: dict[str, dict[str, Any]] = {split: {} for split in splits}
    subject_values: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {
        split: {variant: defaultdict(list) for variant in variants} for split in splits
    }
    report_paths: dict[str, dict[str, str]] = {}
    for seed_dir in seed_dirs:
        seed = _seed_from_dir(seed_dir)
        report_paths[str(seed)] = {}
        for split in splits:
            path = seed_dir / f"{split.lower()}_metrics.json"
            if not path.exists():
                continue
            report_paths[str(seed)][split] = str(path)
            report = json.loads(path.read_text(encoding="utf-8"))
            for variant in variants:
                experiment = report.get("experiments", {}).get(variant)
                if experiment is None:
                    continue
                overall.setdefault(split, {}).setdefault(variant, {"pearson_r": [], "rmse": [], "seeds": []})
                overall[split][variant]["pearson_r"].append(experiment.get("pearson_r_mean"))
                overall[split][variant]["rmse"].append(experiment.get("rmse_mean"))
                overall[split][variant]["seeds"].append(seed)
                for subject, metrics in _subject_metrics_from_folds(experiment.get("folds", [])).items():
                    subject_values[split][variant][subject].append({"seed": seed, **metrics})

    overall_summary = {
        split: {variant: _summarize_overall(values) for variant, values in variants_map.items()}
        for split, variants_map in overall.items()
    }
    subject_summary = {
        split: {
            variant: {subject: _summarize_subject(rows) for subject, rows in sorted(subjects.items())}
            for variant, subjects in variants_map.items()
        }
        for split, variants_map in subject_values.items()
    }
    result = {
        "report_root": str(root),
        "variants": list(variants),
        "splits": list(splits),
        "report_paths": report_paths,
        "overall": overall_summary,
        "subject_metrics": subject_summary,
    }
    _write_json(result, out_json)
    _write_stability_table(result, out_table)
    return _json_ready(result)


def audit_grl_representations(
    *,
    representations: Path | str,
    variants: Sequence[str],
    out_json: Path | str,
    out_table: Path | str,
    ridge_strategies: Sequence[str] = (
        "leave_one_subject_out",
        "within_subject_event_split",
        "within_subject_session_leave_out",
        "within_subject_chronological_split",
    ),
    seed: int = 41,
    n_splits: int = 5,
) -> dict[str, Any]:
    path = Path(representations)
    with np.load(path, allow_pickle=True) as loaded:
        sample_id = loaded["sample_id"].astype(str)
        subject_id = loaded["subject_id"].astype(str)
        event_id = loaded["event_id"].astype(str)
        session_id = loaded["session_id"].astype(str)
        target = loaded["target"].astype(np.float32)
        variant_results = {}
        for variant in variants:
            key = f"repr__{variant}"
            if key not in loaded.files:
                raise ValueError(f"{path} missing {key}")
            x = loaded[key].astype(np.float32)
            valid = np.isfinite(x).all(axis=1)
            pred_key = f"pred__{variant}"
            pred = loaded[pred_key].astype(np.float32) if pred_key in loaded.files else None
            data = {
                "sample_id": sample_id[valid],
                "subject_id": subject_id[valid],
                "event_id": event_id[valid],
                "session_id": session_id[valid],
                "target": target[valid],
                "embedding": x[valid],
            }
            fatigue = {
                strategy: _ridge_probe(data, seed=seed, n_splits=n_splits, fold_strategy=strategy)
                for strategy in ridge_strategies
            }
            variant_results[variant] = {
                "row_count": int(np.sum(valid)),
                "embedding_dim": int(x.shape[1]),
                "subject_probe": _classification_probe(x[valid], subject_id[valid], seed=seed, n_splits=n_splits),
                "session_probe": _within_subject_session_probe(data, seed=seed, n_splits=n_splits),
                "fatigue_ridge": fatigue,
                "embedding_variance": _embedding_variance(x[valid]),
                "embedding_norm": _embedding_norm(x[valid]),
                "prediction_std": None if pred is None else _safe_std(pred[valid]),
            }
    result = {"representations": str(path), "variants": variant_results}
    _write_json(result, out_json)
    _write_representation_table(result, out_table)
    return _json_ready(result)


def _subject_metrics_from_folds(folds: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"pred": [], "target": []})
    for fold in folds:
        for sample_id, pred, target in zip(
            fold.get("test_sample_ids", []),
            fold.get("test_predictions", []),
            fold.get("test_targets", []),
        ):
            subject = _subject_from_sample_id(str(sample_id))
            grouped[subject]["pred"].append(float(pred))
            grouped[subject]["target"].append(float(target))
    out = {}
    for subject, values in grouped.items():
        metrics = _evaluate_predictions(
            np.asarray(values["pred"], dtype=np.float32),
            np.asarray(values["target"], dtype=np.float32),
        )
        error = np.asarray(values["pred"], dtype=np.float32) - np.asarray(values["target"], dtype=np.float32)
        out[subject] = {
            "count": metrics["count"],
            "pearson": metrics["pearson"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias": None if error.size == 0 else float(np.mean(error)),
        }
    return out


def _subject_from_sample_id(sample_id: str) -> str:
    match = re.search(r"(sub-[^_]+)", sample_id)
    return match.group(1) if match else "unknown"


def _seed_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return len(path.name)


def _summarize_overall(values: Mapping[str, list[Any]]) -> dict[str, Any]:
    r = _finite_array(values.get("pearson_r", []))
    rmse = _finite_array(values.get("rmse", []))
    return {
        "seed_count": int(len(values.get("seeds", []))),
        "seeds": [int(seed) for seed in values.get("seeds", [])],
        "pearson_r_mean_mean": None if r.size == 0 else float(np.mean(r)),
        "pearson_r_mean_std": None if r.size == 0 else float(np.std(r)),
        "pearson_r_mean_min": None if r.size == 0 else float(np.min(r)),
        "pearson_r_mean_max": None if r.size == 0 else float(np.max(r)),
        "rmse_mean_mean": None if rmse.size == 0 else float(np.mean(rmse)),
        "rmse_mean_std": None if rmse.size == 0 else float(np.std(rmse)),
    }


def _summarize_subject(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    r = _finite_array([row.get("pearson") for row in rows])
    rmse = _finite_array([row.get("rmse") for row in rows])
    mae = _finite_array([row.get("mae") for row in rows])
    bias = _finite_array([row.get("bias") for row in rows])
    count = _finite_array([row.get("count") for row in rows])
    return {
        "seed_count": int(len(rows)),
        "count_mean": None if count.size == 0 else float(np.mean(count)),
        "pearson_mean": None if r.size == 0 else float(np.mean(r)),
        "pearson_std": None if r.size == 0 else float(np.std(r)),
        "rmse_mean": None if rmse.size == 0 else float(np.mean(rmse)),
        "rmse_std": None if rmse.size == 0 else float(np.std(rmse)),
        "mae_mean": None if mae.size == 0 else float(np.mean(mae)),
        "bias_mean": None if bias.size == 0 else float(np.mean(bias)),
    }


def _embedding_variance(x: np.ndarray) -> dict[str, Any]:
    var = np.var(x, axis=0)
    return {
        "per_dim_mean": float(np.mean(var)),
        "per_dim_std": float(np.std(var)),
        "per_dim_min": float(np.min(var)),
        "per_dim_max": float(np.max(var)),
        "total": float(np.sum(var)),
    }


def _embedding_norm(x: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(x, axis=1)
    return {"mean": float(np.mean(norms)), "std": float(np.std(norms))}


def _safe_std(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return None if finite.size == 0 else float(np.std(finite))


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    out = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            out.append(number)
    return np.asarray(out, dtype=np.float32)


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_stability_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| split | variant | seeds | r mean | r std | r min | r max | rmse mean | rmse std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, variants in result["overall"].items():
        for variant, metrics in variants.items():
            rows.append(
                "| {split} | {variant} | {seeds} | {r_mean} | {r_std} | {r_min} | {r_max} | {rmse_mean} | {rmse_std} |".format(
                    split=split,
                    variant=variant,
                    seeds=metrics.get("seed_count", 0),
                    r_mean=_fmt(metrics.get("pearson_r_mean_mean")),
                    r_std=_fmt(metrics.get("pearson_r_mean_std")),
                    r_min=_fmt(metrics.get("pearson_r_mean_min")),
                    r_max=_fmt(metrics.get("pearson_r_mean_max")),
                    rmse_mean=_fmt(metrics.get("rmse_mean_mean")),
                    rmse_std=_fmt(metrics.get("rmse_mean_std")),
                )
            )
    for split, title in (("LOSO", "LOSO per-subject r"), ("S4", "S4 per-subject r")):
        rows.extend(["", f"## {title}", "", "| variant | subject | seeds | r mean | r std | rmse mean |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for variant, subjects in result["subject_metrics"].get(split, {}).items():
            for subject, metrics in subjects.items():
                rows.append(
                    f"| {variant} | {subject} | {metrics.get('seed_count', 0)} | {_fmt(metrics.get('pearson_mean'))} | {_fmt(metrics.get('pearson_std'))} | {_fmt(metrics.get('rmse_mean'))} |"
                )
    rows.extend(["", "## S2 subject-wise error", "", "| variant | subject | seeds | r mean | rmse mean | mae mean | bias mean |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for variant, subjects in result["subject_metrics"].get("S2", {}).items():
        for subject, metrics in subjects.items():
            rows.append(
                f"| {variant} | {subject} | {metrics.get('seed_count', 0)} | {_fmt(metrics.get('pearson_mean'))} | {_fmt(metrics.get('rmse_mean'))} | {_fmt(metrics.get('mae_mean'))} | {_fmt(metrics.get('bias_mean'))} |"
            )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_representation_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "| variant | dim | Subject Probe | Session Probe | Fatigue Ridge LOSO r | Fatigue Ridge S1 r | Fatigue Ridge S4 r | Fatigue Ridge S2 r | var mean | norm mean | pred std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    strategy_label = {
        "leave_one_subject_out": "LOSO",
        "within_subject_event_split": "S1",
        "within_subject_session_leave_out": "S4",
        "within_subject_chronological_split": "S2",
    }
    for variant, row in result["variants"].items():
        ridge = row.get("fatigue_ridge", {})
        values = {label: ridge.get(strategy, {}).get("pearson_r_mean") for strategy, label in strategy_label.items()}
        rows.append(
            "| {variant} | {dim} | {subject} | {session} | {loso} | {s1} | {s4} | {s2} | {var} | {norm} | {pred} |".format(
                variant=variant,
                dim=row.get("embedding_dim"),
                subject=_fmt(row.get("subject_probe", {}).get("accuracy_mean")),
                session=_fmt(row.get("session_probe", {}).get("accuracy_mean")),
                loso=_fmt(values.get("LOSO")),
                s1=_fmt(values.get("S1")),
                s4=_fmt(values.get("S4")),
                s2=_fmt(values.get("S2")),
                var=_fmt(row.get("embedding_variance", {}).get("per_dim_mean")),
                norm=_fmt(row.get("embedding_norm", {}).get("mean")),
                pred=_fmt(row.get("prediction_std")),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    return "NA" if not math.isfinite(number) else f"{number:.4f}"
