from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def analyze_loso_failure(
    *,
    representations: Path | str,
    fold_report: Path | str | None = None,
    variant: str = "B1",
    out_json: Path | str,
    out_table: Path | str,
) -> dict[str, Any]:
    label_rows = subject_label_distribution(representations)
    prediction_rows = _subject_prediction_metrics(fold_report, variant=variant) if fold_report is not None else []
    group_summary = _prediction_group_summary(prediction_rows)
    result = {
        "representations": str(representations),
        "fold_report": str(fold_report) if fold_report is not None else None,
        "variant": variant,
        "subject_label_distribution": label_rows,
        "subject_prediction_metrics": prediction_rows,
        "prediction_group_summary": group_summary,
        "interpretation": _interpret(group_summary),
    }
    _write_json(result, out_json)
    _write_table(result, out_table)
    return result


def subject_label_distribution(representations: Path | str) -> list[dict[str, Any]]:
    path = Path(representations)
    with np.load(path, allow_pickle=True) as loaded:
        subject_id = loaded["subject_id"].astype(str)
        event_id = loaded["event_id"].astype(str)
        session_id = loaded["session_id"].astype(str) if "session_id" in loaded.files else np.asarray([_session_from_event(e) for e in event_id], dtype=str)
        target = np.asarray(loaded["target"], dtype=float)
    rows = []
    for subject in dict.fromkeys(subject_id.tolist()):
        mask = subject_id == subject
        values = target[mask]
        rows.append(
            {
                "subject_id": str(subject),
                "count": int(mask.sum()),
                "event_count": int(len(set(event_id[mask].tolist()))),
                "session_count": int(len(set(session_id[mask].tolist()))),
                "target_mean": _float(values.mean()),
                "target_std": _float(values.std()),
                "target_min": _float(values.min()),
                "target_max": _float(values.max()),
                "target_range": _float(values.max() - values.min()),
            }
        )
    return rows


def add_subject_centered_label(
    *,
    embeddings: Path | str,
    out: Path | str,
    target_label: str = "fatigue",
    centered_label: str = "fatigue_subject_centered",
) -> dict[str, Any]:
    path = Path(embeddings)
    with np.load(path, allow_pickle=True) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    if "subject_id" not in arrays or "labels" not in arrays:
        raise ValueError(f"{path} must contain subject_id and labels arrays")
    subject_id = arrays["subject_id"].astype(str)
    labels = [_parse_json(value) for value in arrays["labels"].tolist()]
    targets = np.asarray([float(row[target_label]) for row in labels], dtype=float)
    subject_means = {
        subject: float(targets[subject_id == subject].mean())
        for subject in dict.fromkeys(subject_id.tolist())
    }
    centered_labels = []
    for subject, row, target in zip(subject_id, labels, targets):
        updated = dict(row)
        updated[centered_label] = float(target - subject_means[str(subject)])
        centered_labels.append(json.dumps(updated, ensure_ascii=False))
    arrays["labels"] = np.asarray(centered_labels, dtype=object)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    result = {
        "input": str(path),
        "output": str(output),
        "target_label": target_label,
        "centered_label": centered_label,
        "row_count": int(len(subject_id)),
        "subject_means": {key: _float(value) for key, value in subject_means.items()},
        "diagnostic_only": True,
        "warning": "Subject-centered labels use each subject's true label mean and are diagnostic only, not a strict deployment protocol.",
    }
    return result


def _subject_prediction_metrics(fold_report: Path | str, *, variant: str) -> list[dict[str, Any]]:
    report = json.loads(Path(fold_report).read_text(encoding="utf-8"))
    experiment = report.get("experiments", {}).get(variant)
    if experiment is None:
        raise ValueError(f"{fold_report} missing experiment {variant!r}")
    buckets: dict[str, dict[str, list[float] | list[str]]] = {}
    for fold in experiment.get("folds", []):
        for sample_id, pred, target in zip(
            fold.get("test_sample_ids", []),
            fold.get("test_predictions", []),
            fold.get("test_targets", []),
        ):
            subject = _subject_from_sample_id(str(sample_id))
            bucket = buckets.setdefault(subject, {"sample_ids": [], "pred": [], "target": []})
            bucket["sample_ids"].append(str(sample_id))  # type: ignore[union-attr]
            bucket["pred"].append(float(pred))  # type: ignore[union-attr]
            bucket["target"].append(float(target))  # type: ignore[union-attr]
    rows = []
    for subject, bucket in buckets.items():
        pred = np.asarray(bucket["pred"], dtype=float)
        target = np.asarray(bucket["target"], dtype=float)
        error = pred - target
        rows.append(
            {
                "subject_id": subject,
                "count": int(len(pred)),
                "pearson_r": _pearson(pred, target),
                "rmse": _float(np.sqrt(np.mean(error**2))),
                "bias": _float(error.mean()),
                "prediction_std": _float(pred.std()),
                "target_std": _float(target.std()),
                "sample_ids": bucket["sample_ids"],
            }
        )
    rows.sort(key=lambda row: row["subject_id"])
    return rows


def _prediction_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "positive": [row for row in rows if row.get("pearson_r") is not None and row["pearson_r"] > 0],
        "negative": [row for row in rows if row.get("pearson_r") is not None and row["pearson_r"] < 0],
        "zero_or_undefined": [row for row in rows if row.get("pearson_r") is None or row["pearson_r"] == 0],
    }
    return {name: _summarize_group(group) for name, group in groups.items()}


def _summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"subject_count": 0, "subjects": [], "target_std_mean": None, "prediction_std_mean": None, "rmse_mean": None, "bias_mean": None}
    return {
        "subject_count": len(rows),
        "subjects": [row["subject_id"] for row in rows],
        "target_std_mean": _mean_field(rows, "target_std"),
        "prediction_std_mean": _mean_field(rows, "prediction_std"),
        "rmse_mean": _mean_field(rows, "rmse"),
        "bias_mean": _mean_field(rows, "bias"),
    }


def _interpret(group_summary: dict[str, Any]) -> str:
    positive = group_summary.get("positive", {}).get("subject_count", 0)
    negative = group_summary.get("negative", {}).get("subject_count", 0)
    if positive and negative:
        return "LOSO contains mixed positive and negative subject correlations, consistent with subject-dependent fatigue-behavior mapping differences."
    if negative:
        return "LOSO is dominated by negative subject correlations; inspect subject calibration and mapping direction before adding model complexity."
    if positive:
        return "LOSO subject correlations are mostly positive; baseline calibration and label distribution should be checked first."
    return "LOSO subject prediction signs are unavailable; inspect fold-level predictions before drawing a mapping conclusion."


def _mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = np.asarray([row[field] for row in rows if row.get(field) is not None], dtype=float)
    if len(values) == 0:
        return None
    return _float(values.mean())


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return _float(np.corrcoef(a, b)[0, 1])


def _subject_from_sample_id(sample_id: str) -> str:
    match = re.search(r"(sub-[^_]+)", sample_id)
    return match.group(1) if match else "unknown-subject"


def _session_from_event(event_id: str) -> str:
    match = re.search(r"(sub-[^_]+)_+(ses-[^_]+)", event_id)
    return f"{match.group(1)}_{match.group(2)}" if match else "unknown-session"


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise ValueError(f"expected JSON object string, got {type(value).__name__}")


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "# Video LOSO Failure Diagnostics",
        "",
        result["interpretation"],
        "",
        "## Subject Label Distribution",
        "",
        "| subject | rows | events | sessions | mean | std | min | max | range |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["subject_label_distribution"]:
        rows.append(
            "| {subject_id} | {count} | {event_count} | {session_count} | {target_mean:.4f} | {target_std:.4f} | {target_min:.4f} | {target_max:.4f} | {target_range:.4f} |".format(
                **row
            )
        )
    if result["subject_prediction_metrics"]:
        rows.extend(
            [
                "",
                "## Subject Prediction Metrics",
                "",
                "| subject | rows | r | rmse | bias | pred std | target std |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in result["subject_prediction_metrics"]:
            rows.append(
                "| {subject_id} | {count} | {r} | {rmse:.4f} | {bias:.4f} | {prediction_std:.4f} | {target_std:.4f} |".format(
                    subject_id=row["subject_id"],
                    count=row["count"],
                    r=_fmt(row["pearson_r"]),
                    rmse=row["rmse"],
                    bias=row["bias"],
                    prediction_std=row["prediction_std"],
                    target_std=row["target_std"],
                )
            )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


def _float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite metric: {out}")
    return out
