from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def analyze_cross_subject_transfer(
    *,
    representations: Path | str,
    variant: str = "B1",
    out_json: Path | str,
    out_table: Path | str,
    ridge_alpha: float = 10.0,
) -> dict[str, Any]:
    data = _load_repr(representations, variant=variant)
    subjects = list(dict.fromkeys(data["subject_id"].astype(str).tolist()))
    centered_target = _subject_centered_target(data["subject_id"], data["target"])
    matrix: dict[str, dict[str, Any]] = {}
    pair_rows = []
    for train_subject in subjects:
        train_mask = data["subject_id"] == train_subject
        model = _fit_ridge(data["x"][train_mask], centered_target[train_mask], alpha=float(ridge_alpha))
        matrix[train_subject] = {}
        for test_subject in subjects:
            test_mask = data["subject_id"] == test_subject
            pred = _predict_ridge(model, data["x"][test_mask])
            target = centered_target[test_mask]
            error = pred - target
            metrics = {
                "train_subject": train_subject,
                "test_subject": test_subject,
                "train_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "pearson_r": _pearson(pred, target),
                "rmse": _float(np.sqrt(np.mean(error**2))),
                "bias": _float(error.mean()),
                "pred_std": _float(pred.std()),
                "target_std": _float(target.std()),
            }
            matrix[train_subject][test_subject] = metrics
            pair_rows.append(metrics)
    result = {
        "representations": str(representations),
        "variant": variant,
        "ridge_alpha": float(ridge_alpha),
        "subject_count": int(len(subjects)),
        "subjects": subjects,
        "matrix": matrix,
        "sign_summary": _sign_summary(pair_rows),
    }
    _write_json(result, out_json)
    _write_table(result, out_table)
    return result


def _load_repr(path: Path | str, *, variant: str) -> dict[str, Any]:
    key = f"repr__{variant}"
    path = Path(path)
    with np.load(path, allow_pickle=True) as loaded:
        required = {"subject_id", "target", key}
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing required arrays: {', '.join(missing)}")
        x = np.asarray(loaded[key], dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"{key} expected shape (N, D), got {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(f"{key} contains non-finite values")
        subject_id = loaded["subject_id"].astype(str)
        target = np.asarray(loaded["target"], dtype=np.float32)
    if len(subject_id) != x.shape[0] or len(target) != x.shape[0]:
        raise ValueError(f"{path} has inconsistent row counts")
    return {"subject_id": subject_id, "target": target, "x": x}


def _subject_centered_target(subject_id: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = np.zeros_like(target, dtype=np.float32)
    for subject in dict.fromkeys(subject_id.astype(str).tolist()):
        mask = subject_id == subject
        centered[mask] = target[mask] - float(target[mask].mean())
    return centered


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0
    x_norm = (x - x_mean) / x_std
    y_mean = float(y.mean())
    y_centered = y - y_mean
    lhs = x_norm.T @ x_norm + float(alpha) * np.eye(x_norm.shape[1], dtype=np.float32)
    rhs = x_norm.T @ y_centered
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(lhs) @ rhs
    return coef.astype(np.float32), y_mean, x_mean.reshape(-1).astype(np.float32), x_std.reshape(-1).astype(np.float32)


def _predict_ridge(model: tuple[np.ndarray, float, np.ndarray, np.ndarray], x: np.ndarray) -> np.ndarray:
    coef, intercept, x_mean, x_std = model
    return ((x - x_mean) / x_std) @ coef + float(intercept)


def _sign_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["pearson_r"] is not None and row["pearson_r"] > 0]
    negative = [row for row in rows if row["pearson_r"] is not None and row["pearson_r"] < 0]
    diagonal = [row for row in rows if row["train_subject"] == row["test_subject"]]
    off_diagonal = [row for row in rows if row["train_subject"] != row["test_subject"]]
    return {
        "positive_pairs": int(len(positive)),
        "negative_pairs": int(len(negative)),
        "diagonal_r_mean": _mean([row["pearson_r"] for row in diagonal if row["pearson_r"] is not None]),
        "off_diagonal_r_mean": _mean([row["pearson_r"] for row in off_diagonal if row["pearson_r"] is not None]),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return _float(np.corrcoef(a, b)[0, 1])


def _mean(values: list[float]) -> float | None:
    return None if not values else _float(np.asarray(values, dtype=float).mean())


def _float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite transfer metric: {out}")
    return out


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    subjects = result["subjects"]
    rows = [
        "# Video Cross-Subject Transfer Matrix",
        "",
        f"Ridge alpha: `{result['ridge_alpha']}`",
        "",
        "## Pearson r",
        "",
        "| train \\ test | " + " | ".join(subjects) + " |",
        "| --- | " + " | ".join(["---:"] * len(subjects)) + " |",
    ]
    for train_subject in subjects:
        values = [_fmt(result["matrix"][train_subject][test_subject]["pearson_r"]) for test_subject in subjects]
        rows.append(f"| {train_subject} | " + " | ".join(values) + " |")
    rows.extend(
        [
            "",
            "## Summary",
            "",
            f"- positive pairs: `{result['sign_summary']['positive_pairs']}`",
            f"- negative pairs: `{result['sign_summary']['negative_pairs']}`",
            f"- diagonal r mean: `{_fmt(result['sign_summary']['diagonal_r_mean'])}`",
            f"- off-diagonal r mean: `{_fmt(result['sign_summary']['off_diagonal_r_mean'])}`",
        ]
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.4f}"
