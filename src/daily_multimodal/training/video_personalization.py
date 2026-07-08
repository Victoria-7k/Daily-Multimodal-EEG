from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def run_video_personalization(
    *,
    fold_report: Path | str,
    variant: str = "B1",
    out_json: Path | str,
    out_table: Path | str,
    k_events: Iterable[int] = (1, 3, 5),
    include_affine: bool = False,
    affine_slope_penalty: float = 10.0,
    affine_bias_penalty: float = 10.0,
) -> dict[str, Any]:
    rows = _load_event_predictions(fold_report, variant=variant)
    by_subject = _group_by_subject(rows)
    protocols = [_protocol_zero_shot(by_subject)]
    for k in k_events:
        protocols.append(_protocol_k_event(by_subject, k=int(k)))
    protocols.append(_protocol_one_session(by_subject))
    if include_affine:
        for k in k_events:
            protocols.append(
                _protocol_affine_k_event(
                    by_subject,
                    k=int(k),
                    slope_penalty=float(affine_slope_penalty),
                    bias_penalty=float(affine_bias_penalty),
                )
            )
        protocols.append(
            _protocol_affine_one_session(
                by_subject,
                slope_penalty=float(affine_slope_penalty),
                bias_penalty=float(affine_bias_penalty),
            )
        )
    result = {
        "fold_report": str(fold_report),
        "variant": variant,
        "event_count": int(len(rows)),
        "subject_count": int(len(by_subject)),
        "protocol_note": "All calibration/test disjoint protocols use held-out subject test predictions only and estimate a residual bias from calibration events.",
        "affine_regularization": {
            "enabled": bool(include_affine),
            "slope_prior": 1.0,
            "bias_prior": 0.0,
            "slope_penalty": float(affine_slope_penalty),
            "bias_penalty": float(affine_bias_penalty),
        },
        "protocols": protocols,
    }
    _write_json(result, out_json)
    _write_table(result, out_table)
    return result


def _load_event_predictions(fold_report: Path | str, *, variant: str) -> list[dict[str, Any]]:
    report = json.loads(Path(fold_report).read_text(encoding="utf-8"))
    experiment = report.get("experiments", {}).get(variant)
    if experiment is None:
        raise ValueError(f"{fold_report} missing experiment {variant!r}")
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for fold in experiment.get("folds", []):
        fold_name = str(fold.get("fold", "fold"))
        for sample_id, pred, target in zip(
            fold.get("test_sample_ids", []),
            fold.get("test_predictions", []),
            fold.get("test_targets", []),
        ):
            sample = str(sample_id)
            subject = _subject_from_id(sample)
            event = _event_from_sample_id(sample)
            session = _session_from_id(sample)
            key = (fold_name, subject, event)
            bucket = buckets.setdefault(
                key,
                {"fold": fold_name, "subject_id": subject, "event_id": event, "session_id": session, "sample_ids": [], "pred": [], "target": []},
            )
            bucket["sample_ids"].append(sample)
            bucket["pred"].append(float(pred))
            bucket["target"].append(float(target))
    rows = []
    for bucket in buckets.values():
        pred = np.asarray(bucket["pred"], dtype=float)
        target = np.asarray(bucket["target"], dtype=float)
        rows.append(
            {
                "fold": bucket["fold"],
                "subject_id": bucket["subject_id"],
                "event_id": bucket["event_id"],
                "session_id": bucket["session_id"],
                "sample_ids": bucket["sample_ids"],
                "prediction": _float(pred.mean()),
                "target": _float(target.mean()),
                "window_count": int(len(pred)),
            }
        )
    rows.sort(key=lambda row: (row["subject_id"], row["session_id"], row["event_id"]))
    return rows


def _group_by_subject(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row["subject_id"]), []).append(row)
    return out


def _protocol_zero_shot(by_subject: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    subject_results = []
    for subject, rows in by_subject.items():
        subject_results.append(_evaluate_subject(subject, rows, calibration_rows=[], test_rows=rows, correction=0.0))
    return _summarize_protocol("0-shot", 0, subject_results, [])


def _protocol_k_event(by_subject: dict[str, list[dict[str, Any]]], *, k: int) -> dict[str, Any]:
    subject_results = []
    skipped = []
    for subject, rows in by_subject.items():
        ordered = sorted(rows, key=lambda row: (row["session_id"], row["event_id"]))
        if len(ordered) <= k:
            skipped.append({"subject_id": subject, "reason": "not_enough_events", "event_count": len(ordered), "required_calibration_events": k})
            continue
        calibration = ordered[:k]
        test = ordered[k:]
        correction = _residual_bias(calibration)
        subject_results.append(_evaluate_subject(subject, ordered, calibration_rows=calibration, test_rows=test, correction=correction))
    return _summarize_protocol(f"k_event_{k}", k, subject_results, skipped)


def _protocol_one_session(by_subject: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    subject_results = []
    skipped = []
    for subject, rows in by_subject.items():
        sessions = list(dict.fromkeys(row["session_id"] for row in sorted(rows, key=lambda row: (row["session_id"], row["event_id"]))))
        if len(sessions) < 2:
            skipped.append({"subject_id": subject, "reason": "not_enough_sessions", "session_count": len(sessions)})
            continue
        calibration_session = sessions[0]
        calibration = [row for row in rows if row["session_id"] == calibration_session]
        test = [row for row in rows if row["session_id"] != calibration_session]
        if not calibration or not test:
            skipped.append({"subject_id": subject, "reason": "empty_calibration_or_test", "session_count": len(sessions)})
            continue
        correction = _residual_bias(calibration)
        subject_results.append(_evaluate_subject(subject, rows, calibration_rows=calibration, test_rows=test, correction=correction))
    return _summarize_protocol("1-session", 1, subject_results, skipped)


def _protocol_affine_k_event(
    by_subject: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    slope_penalty: float,
    bias_penalty: float,
) -> dict[str, Any]:
    subject_results = []
    skipped = []
    for subject, rows in by_subject.items():
        ordered = sorted(rows, key=lambda row: (row["session_id"], row["event_id"]))
        if len(ordered) <= k:
            skipped.append({"subject_id": subject, "reason": "not_enough_events", "event_count": len(ordered), "required_calibration_events": k})
            continue
        calibration = ordered[:k]
        test = ordered[k:]
        slope, bias = _fit_regularized_affine(calibration, slope_penalty=slope_penalty, bias_penalty=bias_penalty)
        subject_results.append(_evaluate_subject_affine(subject, ordered, calibration_rows=calibration, test_rows=test, slope=slope, bias=bias))
    return _summarize_protocol(f"affine_k_event_{k}", k, subject_results, skipped)


def _protocol_affine_one_session(
    by_subject: dict[str, list[dict[str, Any]]],
    *,
    slope_penalty: float,
    bias_penalty: float,
) -> dict[str, Any]:
    subject_results = []
    skipped = []
    for subject, rows in by_subject.items():
        sessions = list(dict.fromkeys(row["session_id"] for row in sorted(rows, key=lambda row: (row["session_id"], row["event_id"]))))
        if len(sessions) < 2:
            skipped.append({"subject_id": subject, "reason": "not_enough_sessions", "session_count": len(sessions)})
            continue
        calibration_session = sessions[0]
        calibration = [row for row in rows if row["session_id"] == calibration_session]
        test = [row for row in rows if row["session_id"] != calibration_session]
        if not calibration or not test:
            skipped.append({"subject_id": subject, "reason": "empty_calibration_or_test", "session_count": len(sessions)})
            continue
        slope, bias = _fit_regularized_affine(calibration, slope_penalty=slope_penalty, bias_penalty=bias_penalty)
        subject_results.append(_evaluate_subject_affine(subject, rows, calibration_rows=calibration, test_rows=test, slope=slope, bias=bias))
    return _summarize_protocol("affine_1-session", 1, subject_results, skipped)


def _residual_bias(rows: list[dict[str, Any]]) -> float:
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    target = np.asarray([row["target"] for row in rows], dtype=float)
    return _float((target - pred).mean())


def _fit_regularized_affine(
    rows: list[dict[str, Any]],
    *,
    slope_penalty: float,
    bias_penalty: float,
) -> tuple[float, float]:
    pred = np.asarray([row["prediction"] for row in rows], dtype=float)
    target = np.asarray([row["target"] for row in rows], dtype=float)
    x = np.stack([pred, np.ones_like(pred)], axis=1)
    penalty = np.diag([max(0.0, float(slope_penalty)), max(0.0, float(bias_penalty))])
    prior = np.asarray([1.0, 0.0], dtype=float)
    lhs = x.T @ x + penalty
    rhs = x.T @ target + penalty @ prior
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(lhs) @ rhs
    return _float(solution[0]), _float(solution[1])


def _evaluate_subject(
    subject: str,
    all_rows: list[dict[str, Any]],
    *,
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    correction: float,
) -> dict[str, Any]:
    pred = np.asarray([row["prediction"] + correction for row in test_rows], dtype=float)
    target = np.asarray([row["target"] for row in test_rows], dtype=float)
    error = pred - target
    return {
        "subject_id": subject,
        "event_count": int(len(all_rows)),
        "calibration_event_ids": [row["event_id"] for row in calibration_rows],
        "test_event_ids": [row["event_id"] for row in test_rows],
        "test_count": int(len(test_rows)),
        "residual_bias": _float(correction),
        "rmse": _float(np.sqrt(np.mean(error**2))),
        "pearson_r": _pearson(pred, target),
    }


def _evaluate_subject_affine(
    subject: str,
    all_rows: list[dict[str, Any]],
    *,
    calibration_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    slope: float,
    bias: float,
) -> dict[str, Any]:
    pred = np.asarray([slope * row["prediction"] + bias for row in test_rows], dtype=float)
    target = np.asarray([row["target"] for row in test_rows], dtype=float)
    error = pred - target
    return {
        "subject_id": subject,
        "event_count": int(len(all_rows)),
        "calibration_event_ids": [row["event_id"] for row in calibration_rows],
        "test_event_ids": [row["event_id"] for row in test_rows],
        "test_count": int(len(test_rows)),
        "affine_slope": _float(slope),
        "affine_bias": _float(bias),
        "rmse": _float(np.sqrt(np.mean(error**2))),
        "pearson_r": _pearson(pred, target),
    }


def _summarize_protocol(
    name: str,
    calibration_count: int,
    subject_results: list[dict[str, Any]],
    skipped_subjects: list[dict[str, Any]],
) -> dict[str, Any]:
    rmse_values = np.asarray([row["rmse"] for row in subject_results], dtype=float)
    r_values = np.asarray([row["pearson_r"] for row in subject_results if row.get("pearson_r") is not None], dtype=float)
    return {
        "protocol": name,
        "calibration_count": int(calibration_count),
        "eligible_subjects": int(len(subject_results)),
        "skipped_subjects": skipped_subjects,
        "rmse_mean": _optional_mean(rmse_values),
        "rmse_std": _optional_std(rmse_values),
        "pearson_r_mean": _optional_mean(r_values),
        "pearson_r_std": _optional_std(r_values),
        "subjects": subject_results,
    }


def _subject_from_id(value: str) -> str:
    match = re.search(r"(sub-[^_]+)", value)
    return match.group(1) if match else "unknown-subject"


def _session_from_id(value: str) -> str:
    match = re.search(r"(sub-[^_]+)_+(ses-[^_]+)", value)
    return f"{match.group(1)}_{match.group(2)}" if match else f"{_subject_from_id(value)}_unknown-session"


def _event_from_sample_id(value: str) -> str:
    match = re.match(r"(.+?)_win-\d+$", value)
    return match.group(1) if match else value


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return _float(np.corrcoef(a, b)[0, 1])


def _optional_mean(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else _float(values.mean())


def _optional_std(values: np.ndarray) -> float | None:
    return None if len(values) == 0 else _float(values.std())


def _float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite personalization metric: {out}")
    return out


def _write_json(result: dict[str, Any], output: Path | str) -> None:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _write_table(result: dict[str, Any], output: Path | str) -> None:
    rows = [
        "# Video Few-Shot Personalization",
        "",
        result["protocol_note"],
        "",
        "| protocol | calibration count | eligible subjects | skipped subjects | RMSE mean +/- std | Pearson r mean +/- std |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["protocols"]:
        rows.append(
            "| {protocol} | {calibration_count} | {eligible_subjects} | {skipped} | {rmse} | {r} |".format(
                protocol=row["protocol"],
                calibration_count=row["calibration_count"],
                eligible_subjects=row["eligible_subjects"],
                skipped=len(row["skipped_subjects"]),
                rmse=_fmt_pair(row["rmse_mean"], row["rmse_std"]),
                r=_fmt_pair(row["pearson_r_mean"], row["pearson_r_std"]),
            )
        )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _fmt_pair(mean: Any, std: Any) -> str:
    if mean is None:
        return "NA"
    return f"{float(mean):.4f} +/- {float(std or 0.0):.4f}"
