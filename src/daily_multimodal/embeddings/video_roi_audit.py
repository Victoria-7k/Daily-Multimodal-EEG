from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROI_AUDIT_FEATURE_NAMES = [
    "roi_area_ratio",
    "roi_center_x",
    "roi_center_y",
    "face_roi_ratio",
    "fallback_ratio",
    "brightness_mean",
]
SUMMARY_COLUMNS = [
    "subject_id",
    "session_id",
    "window_count",
    "roi_area_ratio_mean",
    "roi_area_ratio_std",
    "roi_center_x_mean",
    "roi_center_x_std",
    "roi_center_y_mean",
    "roi_center_y_std",
    "face_roi_ratio_mean",
    "face_roi_ratio_std",
    "fallback_ratio",
    "brightness_mean",
]
FrameSizeReader = Callable[[Path], tuple[int, int] | None]
BrightnessReader = Callable[[Path], float | None]


def run_video_roi_audit(
    *,
    window_index_path: Path | str,
    region_cache_root: Path | str,
    out_csv: Path | str,
    out_probe_json: Path | str,
    out_summary_md: Path | str,
    video_region: str = "upper_body",
    seed: int = 41,
    n_splits: int = 5,
    min_probe_windows_per_session: int = 2,
    frame_size_reader: FrameSizeReader | None = None,
    brightness_reader: BrightnessReader | None = None,
) -> dict[str, Any]:
    windows = _read_jsonl(Path(window_index_path))
    region_root = Path(region_cache_root)
    frame_size = frame_size_reader or _video_frame_size
    brightness = brightness_reader or _video_brightness_mean
    window_rows = [
        row
        for row in (
            _window_metric_row(
                window,
                region_root=region_root,
                video_region=video_region,
                frame_size_reader=frame_size,
                brightness_reader=brightness,
            )
            for window in windows
        )
        if row is not None
    ]
    session_rows = _session_summary_rows(window_rows)
    probe = _geometry_session_probe(
        window_rows,
        seed=seed,
        n_splits=n_splits,
        min_windows_per_session=min_probe_windows_per_session,
    )
    diagnostics = _diagnostics(session_rows, probe)
    _write_session_csv(session_rows, Path(out_csv))
    _write_json(probe, Path(out_probe_json))
    _write_summary_md(
        Path(out_summary_md),
        session_rows=session_rows,
        probe=probe,
        diagnostics=diagnostics,
    )
    return {
        "window_count": int(len(window_rows)),
        "session_count": int(len(session_rows)),
        "probe": probe,
        "diagnostics": diagnostics,
        "out_csv": str(out_csv),
        "out_probe_json": str(out_probe_json),
        "out_summary_md": str(out_summary_md),
    }


def _window_metric_row(
    window: dict[str, Any],
    *,
    region_root: Path,
    video_region: str,
    frame_size_reader: FrameSizeReader,
    brightness_reader: BrightnessReader,
) -> dict[str, Any] | None:
    sample_id = str(window.get("sample_id", ""))
    sidecar_path = region_root / video_region / sample_id / "region.json"
    if not sidecar_path.is_file():
        return None
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_video = Path(str(sidecar.get("source_video_path") or ""))
    output_video = Path(str(sidecar.get("output_video_path") or (sidecar_path.parent / "window.mp4")))
    size = frame_size_reader(source_video)
    if size is None:
        return None
    frame_width, frame_height = int(size[0]), int(size[1])
    if frame_width <= 0 or frame_height <= 0:
        return None
    fallback = bool(sidecar.get("upper_body_fallback_full_frame")) or str(sidecar.get("effective_region", "")) == "full_frame"
    crop_bbox = None if fallback else _coerce_bbox(sidecar.get("crop_bbox"))
    roi_bbox = _full_frame_bbox(frame_width, frame_height) if crop_bbox is None else _clamp_bbox(crop_bbox, frame_width, frame_height)
    roi_area = _bbox_area(roi_bbox)
    frame_area = float(frame_width * frame_height)
    face_bbox = _face_bbox_from_window(window)
    face_area = 0.0 if face_bbox is None else _bbox_area(_clamp_bbox(face_bbox, frame_width, frame_height))
    return {
        "sample_id": sample_id,
        "event_id": str(window.get("event_id", "")),
        "subject_id": str(window.get("subject_id", "")),
        "session_id": _session_id(str(window.get("subject_id", "")), str(window.get("event_id", "")), sample_id),
        "roi_area_ratio": _safe_ratio(roi_area, frame_area),
        "roi_center_x": _safe_ratio((roi_bbox[0] + roi_bbox[2]) / 2.0, float(frame_width)),
        "roi_center_y": _safe_ratio((roi_bbox[1] + roi_bbox[3]) / 2.0, float(frame_height)),
        "face_roi_ratio": _safe_ratio(face_area, roi_area),
        "fallback_ratio": 1.0 if fallback else 0.0,
        "brightness_mean": brightness_reader(output_video),
    }


def _session_summary_rows(window_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[(str(row["subject_id"]), str(row["session_id"]))].append(row)
    summary_rows = []
    for (subject_id, session_id), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "subject_id": subject_id,
                "session_id": session_id,
                "window_count": int(len(rows)),
                "roi_area_ratio_mean": _mean(rows, "roi_area_ratio"),
                "roi_area_ratio_std": _std(rows, "roi_area_ratio"),
                "roi_center_x_mean": _mean(rows, "roi_center_x"),
                "roi_center_x_std": _std(rows, "roi_center_x"),
                "roi_center_y_mean": _mean(rows, "roi_center_y"),
                "roi_center_y_std": _std(rows, "roi_center_y"),
                "face_roi_ratio_mean": _mean(rows, "face_roi_ratio"),
                "face_roi_ratio_std": _std(rows, "face_roi_ratio"),
                "fallback_ratio": _mean(rows, "fallback_ratio"),
                "brightness_mean": _mean(rows, "brightness_mean"),
            }
        )
    return summary_rows


def _geometry_session_probe(
    window_rows: list[dict[str, Any]],
    *,
    seed: int,
    n_splits: int,
    min_windows_per_session: int,
) -> dict[str, Any]:
    subject_results = []
    for subject_id in sorted({str(row["subject_id"]) for row in window_rows}):
        rows = [row for row in window_rows if str(row["subject_id"]) == subject_id]
        sessions = sorted({str(row["session_id"]) for row in rows})
        if len(sessions) < 2:
            continue
        counts = {session: sum(str(row["session_id"]) == session for row in rows) for session in sessions}
        valid_sessions = [session for session in sessions if counts[session] >= int(min_windows_per_session)]
        if len(valid_sessions) < 2:
            continue
        filtered = [row for row in rows if str(row["session_id"]) in set(valid_sessions)]
        x = _feature_matrix(filtered)
        y = np.asarray([str(row["session_id"]) for row in filtered], dtype=str)
        result = _classification_probe(x, y, seed=seed, n_splits=n_splits)
        if "failure" in result:
            continue
        subject_results.append(
            {
                **result,
                "subject_id": subject_id,
                "window_count": int(len(filtered)),
                "session_count": int(len(valid_sessions)),
                "sessions": valid_sessions,
            }
        )
    accuracies = np.asarray([row["accuracy_mean"] for row in subject_results], dtype=np.float32)
    f1s = np.asarray([row["macro_f1_mean"] for row in subject_results], dtype=np.float32)
    return {
        "probe": "geometry_only_within_subject_session_logreg",
        "feature_names": ROI_AUDIT_FEATURE_NAMES,
        "subject_count": int(len(subject_results)),
        "accuracy_mean": None if accuracies.size == 0 else float(np.mean(accuracies)),
        "accuracy_std": None if accuracies.size == 0 else float(np.std(accuracies)),
        "macro_f1_mean": None if f1s.size == 0 else float(np.mean(f1s)),
        "macro_f1_std": None if f1s.size == 0 else float(np.std(f1s)),
        "subjects": subject_results,
    }


def _classification_probe(x: np.ndarray, y: np.ndarray, *, seed: int, n_splits: int) -> dict[str, Any]:
    split_count = _stratified_split_count(y, n_splits)
    if split_count < 2:
        return {"failure": "not enough samples per session", "class_count": int(len(set(y.tolist())))}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        cv = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=seed)
        folds = []
        accs = []
        f1s = []
        for index, (train, test) in enumerate(cv.split(x, y)):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
            )
            model.fit(x[train], y[train])
            pred = model.predict(x[test])
            acc = float(accuracy_score(y[test], pred))
            f1 = float(f1_score(y[test], pred, average="macro"))
            accs.append(acc)
            f1s.append(f1)
            folds.append({"fold": int(index), "test_count": int(len(test)), "accuracy": acc, "macro_f1": f1})
    except ImportError:
        folds, accs, f1s = _centroid_probe(x, y, split_count=split_count, seed=seed)
    return {
        "class_count": int(len(set(y.tolist()))),
        "fold_count": int(split_count),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "folds": folds,
    }


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    matrix = []
    for row in rows:
        matrix.append([_finite_or_zero(row.get(name)) for name in ROI_AUDIT_FEATURE_NAMES])
    return np.asarray(matrix, dtype=np.float32)


def _centroid_probe(x: np.ndarray, y: np.ndarray, *, split_count: int, seed: int) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    folds = []
    accs = []
    f1s = []
    for index, (train, test) in enumerate(_stratified_folds(y, split_count, seed=seed)):
        train_x, test_x = _standardize(x[train], x[test])
        classes = sorted(set(y[train].tolist()))
        centroids = np.stack([train_x[y[train] == value].mean(axis=0) for value in classes])
        distances = np.linalg.norm(test_x[:, None, :] - centroids[None, :, :], axis=2)
        pred = np.asarray([classes[item] for item in np.argmin(distances, axis=1)], dtype=str)
        acc = float(np.mean(pred == y[test]))
        f1 = _macro_f1(y[test], pred)
        accs.append(acc)
        f1s.append(f1)
        folds.append({"fold": int(index), "test_count": int(len(test)), "accuracy": acc, "macro_f1": f1})
    return folds, accs, f1s


def _diagnostics(session_rows: list[dict[str, Any]], probe: dict[str, Any]) -> dict[str, Any]:
    scale_ranges = _subject_ranges(session_rows, "roi_area_ratio_mean")
    center_x_ranges = _subject_ranges(session_rows, "roi_center_x_mean")
    center_y_ranges = _subject_ranges(session_rows, "roi_center_y_mean")
    face_ranges = _subject_ranges(session_rows, "face_roi_ratio_mean")
    fallback_ranges = _subject_ranges(session_rows, "fallback_ratio")
    return {
        "roi_scale_changed": bool(scale_ranges and max(scale_ranges.values()) >= 0.10),
        "roi_center_or_face_ratio_drifted": bool(
            (center_x_ranges and max(center_x_ranges.values()) >= 0.10)
            or (center_y_ranges and max(center_y_ranges.values()) >= 0.10)
            or (face_ranges and max(face_ranges.values()) >= 0.08)
        ),
        "geometry_probe_high": probe.get("accuracy_mean") is not None and float(probe["accuracy_mean"]) >= 0.70,
        "max_roi_area_ratio_range": None if not scale_ranges else float(max(scale_ranges.values())),
        "max_roi_center_x_range": None if not center_x_ranges else float(max(center_x_ranges.values())),
        "max_roi_center_y_range": None if not center_y_ranges else float(max(center_y_ranges.values())),
        "max_face_roi_ratio_range": None if not face_ranges else float(max(face_ranges.values())),
        "max_fallback_ratio_range": None if not fallback_ranges else float(max(fallback_ranges.values())),
    }


def _write_session_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in SUMMARY_COLUMNS})


def _write_summary_md(
    path: Path,
    *,
    session_rows: list[dict[str, Any]],
    probe: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    lines = [
        "# ROI Geometry Audit Summary",
        "",
        f"- Session rows: {len(session_rows)}",
        f"- Geometry-only session probe accuracy: {_format_metric(probe.get('accuracy_mean'))}",
        f"- Geometry-only session probe subjects: {probe.get('subject_count', 0)}",
        "",
        "## 1. Same-subject ROI scale change",
        _answer("ROI scale changed", diagnostics["roi_scale_changed"], diagnostics.get("max_roi_area_ratio_range")),
        "",
        "## 2. ROI center / face_roi_ratio drift",
        _answer(
            "ROI center or face_roi_ratio drifted",
            diagnostics["roi_center_or_face_ratio_drifted"],
            diagnostics.get("max_face_roi_ratio_range"),
        ),
        "",
        "## 3. Geometry-only Session Probe",
        _answer("Geometry-only Session Probe is high", diagnostics["geometry_probe_high"], probe.get("accuracy_mean")),
        "",
        "## Decision",
        _decision(diagnostics),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _answer(label: str, flag: bool, value: Any) -> str:
    return f"- {label}: {'YES' if flag else 'NO'}. Evidence value: {_format_metric(value)}."


def _decision(diagnostics: dict[str, Any]) -> str:
    if diagnostics["geometry_probe_high"]:
        return "ROI geometry alone can identify sessions, so ROI stabilization should be prioritized."
    if diagnostics["roi_scale_changed"] or diagnostics["roi_center_or_face_ratio_drifted"]:
        return "ROI geometry drifts enough to warrant stabilization/quality control, but the low geometry-only probe suggests appearance augmentation remains the primary shortcut-reduction priority."
    return "Geometry does not appear to be the dominant shortcut; prioritize appearance augmentation and session appearance robustness."


def _subject_ranges(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            by_subject[str(row["subject_id"])].append(number)
    return {
        subject: float(max(values) - min(values))
        for subject, values in by_subject.items()
        if len(values) >= 2
    }


def _video_frame_size(path: Path) -> tuple[int, int] | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    return (width, height) if width > 0 and height > 0 else None


def _video_brightness_mean(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    values = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            values.append(float(frame.mean()))
    finally:
        cap.release()
    return None if not values else float(np.mean(values))


def _face_bbox_from_window(window: dict[str, Any]) -> list[int] | None:
    for container in (window, window.get("face_presence") if isinstance(window.get("face_presence"), dict) else {}):
        bbox = _coerce_face_bbox(container.get("main_face_bbox") or container.get("face_bbox"))
        if bbox is not None:
            return bbox
    return None


def _coerce_face_bbox(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x", "y", "w", "h")):
            x = _number(value["x"])
            y = _number(value["y"])
            return [int(round(x)), int(round(y)), int(round(x + _number(value["w"]))), int(round(y + _number(value["h"])))]
        return _coerce_bbox(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x, y, w, h = [_number(item) for item in value[:4]]
        return [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]
    return None


def _coerce_bbox(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            raw = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("x", "y", "w", "h")):
            x = _number(value["x"])
            y = _number(value["y"])
            raw = [x, y, x + _number(value["w"]), y + _number(value["h"])]
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw = list(value[:4])
    else:
        return None
    return [int(round(_number(item))) for item in raw]


def _clamp_bbox(bbox: list[int], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    left = max(0.0, min(float(width - 1), float(x1)))
    top = max(0.0, min(float(height - 1), float(y1)))
    right = max(left + 1.0, min(float(width), float(x2)))
    bottom = max(top + 1.0, min(float(height), float(y2)))
    return [left, top, right, bottom]


def _full_frame_bbox(width: int, height: int) -> list[float]:
    return [0.0, 0.0, float(width), float(height)]


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return None if values.size == 0 else float(np.mean(values))


def _std(rows: list[dict[str, Any]], key: str) -> float | None:
    values = _values(rows, key)
    return None if values.size == 0 else float(np.std(values))


def _values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return np.asarray(values, dtype=np.float32)


def _finite_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _stratified_split_count(y: np.ndarray, n_splits: int) -> int:
    counts = [int(np.sum(y == value)) for value in set(y.tolist())]
    return 0 if not counts else min(int(n_splits), min(counts))


def _stratified_folds(y: np.ndarray, n_splits: int, *, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    by_class = []
    for value in sorted(set(y.tolist())):
        indices = np.flatnonzero(y == value)
        rng.shuffle(indices)
        by_class.append(np.array_split(indices, n_splits))
    all_indices = np.arange(len(y), dtype=np.int64)
    folds = []
    for fold_index in range(n_splits):
        test = np.concatenate([parts[fold_index] for parts in by_class]).astype(np.int64)
        train = np.setdiff1d(all_indices, test, assume_unique=False).astype(np.int64)
        folds.append((train, test))
    return folds


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (train_x - mean) / std, (test_x - mean) / std


def _macro_f1(truth: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for value in sorted(set(truth.tolist()) | set(pred.tolist())):
        tp = float(np.sum((truth == value) & (pred == value)))
        fp = float(np.sum((truth != value) & (pred == value)))
        fn = float(np.sum((truth == value) & (pred != value)))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        scores.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def _session_id(subject_id: str, event_id: str, sample_id: str) -> str:
    for value in (event_id, sample_id):
        match = re.search(r"(sub-[^_]+)_+(ses-[^_]+)", value)
        if match:
            return f"{match.group(1)}_{match.group(2)}"
    return f"{subject_id}_unknown-session"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


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


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if not math.isfinite(value) else f"{value:.6f}"
    return str(value)


def _format_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(number) else f"{number:.4f}"
