from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_REQUIRED_SPLITS = ("LOSO", "S4", "S2")


def evaluate_v4d_success(
    *,
    baseline_probe: Mapping[str, Any],
    candidate_probe: Mapping[str, Any],
    baseline_variants: Mapping[str, Mapping[str, Any]],
    candidate_variants: Mapping[str, Mapping[str, Any]],
    variant_name: str,
    required_splits: tuple[str, ...] = DEFAULT_REQUIRED_SPLITS,
    min_probe_drop: float = 0.0,
    rmse_tolerance: float = 0.0,
    pearson_tolerance: float = 0.0,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    checks["subject_probe_drop"] = _probe_drop_check(
        baseline_probe,
        candidate_probe,
        probe_name="P1_subject_logreg",
        min_drop=min_probe_drop,
    )
    checks["session_probe_drop"] = _probe_drop_check(
        baseline_probe,
        candidate_probe,
        probe_name="P2_within_subject_session_logreg",
        min_drop=min_probe_drop,
    )
    for split in required_splits:
        checks[f"fatigue_{split}_no_regression"] = _fatigue_check(
            baseline_variants,
            candidate_variants,
            split=split,
            variant_name=variant_name,
            rmse_tolerance=rmse_tolerance,
            pearson_tolerance=pearson_tolerance,
        )
    return {
        "variant_name": variant_name,
        "required_splits": list(required_splits),
        "thresholds": {
            "min_probe_drop": float(min_probe_drop),
            "rmse_tolerance": float(rmse_tolerance),
            "pearson_tolerance": float(pearson_tolerance),
        },
        "passed": all(bool(check.get("passed")) for check in checks.values()),
        "checks": checks,
    }


def evaluate_v4d_success_from_files(
    *,
    baseline_probe_path: Path | str,
    candidate_probe_path: Path | str,
    baseline_variant_paths: Mapping[str, Path | str],
    candidate_variant_paths: Mapping[str, Path | str],
    variant_name: str,
    required_splits: tuple[str, ...] = DEFAULT_REQUIRED_SPLITS,
    min_probe_drop: float = 0.0,
    rmse_tolerance: float = 0.0,
    pearson_tolerance: float = 0.0,
) -> dict[str, Any]:
    return evaluate_v4d_success(
        baseline_probe=_read_json(baseline_probe_path),
        candidate_probe=_read_json(candidate_probe_path),
        baseline_variants={split: _read_json(path) for split, path in baseline_variant_paths.items()},
        candidate_variants={split: _read_json(path) for split, path in candidate_variant_paths.items()},
        variant_name=variant_name,
        required_splits=required_splits,
        min_probe_drop=min_probe_drop,
        rmse_tolerance=rmse_tolerance,
        pearson_tolerance=pearson_tolerance,
    )


def write_v4d_success_report(result: Mapping[str, Any], *, out_json: Path | str, out_table: Path | str) -> None:
    json_path = Path(out_json)
    table_path = Path(out_table)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    table_path.write_text(_markdown_table(result), encoding="utf-8")


def _probe_drop_check(
    baseline_probe: Mapping[str, Any],
    candidate_probe: Mapping[str, Any],
    *,
    probe_name: str,
    min_drop: float,
) -> dict[str, Any]:
    baseline = _nested_float(baseline_probe, "probes", probe_name, "accuracy_mean")
    candidate = _nested_float(candidate_probe, "probes", probe_name, "accuracy_mean")
    delta = None if baseline is None or candidate is None else candidate - baseline
    required_delta = -float(min_drop)
    passed = delta is not None and (delta < 0.0 if min_drop <= 0 else delta <= required_delta)
    return {
        "passed": bool(passed),
        "baseline_accuracy": baseline,
        "candidate_accuracy": candidate,
        "accuracy_delta": delta,
        "required_delta_lte": required_delta,
        "strict_drop_required": bool(min_drop <= 0),
    }


def _fatigue_check(
    baseline_variants: Mapping[str, Mapping[str, Any]],
    candidate_variants: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    variant_name: str,
    rmse_tolerance: float,
    pearson_tolerance: float,
) -> dict[str, Any]:
    baseline_experiment = _experiment_for_split(baseline_variants, split, preferred_name=variant_name)
    candidate_experiment = _experiment_for_split(candidate_variants, split, preferred_name=variant_name)
    baseline_rmse = _safe_float(baseline_experiment.get("rmse_mean"))
    candidate_rmse = _safe_float(candidate_experiment.get("rmse_mean"))
    baseline_r = _safe_float(baseline_experiment.get("pearson_r_mean"))
    candidate_r = _safe_float(candidate_experiment.get("pearson_r_mean"))
    rmse_delta = None if baseline_rmse is None or candidate_rmse is None else candidate_rmse - baseline_rmse
    r_delta = None if baseline_r is None or candidate_r is None else candidate_r - baseline_r
    rmse_ok = rmse_delta is not None and rmse_delta <= float(rmse_tolerance)
    r_ok = r_delta is not None and r_delta >= -float(pearson_tolerance)
    return {
        "passed": bool(rmse_ok and r_ok),
        "split": split,
        "baseline_rmse": baseline_rmse,
        "candidate_rmse": candidate_rmse,
        "rmse_delta": rmse_delta,
        "rmse_delta_lte": float(rmse_tolerance),
        "baseline_pearson_r": baseline_r,
        "candidate_pearson_r": candidate_r,
        "pearson_r_delta": r_delta,
        "pearson_r_delta_gte": -float(pearson_tolerance),
    }


def _experiment_for_split(
    variants_by_split: Mapping[str, Mapping[str, Any]],
    split: str,
    *,
    preferred_name: str,
) -> Mapping[str, Any]:
    result = variants_by_split.get(split)
    if result is None:
        return {}
    experiments = result.get("experiments", {})
    if preferred_name in experiments:
        return experiments[preferred_name]
    if len(experiments) == 1:
        return next(iter(experiments.values()))
    for fallback in ("V4d", "V2", "V1"):
        if fallback in experiments:
            return experiments[fallback]
    return {}


def _nested_float(data: Mapping[str, Any], *keys: str) -> float | None:
    value: Any = data
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return _safe_float(value)


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _markdown_table(result: Mapping[str, Any]) -> str:
    rows = [
        f"# V4d Success Gate: {'PASS' if result.get('passed') else 'FAIL'}",
        "",
        "| check | passed | primary delta | details |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in result.get("checks", {}).items():
        if "accuracy_delta" in check:
            delta = _format_metric(check.get("accuracy_delta"))
            details = "baseline_acc={base} candidate_acc={candidate}".format(
                base=_format_metric(check.get("baseline_accuracy")),
                candidate=_format_metric(check.get("candidate_accuracy")),
            )
        else:
            delta = "rmse {rmse}; r {r}".format(
                rmse=_format_metric(check.get("rmse_delta")),
                r=_format_metric(check.get("pearson_r_delta")),
            )
            details = "split={split} baseline_rmse={base} candidate_rmse={candidate}".format(
                split=check.get("split", ""),
                base=_format_metric(check.get("baseline_rmse")),
                candidate=_format_metric(check.get("candidate_rmse")),
            )
        rows.append(f"| {name} | {bool(check.get('passed'))} | {delta} | {details} |")
    return "\n".join(rows) + "\n"


def _format_metric(value: Any) -> str:
    value = _safe_float(value)
    return "NA" if value is None else f"{value:.4f}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
