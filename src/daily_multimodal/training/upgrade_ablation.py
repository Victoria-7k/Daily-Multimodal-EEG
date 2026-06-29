from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daily_multimodal.training.baseline_mlp import (
    _evaluate_split,
    _fit_mlp,
    _load_embedding_dataset,
    _run_overfit_check,
    _save_model,
    _subject_split,
)


def snapshot_baseline_reference(
    *,
    embeddings_path: Path | str,
    baseline_metrics_path: Path | str,
    baseline_table_path: Path | str,
    stage8_report_path: Path | str | None,
    metrics_out: Path | str,
    table_out: Path | str,
    manifest_out: Path | str,
    created_at: str | None = None,
) -> dict[str, Any]:
    metrics_path = Path(baseline_metrics_path)
    table_path = Path(baseline_table_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _validate_baseline_metrics(metrics)

    metrics_output = Path(metrics_out)
    table_output = Path(table_out)
    manifest_output = Path(manifest_out)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    table_output.write_text(table_path.read_text(encoding="utf-8"), encoding="utf-8")

    stage8_summary = _read_stage8_summary(stage8_report_path)
    manifest = {
        "source_metrics_path": str(metrics_path),
        "source_table_path": str(table_path),
        "embeddings_path": str(Path(embeddings_path)),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "target_label": metrics.get("target_label", ""),
        "stage8_success_count": stage8_summary.get("success_count"),
        "stage8_failure_count": stage8_summary.get("failure_count"),
        "baseline_overfit_passed": bool(metrics.get("overfit_check", {}).get("passed")),
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def run_upgrade_ablation(
    *,
    embeddings_path: Path | str,
    baseline_metrics_path: Path | str,
    upgrade: str,
    target_label: str | None,
    out_table: Path | str,
    failures_out: Path | str,
    metrics_out: Path | str = "outputs/reports/modality_token_fusion_metrics.json",
    model_out: Path | str = "outputs/models/modality_token_fusion.pt",
    epochs: int = 200,
    overfit_limit: int = 128,
    hidden_dim: int = 32,
    learning_rate: float = 0.05,
    seed: int = 29,
) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_metrics_path).read_text(encoding="utf-8"))
    label = target_label or baseline.get("target_label")
    if not label:
        raise ValueError("target_label is required when baseline metrics do not declare one.")
    baseline_metric = _baseline_full_rmse(baseline)
    failures = _validate_embedding_split(embeddings_path, target_label=label)

    if upgrade == "modality_token_attention":
        if failures:
            upgrade_metric = None
            decision = "rollback"
            reason = "subject split validation failed"
        else:
            attention = _run_modality_token_attention(
                embeddings_path=embeddings_path,
                target_label=label,
                metrics_out=metrics_out,
                model_out=model_out,
                epochs=epochs,
                overfit_limit=overfit_limit,
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                seed=seed,
            )
            upgrade_metric = attention["test"]["rmse"]
            decision = (
                "accepted"
                if attention["overfit_check"]["passed"] and upgrade_metric is not None and upgrade_metric < baseline_metric
                else "rollback"
            )
            reason = "test_rmse improved" if decision == "accepted" else "test_rmse did not improve baseline"
        upgrade_type = "fusion"
    elif upgrade != "registry_smoke":
        failures.append(
            {
                "experiment": upgrade,
                "error_type": "unsupported_upgrade",
                "error": f"Upgrade '{upgrade}' is not registered.",
            }
        )
        upgrade_metric = None
        decision = "rollback"
        reason = "unsupported upgrade"
        upgrade_type = "unknown"
    else:
        upgrade_metric = baseline_metric
        decision = "rollback"
        reason = "registry_smoke does not improve baseline"
        upgrade_type = "framework"
        if failures:
            reason = "subject split validation failed"

    result = {
        "experiment": upgrade,
        "upgrade_type": upgrade_type,
        "baseline_metric": baseline_metric,
        "upgrade_metric": upgrade_metric,
        "delta": None if upgrade_metric is None else upgrade_metric - baseline_metric,
        "decision": decision,
        "reason": reason,
        "failures": failures,
    }
    _write_ablation_table([result], out_table)
    _write_failures(failures, failures_out)
    return result


def _validate_baseline_metrics(metrics: dict[str, Any]) -> None:
    if not metrics.get("overfit_check", {}).get("passed"):
        raise ValueError("Baseline overfit check did not pass; stop before stage 10.2.")
    full = metrics.get("runs", {}).get("full", {})
    test = full.get("test", {})
    if "rmse" not in test:
        raise ValueError("Baseline metrics must include runs.full.test.rmse.")


def _read_stage8_summary(stage8_report_path: Path | str | None) -> dict[str, Any]:
    if not stage8_report_path:
        return {}
    path = Path(stage8_report_path)
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    return report.get("summary", {})


def _baseline_full_rmse(metrics: dict[str, Any]) -> float:
    try:
        return float(metrics["runs"]["full"]["test"]["rmse"])
    except KeyError as exc:
        raise ValueError("Baseline metrics must include runs.full.test.rmse.") from exc


def _validate_embedding_split(embeddings_path: Path | str, *, target_label: str) -> list[dict[str, Any]]:
    data = _load_embedding_dataset(embeddings_path, target_label=target_label)
    split = _subject_split(data["subject_id"])
    missing = [
        name
        for name, indices in split["indices"].items()
        if len(indices) == 0
    ]
    if not missing:
        return []
    return [
        {
            "experiment": "registry_smoke",
            "error_type": "subject_split_incomplete",
            "error": f"Missing samples for split(s): {', '.join(missing)}",
            "missing_splits": missing,
        }
    ]


def _run_modality_token_attention(
    *,
    embeddings_path: Path | str,
    target_label: str,
    metrics_out: Path | str,
    model_out: Path | str,
    epochs: int,
    overfit_limit: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    data = _load_embedding_dataset(embeddings_path, target_label=target_label)
    split = _subject_split(data["subject_id"])
    x = _modality_attention_features(data)
    overfit = _run_overfit_check(
        x,
        data["target"],
        limit=overfit_limit,
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    model = _fit_mlp(
        x[split["indices"]["train"]],
        data["target"][split["indices"]["train"]],
        epochs=epochs,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        seed=seed,
    )
    metrics = {
        "stage": 10,
        "upgrade": "modality_token_attention",
        "target_label": data["target_label"],
        "split": {
            "train_subjects": split["train_subjects"],
            "val_subjects": split["val_subjects"],
            "test_subjects": split["test_subjects"],
        },
        "overfit_check": overfit,
        "train": _evaluate_split(model, x, data["target"], split["indices"]["train"]),
        "val": _evaluate_split(model, x, data["target"], split["indices"]["val"]),
        "test": _evaluate_split(model, x, data["target"], split["indices"]["test"]),
    }
    Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_model(model, model_out, metadata={"target_label": data["target_label"], "upgrade": "modality_token_attention"})
    return metrics


def _modality_attention_features(data: dict[str, Any]) -> Any:
    import numpy as np

    tokens = np.stack(
        [data["eeg_emb"], data["wear_emb"], data["audio_emb"], data["face_emb"]],
        axis=1,
    ).astype(np.float32)
    mask = data["modality_mask"][:, [0, 1, 3, 2]].astype(bool)
    raw_scores = np.linalg.norm(tokens, axis=2)
    raw_scores = np.where(mask, raw_scores, -np.inf)
    all_missing = ~mask.any(axis=1)
    raw_scores[all_missing] = 0.0
    shifted = raw_scores - np.max(raw_scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights = np.where(mask, weights, 0.0)
    weights[all_missing] = 0.25
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_sum[weights_sum == 0.0] = 1.0
    weights = weights / weights_sum
    return np.sum(tokens * weights[:, :, None], axis=1).astype(np.float32)


def _write_ablation_table(results: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "| experiment | upgrade_type | baseline_metric | upgrade_metric | delta | decision | reason |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for result in results:
        rows.append(
            "| {experiment} | {upgrade_type} | {baseline} | {upgrade} | {delta} | {decision} | {reason} |".format(
                experiment=result["experiment"],
                upgrade_type=result["upgrade_type"],
                baseline=_format_metric(result["baseline_metric"]),
                upgrade=_format_metric(result["upgrade_metric"]),
                delta=_format_metric(result["delta"]),
                decision=result["decision"],
                reason=result["reason"],
            )
        )
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def _write_failures(failures: list[dict[str, Any]], output: Path | str) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"
