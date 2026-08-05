from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from daily_multimodal.training.fusion_matrix import (  # noqa: E402
    branches_for_experiment,
    load_fusion_matrix_config,
    matrix_experiment_specs,
)
from daily_multimodal.training.cross_attention_fusion import (  # noqa: E402
    LearnableAttentionConfig,
    build_fusion_dataset,
    fit_learnable_cross_attention,
    predict_with_learnable_cross_attention,
    save_learnable_cross_attention_model,
)
from daily_multimodal.training.subject_cv import build_subject_folds  # noqa: E402

import numpy as np  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or inspect the multimodal fusion experiment matrix.")
    parser.add_argument("--config", default="configs/fusion_matrix.yaml")
    parser.add_argument("--target-label")
    parser.add_argument("--model", choices=["concat_mlp", "learnable_cross_attention"], default="learnable_cross_attention")
    parser.add_argument("--out-dir", default="outputs/reports/fusion_matrix")
    parser.add_argument("--model-dir", default="outputs/models/fusion_matrix")
    parser.add_argument("--strategy", choices=["leave_one_subject_out", "grouped_k_fold"], default="leave_one_subject_out")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device")
    parser.add_argument("--max-experiments", type=int, help="Run only the first N expanded experiments for smoke validation.")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and write the expanded experiment manifest only.")
    args = parser.parse_args()

    config = load_fusion_matrix_config(args.config)
    specs = matrix_experiment_specs(config)
    if args.max_experiments is not None:
        specs = specs[: max(0, int(args.max_experiments))]
    target_label = args.target_label or config.target_label
    manifest = {
        "config": str(args.config),
        "target_label": target_label,
        "model": args.model,
        "out_dir": args.out_dir,
        "model_dir": args.model_dir,
        "strategy": args.strategy,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "decision_rule": "promote only when a full run beats its matched no_audio and available no_video controls on RMSE, without lower Pearson r",
        "experiment_count": len(specs),
        "experiments": [
            {
                "name": spec.name,
                "comparison_family": _comparison_family(spec.name),
                "enabled_modalities": list(spec.enabled_modalities),
                "min_available_modalities": spec.min_available_modalities,
                "branches": {
                    modality: {
                        "path": str(branch.path),
                        "profile": branch.profile,
                        "modality": branch.modality,
                    }
                    for modality, branch in branches_for_experiment(config, spec.name).items()
                },
            }
            for spec in specs
        ],
        "metadata_source": None
        if config.metadata_source is None
        else {
            "path": str(config.metadata_source.path),
            "profile": config.metadata_source.profile,
            "modality": config.metadata_source.modality,
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "fusion_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"experiment_count={len(specs)}")
    print(f"manifest_path={manifest_path}")
    if args.dry_run:
        return 0
    if args.model != "learnable_cross_attention":
        raise ValueError("fusion matrix runner currently trains only learnable_cross_attention")
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    paired_base_sample_ids: dict[str, np.ndarray] = {}
    for exp_index, spec in enumerate(specs):
        branches = branches_for_experiment(config, spec.name)
        dataset = build_fusion_dataset(
            branches=branches,
            experiment=spec,
            base_sample_ids=paired_base_sample_ids.get(spec.name),
            metadata_source=config.metadata_source,
        )
        _record_paired_sample_ids(spec.name, dataset.sample_id, paired_base_sample_ids)
        result = _run_experiment_subject_cv(
            dataset=dataset,
            strategy=args.strategy,
            n_splits=args.n_splits,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            seed=args.seed + exp_index,
            device=args.device,
            model_dir=model_dir / spec.name,
        )
        result.update(
            {
                "experiment": spec.name,
                "comparison_family": _comparison_family(spec.name),
                "modalities": list(dataset.modalities),
                "branch_profiles": dataset.branch_profiles,
                "row_count": int(len(dataset.sample_id)),
                "target_label": target_label,
            }
        )
        exp_json = out_dir / f"{spec.name}_metrics.json"
        exp_table = out_dir / f"{spec.name}_table.md"
        exp_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        exp_table.write_text(_result_table(result), encoding="utf-8")
        results.append(result)
        print(f"completed={spec.name} folds={result['fold_count']} rmse_mean={_fmt(result['rmse_mean'])} pearson_r_mean={_fmt(result['pearson_r_mean'])}")
    _add_matrix_decisions(results)
    summary = {
        "config": str(args.config),
        "target_label": target_label,
        "model": args.model,
        "strategy": args.strategy,
        "experiment_count": len(results),
        "experiments": [
            {
                "experiment": row["experiment"],
                "comparison_family": row["comparison_family"],
                "modalities": row["modalities"],
                "branch_profiles": row["branch_profiles"],
                "row_count": row["row_count"],
                "fold_count": row["fold_count"],
                "rmse_mean": row["rmse_mean"],
                "rmse_std": row["rmse_std"],
                "pearson_r_mean": row["pearson_r_mean"],
                "pearson_r_std": row["pearson_r_std"],
                "decision": row["decision"],
                "reason": row["reason"],
            }
            for row in results
        ],
    }
    summary_path = out_dir / "fusion_matrix_summary.json"
    table_path = out_dir / "fusion_matrix_summary.md"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    table_path.write_text(_summary_table(summary), encoding="utf-8")
    print(f"summary_path={summary_path}")
    return 0


def _comparison_family(experiment_name: str) -> str:
    parts = experiment_name.split("_")
    if len(parts) >= 4 and parts[-1] in {"full", "audio"}:
        return f"{parts[1]}_{parts[2]}"
    if experiment_name.endswith("_no_audio") and len(parts) >= 5:
        return f"{parts[1]}_{parts[2]}"
    if experiment_name.endswith("_no_video"):
        return f"{parts[1]}_no_video"
    if experiment_name.endswith("_bio_only"):
        return f"{parts[1]}_bio_only"
    return experiment_name.removeprefix("fusion_")


def _record_paired_sample_ids(experiment_name: str, sample_id: np.ndarray, store: dict[str, np.ndarray]) -> None:
    if experiment_name.endswith("_full"):
        store[experiment_name.replace("_full", "_no_audio")] = sample_id


def _run_experiment_subject_cv(
    *,
    dataset,
    strategy: str,
    n_splits: int,
    epochs: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    device: str | None,
    model_dir: Path,
) -> dict:
    folds = build_subject_folds(dataset.subject_id, strategy=strategy, n_splits=n_splits, seed=seed)
    fold_results = []
    for offset, fold in enumerate(folds):
        if len(fold.train) == 0 or len(fold.val) == 0 or len(fold.test) == 0:
            raise ValueError(f"fold {fold.name} has an empty train/val/test split")
        model = fit_learnable_cross_attention(
            dataset,
            train_indices=fold.train,
            val_indices=fold.val,
            config=LearnableAttentionConfig(
                token_dim=int(hidden_dim),
                epochs=int(epochs),
                learning_rate=float(learning_rate),
                seed=int(seed) + offset,
                device=device,
            ),
        )
        checkpoint_path = model_dir / f"{fold.name}.pt"
        save_learnable_cross_attention_model(model, checkpoint_path)
        train_pred, _ = predict_with_learnable_cross_attention(model, dataset, indices=fold.train)
        val_pred, _ = predict_with_learnable_cross_attention(model, dataset, indices=fold.val)
        test_pred, attention = predict_with_learnable_cross_attention(model, dataset, indices=fold.test)
        fold_results.append(
            {
                "fold": fold.name,
                "train_subjects": fold.train_subjects,
                "val_subjects": fold.val_subjects,
                "test_subjects": fold.test_subjects,
                "train": _metrics(train_pred, dataset.target[fold.train]),
                "val": _metrics(val_pred, dataset.target[fold.val]),
                "test": _metrics(test_pred, dataset.target[fold.test]),
                "sample_counts": {
                    "train": int(len(fold.train)),
                    "val": int(len(fold.val)),
                    "test": int(len(fold.test)),
                },
                "attention_summary": _attention_summary(attention, dataset.modalities),
                "checkpoint": str(checkpoint_path),
            }
        )
    rmses = np.asarray([row["test"]["rmse"] for row in fold_results], dtype=np.float32)
    pearsons = np.asarray([row["test"]["pearson"] for row in fold_results if row["test"]["pearson"] is not None], dtype=np.float32)
    return {
        "fold_count": len(fold_results),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "pearson_r_mean": None if pearsons.size == 0 else float(np.mean(pearsons)),
        "pearson_r_std": None if pearsons.size == 0 else float(np.std(pearsons)),
        "folds": fold_results,
    }


def _metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    err = pred - truth
    return {
        "count": int(len(truth)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": _pearson(pred, truth),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 2 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _attention_summary(attention: np.ndarray, modalities: tuple[str, ...]) -> dict[str, float]:
    means = np.mean(np.asarray(attention, dtype=np.float32), axis=0)
    return {modality: float(means[idx]) for idx, modality in enumerate(modalities)}


def _result_table(result: dict) -> str:
    rows = [
        "| fold | test_subjects | train_count | val_count | test_count | test_rmse | test_mae | test_r |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in result["folds"]:
        rows.append(
            "| {fold} | {subjects} | {train_count} | {val_count} | {test_count} | {rmse} | {mae} | {r} |".format(
                fold=fold["fold"],
                subjects=",".join(fold["test_subjects"]),
                train_count=fold["sample_counts"]["train"],
                val_count=fold["sample_counts"]["val"],
                test_count=fold["sample_counts"]["test"],
                rmse=_fmt(fold["test"]["rmse"]),
                mae=_fmt(fold["test"]["mae"]),
                r=_fmt(fold["test"]["pearson"]),
            )
        )
    return "\n".join(rows) + "\n"


def _summary_table(summary: dict) -> str:
    rows = [
        "| experiment | row_count | modalities | rmse_mean | rmse_std | pearson_r_mean | pearson_r_std | decision |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["experiments"]:
        rows.append(
            "| {experiment} | {row_count} | {modalities} | {rmse_mean} | {rmse_std} | {r_mean} | {r_std} | {decision} |".format(
                experiment=row["experiment"],
                row_count=row["row_count"],
                modalities=",".join(row["modalities"]),
                rmse_mean=_fmt(row["rmse_mean"]),
                rmse_std=_fmt(row["rmse_std"]),
                r_mean=_fmt(row["pearson_r_mean"]),
                r_std=_fmt(row["pearson_r_std"]),
                decision=row["decision"],
            )
        )
    return "\n".join(rows) + "\n"


def _add_matrix_decisions(results: list[dict]) -> None:
    by_name = {row["experiment"]: row for row in results}
    no_video_by_wear = {
        row["experiment"].replace("fusion_", "").replace("_no_video", ""): row
        for row in results
        if row["experiment"].endswith("_no_video")
    }
    for row in results:
        name = row["experiment"]
        if name.endswith("_full"):
            no_audio = by_name.get(name.replace("_full", "_no_audio"))
            wear_name = name.split("_")[1]
            no_video = no_video_by_wear.get(wear_name)
            controls = [control for control in (no_audio, no_video) if control is not None]
            row["decision"], row["reason"] = _full_run_decision(row, controls)
        elif name.endswith("_no_audio"):
            row["decision"] = "audio_control"
            row["reason"] = "matched run without audio"
        elif name.endswith("_no_video"):
            row["decision"] = "video_control"
            row["reason"] = "matched wear run without video"
        elif name.endswith("_bio_only"):
            row["decision"] = "bio_only_control"
            row["reason"] = "EEG and wear only sanity baseline"
        else:
            row["decision"] = "needs_review"
            row["reason"] = "unrecognized experiment family"


def _full_run_decision(row: dict, controls: list[dict]) -> tuple[str, str]:
    if not controls:
        return "needs_review", "missing matched controls"
    rmse = row.get("rmse_mean")
    r_value = row.get("pearson_r_mean")
    if rmse is None:
        return "needs_review", "missing full-run RMSE"
    for control in controls:
        control_rmse = control.get("rmse_mean")
        control_r = control.get("pearson_r_mean")
        if control_rmse is None:
            return "needs_review", f"missing control RMSE for {control['experiment']}"
        if float(rmse) >= float(control_rmse):
            return "rollback", f"does not beat {control['experiment']} RMSE"
        if r_value is not None and control_r is not None and float(r_value) < float(control_r):
            return "rollback", f"Pearson r below {control['experiment']}"
    return "accepted_candidate", "beats matched no_audio/no_video controls"


def _fmt(value: float | None) -> str:
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
