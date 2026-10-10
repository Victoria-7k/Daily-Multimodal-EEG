#!/usr/bin/env python3
"""Matched MAE E1: uniform representation pooling followed by the existing 11-label head."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_jsonl
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.training import _seed_everything, load_bag_dataset
from daily_multimodal.training.mae_emotion_adaptation import state_hash
from daily_multimodal.training.mae_eeg_partial_ft import array_hash
from daily_multimodal.training.multihead_regression import evaluate_event_level
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, _predict, event_targets, run_condition

install_numpy_core_pickle_aliases()

def load_helper(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

h = load_helper("basic_pool_paired_metrics", "140_report_round2_experiment_a.py")
a = load_helper("basic_pool_reference_paths", "139_run_round2_experiment_a.py")
SEEDS = a.SEEDS
ROUTES = ("F_C", "C_EVENT", "C_BAG_UNIFORM")
COEFFICIENTS = {"C_BAG_UNIFORM-C_EVENT": (0, -1, 1), "C_BAG_UNIFORM-F_C": (-1, 0, 1)}
PARAMETERS = dict(epochs=80, batch_size=64, hidden_dim=128, learning_rate=1e-3,
                  weight_decay=1e-4, dropout=.1, patience=15, modality_dropout_prob=.1,
                  lambda_d=.25, probe_loss_weight=.1)

def write(path, value):
    a.write(path, value)

def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("immutable contract differs: " + str(path))
    if not path.exists():
        write(path, value)

def progress(root, stage, **details):
    row = dict(stage=stage, updated_utc=datetime.now(timezone.utc).isoformat(), **details)
    write(root / "PROGRESS.json", row)
    print(json.dumps(row), flush=True)

def cell(root, phase, seed):
    return root / "runs" / phase / f"seed_{seed}"

def load_inputs(root, prior):
    prior_contract = json.loads((prior / "inputs/c_fixed_config.json").read_text())
    ds = load_bag_dataset(Path(prior_contract["bag_paths"]["C"]))
    targets = event_targets(ds, load_jsonl(a.ALIGNED / "index/eeg_aligned_window_index.jsonl"))
    contract = json.loads((root / "inputs/fixed_config.json").read_text())
    if a.sha(ds.bag_path) != contract["bag_sha256"]:
        raise ValueError("frozen MAE C bag changed")
    if list(SEEDS) != contract["seeds"] or tuple(contract["label_names"]) != LABEL_NAMES:
        raise ValueError("seed or label contract changed")
    return ds, targets, contract

def initial_audit(seed, device):
    _seed_everything(seed)
    cfg = DailyAffectRegressionConfig(model_id="bag_static", hidden_dim=128, dropout=.1,
                                     adapter_mode="per_modality", temporal_policy="uniform", lambda_d=.25)
    model = MultiEmotionStructureModel(cfg).to(device)
    return {"state_sha256": state_hash(model.state_dict()),
            "cpu_rng_sha256": array_hash(torch.get_rng_state().numpy()),
            "cuda_rng_sha256": [array_hash(value.cpu().numpy()) for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}

def preflight(root, prior, device):
    for name in ("EXPERIMENT_C_REPLAY_VERIFIED", "ROUND2_STAGE_D_TEST_REPORT_COMPLETE"):
        if not (prior / name).exists():
            raise ValueError("prior experiment incomplete: " + name)
    for filename in ("c_execution_source_manifest.json", "d_execution_source_manifest.json"):
        old = json.loads((prior / "inputs" / filename).read_text())
        for name, digest in old["files"].items():
            if a.sha(prior / name) != digest:
                raise ValueError("prior immutable source changed: " + name)
    cc = json.loads((prior / "inputs/c_fixed_config.json").read_text())
    ds = load_bag_dataset(Path(cc["bag_paths"]["C"]))
    if a.sha(ds.bag_path) != cc["bag_sha256"]["C"]:
        raise ValueError("canonical C bag changed")
    split = ds.split_indices()
    if {k: len(split[k]) for k in ("train", "val", "test")} != {"train": 731, "val": 269, "test": 253}:
        raise ValueError("canonical event split changed")
    if ds.tokens.shape != (1253, 23, 4, 256) or not ds.modality_mask.any(axis=2).any(axis=1).all():
        raise ValueError("invalid bag shape or empty event")
    if ds.modality_mask[:, :, 3].any():
        raise ValueError("Audio must remain disabled")
    for x, y in (("train", "val"), ("train", "test"), ("val", "test")):
        if set(ds.event_id[split[x]]) & set(ds.event_id[split[y]]):
            raise ValueError("event split overlap")
    support = json.loads((a.R1 / "inputs/eeg/cross_day/filter_report.json").read_text())
    if support["status"] != "pass" or support["canonical_train_holdout_overlap_count"] or support["canonical_train_holdout_preprocessing_context_overlap_count"]:
        raise ValueError("cross_day support isolation failed")
    for name, digest in support["provenance"].items():
        if a.sha(name) != digest:
            raise ValueError("cross_day provenance changed")
    reused, initialization = [], []
    for seed in SEEDS:
        audit = initial_audit(seed, device)
        for route in ("F_C", "C_EVENT"):
            directory = a.controls("C", seed) if route == "F_C" else prior / f"fusion/C_EVENT/formal/cross_day/seed_{seed}"
            metric, arrays, _ = h.read_val(directory)
            for key in ("event_id", "subject_id", "day_id"):
                if not np.array_equal(arrays[key], getattr(ds, key)[split["val"]]):
                    raise ValueError("control event identity mismatch")
            if metric["model_id"] != "window_replicated" or metric["head_variant"] != "H1_shared2_11xhead2" or metric["temporal_policy"] != "uniform":
                raise ValueError("control structure mismatch")
            if route == "C_EVENT" and audit["state_sha256"] != metric["head_initial_sha256"]:
                raise ValueError("bag/window initial parameters differ")
            if route == "C_EVENT" and (audit["cpu_rng_sha256"] != metric["head_cpu_rng_sha256"] or audit["cuda_rng_sha256"] != metric["head_cuda_rng_sha256"]):
                raise ValueError("bag/window initial RNG differs")
            files = {n: a.sha(directory / n) for n in ("best_checkpoint.pt", "metrics.json", "event_predictions.npz")}
            reused.append(dict(route=route, seed=seed, directory=str(directory), sha256=files))
            dest = root / "inputs/reused" / route / f"seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("metrics.json", "event_predictions.npz"):
                if not (dest / name).exists():
                    shutil.copyfile(directory / name, dest / name)
                if a.sha(dest / name) != files[name]:
                    raise ValueError("reused artifact copy differs")
        initialization.append(dict(seed=seed, **audit))
    contract = dict(scope="MAE_C_basic_representation_pool_only", protocol="cross_day", seeds=list(SEEDS),
        embedding_seed=240800, label_names=list(LABEL_NAMES), bag_path=str(ds.bag_path), bag_sha256=a.sha(ds.bag_path),
        model_id="bag_static", temporal_policy="uniform", head_variant="H1_shared2_11xhead2",
        parameters=PARAMETERS, normalization="per_modality", adapter_mode="per_modality", loss="standardized_event_MSE",
        selection_metric="val_macro_standardized_rmse_min", primary_report_metric="within_subject_centered_r",
        primary_comparison="C_BAG_UNIFORM-C_EVENT", secondary_comparison="C_BAG_UNIFORM-F_C",
        encoder_training_scope="all_frozen", representation_pool="masked_uniform_then_head",
        no_variance_penalty=True, reused=reused, initialization=initialization,
        within_subject_day="inherited_protocol_overlap_stop", evidence_status="exploratory_previously_seen_test",
        next_stage="none_automatic", evaluation_scope="three_downstream_seeds_one_embedding_seed")
    immutable_json(root / "inputs/fixed_config.json", contract)
    (root / "PREFLIGHT_COMPLETE").write_text("pass\n")
    progress(root, "preflight", status="pass", train=731, val=269, test=253, next="smoke")

def train(root, prior, phase, device):
    ds, targets, contract = load_inputs(root, prior)
    if phase == "formal" and not (root / "SMOKE_COMPLETE").exists():
        raise ValueError("smoke prerequisite missing")
    for seed in ((SEEDS[0],) if phase == "smoke" else SEEDS):
        directory = cell(root, phase, seed)
        if (directory / "VAL_TRAINING_COMPLETE").exists():
            h.read_val(directory)
            continue
        directory.mkdir(parents=True, exist_ok=True)
        progress(root, phase, status="running", seed=seed)
        args = dict(PARAMETERS)
        if phase == "smoke":
            args.update(epochs=3, patience=3)
        # Original trainer sees validation rows in both evaluation leaves. No real test inference occurs here.
        result = run_condition(dataset=replace(ds, test_index=ds.val_index.copy()), targets=targets,
            protocol="cross_day", condition_id="C_BAG_UNIFORM", model_id="bag_static", temporal_policy="uniform",
            seed=seed, out_dir=directory, device=device, **args)
        result.pop("test")
        result.update(test_inference_performed=False, training_parameters=args, source="original_frozen_MAE_C")
        write(directory / "metrics.json", result)
        with np.load(directory / "event_predictions.npz", allow_pickle=False) as z:
            val_arrays = {k: z[k] for k in z.files if not k.startswith("test_")}
        np.savez_compressed(directory / "event_predictions.npz", **val_arrays)
        cp = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        control = next(row for row in contract["reused"] if row["route"] == "C_EVENT" and row["seed"] == seed)
        ref = torch.load(Path(control["directory"]) / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        for key in ("x_mean", "x_std", "y_mean", "y_std"):
            if not np.array_equal(cp[key], ref[key]):
                raise ValueError("matched normalization changed: " + key)
        if not all(torch.isfinite(value).all() for value in cp["state_dict"].values()):
            raise ValueError("nonfinite checkpoint")
        h.read_val(directory)
        (directory / "VAL_TRAINING_COMPLETE").write_text("pass\n")
        progress(root, phase, status="cell_complete", seed=seed, best_epoch=result["best_epoch"],
                 epochs_ran=result["epochs_ran"], val_centered=result["val"]["summary"]["within_subject_centered_r"])
    (root / ("SMOKE_COMPLETE" if phase == "smoke" else "FORMAL_COMPLETE")).write_text("pass\n")

def freeze_validation(root):
    if not (root / "FORMAL_COMPLETE").exists():
        raise ValueError("formal phase required")
    contract = json.loads((root / "inputs/fixed_config.json").read_text())
    values, provenance = {}, {}
    for seed in SEEDS:
        for route in ROUTES:
            directory = cell(root, "formal", seed) if route == "C_BAG_UNIFORM" else Path(next(row["directory"] for row in contract["reused"] if row["route"] == route and row["seed"] == seed))
            _, _, values[route, seed] = h.read_val(directory)
            provenance[str(directory)] = {n: a.sha(directory / n) for n in ("metrics.json", "event_predictions.npz", "best_checkpoint.pt")}
    tensor = np.stack([np.stack([values[route, seed] for seed in SEEDS]) for route in ROUTES])
    val = dict(leaf="val_only", test_used_for_selection=False, checkpoint_selection=contract["selection_metric"],
               configuration="one_fixed_bag_static_uniform", primary_metric=contract["primary_report_metric"],
               fixed_config_sha256=a.sha(root / "inputs/fixed_config.json"), all_9_cells_provenance=provenance,
               val_means={route: dict(zip(h.METRICS, tensor[i].mean(axis=(0, 1)).tolist())) for i, route in enumerate(ROUTES)},
               val_contrasts={name: dict(zip(h.METRICS, np.tensordot(co, tensor, axes=(0, 0)).mean(axis=(0, 1)).tolist())) for name, co in COEFFICIENTS.items()})
    path = root / "reports/selection_val_only.json"
    immutable_json(path, val)
    (root / "VAL_SELECTION_FROZEN").write_text(a.sha(path) + "\n")
    progress(root, "selection", status="frozen", val_contrasts=val["val_contrasts"], next="independent_replay_and_test_report")

def report(root, prior, device):
    ds, targets, contract = load_inputs(root, prior)
    frozen_path = root / "reports/selection_val_only.json"
    if not (root / "VAL_SELECTION_FROZEN").exists() or (root / "VAL_SELECTION_FROZEN").read_text().strip() != a.sha(frozen_path):
        raise ValueError("validation freeze required before test")
    frozen = json.loads(frozen_path.read_text())
    for directory, files in frozen["all_9_cells_provenance"].items():
        for name, digest in files.items():
            if a.sha(Path(directory) / name) != digest:
                raise ValueError("frozen checkpoint or artifact changed")
    out = root / "reports"
    evaluations, audits = {}, []
    for seed in SEEDS:
        directory = cell(root, "formal", seed)
        cp = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = DailyAffectRegressionConfig(model_id="bag_static", hidden_dim=128, dropout=.1,
                                         adapter_mode="per_modality", temporal_policy="uniform", lambda_d=.25)
        model = MultiEmotionStructureModel(cfg).to(device)
        model.load_state_dict(cp["state_dict"])
        m = json.loads((directory / "metrics.json").read_text())
        result, arrays = dict(m), {"label_names": np.asarray(LABEL_NAMES)}
        replay_error = None
        for leaf in ("val", "test"):
            idx = ds.split_indices()[leaf]
            pred = _predict(model, ds, idx, cp["x_mean"], cp["x_std"], cp["y_mean"], cp["y_std"], torch.device(device))
            if not np.isfinite(pred).all():
                raise ValueError("nonfinite independent prediction")
            if leaf == "val":
                with np.load(directory / "event_predictions.npz", allow_pickle=False) as z:
                    replay_error = float(np.max(np.abs(pred - z["val_prediction"])))
                if replay_error != 0:
                    raise ValueError("independent val checkpoint replay differs")
            result[leaf] = evaluate_event_level(dict(target=targets[idx], prediction=pred, subject_id=ds.subject_id[idx]), cp["y_std"])
            for key, value in dict(target=targets[idx], prediction=pred, event_id=ds.event_id[idx], subject_id=ds.subject_id[idx], day_id=ds.day_id[idx]).items():
                arrays[f"{leaf}_{key}"] = value
        result.update(test_inference_performed=True, validation_selection_sha256=a.sha(frozen_path))
        dest = out / "evaluated/C_BAG_UNIFORM" / f"seed_{seed}"
        dest.mkdir(parents=True, exist_ok=True)
        write(dest / "metrics.json", result)
        np.savez_compressed(dest / "event_predictions.npz", **arrays)
        evaluations["C_BAG_UNIFORM", seed] = dest
        audits.append(dict(seed=seed, checkpoint_sha256=a.sha(directory / "best_checkpoint.pt"), val_max_abs_error=replay_error))
        for route in ("F_C", "C_EVENT"):
            evaluations[route, seed] = root / "inputs/reused" / route / f"seed_{seed}"
    summaries, contrasts, metric_rows, delta_rows, bootstrap_rows = {}, {}, [], [], []
    reader = load_helper("basic_pool_frozen_test_reader", "147_report_round2_stage_d.py")
    for leaf in ("val", "test"):
        arrays, metrics, values = {}, {}, {}
        for route in ROUTES:
            for seed in SEEDS:
                metrics[route, seed], arrays[route, seed], values[route, seed] = reader.read_leaf(evaluations[route, seed], leaf, selection_frozen=True)
                for li, label in enumerate((*LABEL_NAMES, "macro")):
                    value = values[route, seed][li] if li < 11 else values[route, seed].mean(axis=0)
                    arr = arrays[route, seed]
                    sy, sp = arr["target"].std(axis=0), arr["prediction"].std(axis=0)
                    metric_rows.append(dict(leaf=leaf, route=route, seed=seed, label=label, **dict(zip(h.METRICS, value)),
                        target_std=float(sy[li] if li < 11 else sy.mean()), prediction_std=float(sp[li] if li < 11 else sp.mean())))
        tensor = np.stack([np.stack([values[route, seed] for seed in SEEDS]) for route in ROUTES])
        summaries[leaf] = {route: {metric: dict(mean=float(tensor[i, :, :, j].mean()), sd=float(tensor[i, :, :, j].mean(axis=1).std(ddof=1))) for j, metric in enumerate(h.METRICS)} for i, route in enumerate(ROUTES)}
        deltas = {name: np.tensordot(co, tensor, axes=(0, 0)) for name, co in COEFFICIENTS.items()}
        contrasts[leaf] = {name: dict(means=dict(zip(h.METRICS, delta.mean(axis=(0, 1)).tolist())),
            seed_macro_deltas=delta.mean(axis=1).tolist(), centered_positive_seeds=int((delta.mean(axis=1)[:, 1] > 0).sum()),
            centered_delta_sd=float(delta.mean(axis=1)[:, 1].std(ddof=1))) for name, delta in deltas.items()}
        for si, seed in enumerate(SEEDS):
            scale = np.asarray(metrics[ROUTES[0], seed]["train_target_std"])
            for route in ROUTES[1:]:
                if not np.array_equal(scale, np.asarray(metrics[route, seed]["train_target_std"])):
                    raise ValueError("label normalization differs across comparison")
            intervals, blocks = h.bootstraps([arrays[route, seed] for route in ROUTES], scale, COEFFICIENTS, 2000, seed + (181000 if leaf == "val" else 191000))
            for name, delta in deltas.items():
                for li, label in enumerate((*LABEL_NAMES, "macro")):
                    value = delta[si, li] if li < 11 else delta[si].mean(axis=0)
                    delta_rows.append(dict(leaf=leaf, contrast=name, seed=seed, label=label, **dict(zip(h.METRICS, value))))
                    ci = intervals[name]["per_label"][:, li] if li < 11 else intervals[name]["macro"]
                    for j, metric in enumerate(h.METRICS):
                        bootstrap_rows.append(dict(leaf=leaf, contrast=name, seed=seed, label=label, metric=metric,
                            delta=float(value[j]), ci_low=float(ci[0, j]), ci_high=float(ci[1, j]), subject_day_blocks=blocks, iterations=2000))
    h.csv_write(out / "per_label_metrics.csv", metric_rows)
    h.csv_write(out / "paired_deltas.csv", delta_rows)
    h.csv_write(out / "paired_bootstrap.csv", bootstrap_rows)
    result = dict(status="complete", contract=contract, summaries=summaries, contrasts=contrasts,
                  checkpoint_replays=audits, validation_selection_sha256=a.sha(frozen_path), test_reselected=False)
    write(out / "results.json", result)
    lines = ["# MAE E1：基本0906式event预测配对实验", "", "固定全部有效窗口等权；先平均窗口表示，再经原H1共享11情绪head预测。", "",
             "原C冻结token、split、归一化、seed与预算保留。checkpoint按val平均sRMSE选择；重点读数为centered r。", "",
             "| leaf | route | centered r mean±SD | raw r mean±SD | sRMSE mean±SD |", "|---|---|---:|---:|---:|"]
    for leaf in ("val", "test"):
        for route in ROUTES:
            row = summaries[leaf][route]
            cells = [f"{row[key]['mean']:.6f} ± {row[key]['sd']:.6f}" for key in ("within_subject_centered_r", "raw_r", "standardized_rmse")]
            lines.append("| " + " | ".join((leaf, route, *cells)) + " |")
    lines += ["", "| leaf | comparison | Δcentered r | centered正向seed | Δraw r | ΔsRMSE |", "|---|---|---:|---:|---:|---:|"]
    for leaf in ("val", "test"):
        for name, row in contrasts[leaf].items():
            mean = row["means"]
            lines.append(f"| {leaf} | {name} | {mean['within_subject_centered_r']:+.6f} | {row['centered_positive_seeds']}/3 | {mean['raw_r']:+.6f} | {mean['standardized_rmse']:+.6f} |")
    lines += ["", "## 配对centered r区间", "", "| leaf | comparison | seed | Δcentered r | 95% subject-day配对区间 |", "|---|---|---:|---:|---|"]
    for row in bootstrap_rows:
        if row["label"] == "macro" and row["metric"] == "within_subject_centered_r":
            lines.append(f"| {row['leaf']} | {row['contrast']} | {row['seed']} | {row['delta']:+.6f} | [{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] |")
    lines += ["", "## 验收与解释边界", "", "- 单一bag_static/uniform配置，三下游seed；冻结EEG-MAE上游仅一个seed。", "- formal训练阶段的test索引映射到val；原test首次推理发生在完整val选择hash冻结之后。", "- 三个新checkpoint独立val回放误差均0；18个route×seed×leaf的指标从event预测独立重算。", "- 全部paired bootstrap按subject-day在每seed分别执行2000次。逐情绪指标、预测尺度和所有区间见CSV。", "- centered r描述去除subject均值后的event关联；原EMA没有逐窗情绪真值。", "- 历史test已查看，结果属于固定单配置的探索性配对实验；保留原ABC决策，不自动扩展时间范围/结构/协议。", "", "[逐情绪指标](per_label_metrics.csv) · [配对差](paired_deltas.csv) · [配对区间](paired_bootstrap.csv) · [冻结val记录](selection_val_only.json) · [完整JSON](results.json)", ""]
    (out / "basic_event_pool_report.md").write_text("\n".join(lines), encoding="utf-8")
    write(out / "completion_audit.json", dict(status="pass", formal_cells=3, checkpoint_replays=audits,
        metrics_recomputed_leaves=18, training_selection_test_free=True, original_sources_preserved=True))
    (root / "BASIC_EVENT_POOL_COMPLETE").write_text("pass\n")
    progress(root, "all", status="complete", test=contrasts["test"])

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "smoke", "formal", "report", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(True)
    try:
        if args.stage in ("preflight", "all"):
            preflight(args.root, args.prior_root, args.device)
        if args.stage in ("smoke", "all"):
            train(args.root, args.prior_root, "smoke", args.device)
        if args.stage in ("formal", "all"):
            train(args.root, args.prior_root, "formal", args.device)
        if args.stage in ("report", "all"):
            freeze_validation(args.root)
            report(args.root, args.prior_root, args.device)
    except Exception:
        write(args.root / "FAILURE.json", dict(stage=args.stage, error=traceback.format_exc()))
        progress(args.root, args.stage, status="failed")
        raise

if __name__ == "__main__":
    main()
