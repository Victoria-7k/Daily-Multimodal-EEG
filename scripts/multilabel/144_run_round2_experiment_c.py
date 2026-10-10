#!/usr/bin/env python3
"""C: MAE/B0 paired event objectives; independent of A/B learned weights."""
import argparse
from dataclasses import replace
import importlib.util
import json
import sys
import traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_jsonl
from daily_multimodal.daily_affect.training import load_bag_dataset, fit_token_normalization
from daily_multimodal.training.structure_emotion import event_targets, run_condition
from daily_multimodal.training.event_loss_adaptation import VARIANTS, train_event_loss
from daily_multimodal.training.mae_emotion_adaptation import state_hash

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

a = load("round2_original_contract", "139_run_round2_experiment_a.py")
h = load("round2_val_helpers", "140_report_round2_experiment_a.py")

def controls(source, seed):
    if source == "C":
        return a.controls(source, seed)
    return a.REFERENCE / f"runs/cross_day/window_attention_regression_full_mean/seed_{seed}"

def stage0(root):
    for marker in ("EXPERIMENT_A_REPLAY_VERIFIED", "EXPERIMENT_B_REPLAY_VERIFIED"):
        if not (root / marker).exists():
            raise ValueError("A/B prerequisites missing")
    for manifest in ("execution_source_manifest.json", "b_execution_source_manifest.json"):
        for n, sha in json.loads((root / "inputs" / manifest).read_text())["files"].items():
            if a.sha(root / n) != sha:
                raise ValueError("A/B immutable source changed: " + n)
    bag_paths = {"C": root / "bags/cross_day/F_C/ema_bags.npz", "B0": a.runner()._reference_bag(a.REFERENCE, "cross_day")}
    datasets = {k: load_bag_dataset(v) for k, v in bag_paths.items()}
    rows = load_jsonl(a.ALIGNED / "index/eeg_aligned_window_index.jsonl")
    targets = {k: event_targets(v, rows) for k, v in datasets.items()}
    for n in ("sample_id_matrix", "event_id", "subject_id", "day_id", "split", "modality_mask"):
        if not np.array_equal(getattr(datasets["C"], n), getattr(datasets["B0"], n)):
            raise ValueError("B0/C identity or native masks differ: " + n)
    if not np.array_equal(datasets["C"].tokens[:, :, 1:], datasets["B0"].tokens[:, :, 1:]) or not np.array_equal(targets["C"], targets["B0"]):
        raise ValueError("B0/C non-EEG inputs or labels differ")
    reuse = []
    for source, ds in datasets.items():
        sp = ds.split_indices()
        if {n: len(sp[n]) for n in ("train", "val", "test")} != {"train": 731, "val": 269, "test": 253}:
            raise ValueError("canonical event counts changed")
        xm, xs = fit_token_normalization(ds.tokens, ds.modality_mask, sp["train"], scope="per_modality")
        ym = targets[source][sp["train"]].mean(axis=0, keepdims=True).astype(np.float32)
        ys = targets[source][sp["train"]].std(axis=0, keepdims=True).astype(np.float32)
        ys = np.where(np.isfinite(ys) & (ys >= 1e-6), ys, 1).astype(np.float32)
        for seed in a.SEEDS:
            directory = controls(source, seed)
            m, arrays, _ = h.read_val(directory)
            if tuple(m["train_target_mean"]) != tuple(ym.ravel().tolist()) or tuple(m["train_target_std"]) != tuple(ys.ravel().tolist()):
                raise ValueError("reused label normalization differs")
            for key, value in (("event_id", ds.event_id), ("subject_id", ds.subject_id), ("day_id", ds.day_id), ("target", targets[source])):
                if not np.array_equal(value[sp["val"]], arrays[key]):
                    raise ValueError("reused control val membership differs")
            cp = torch.load(directory / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            for k, expected in (("x_mean", xm), ("x_std", xs), ("y_mean", ym), ("y_std", ys)):
                if not np.array_equal(cp[k], expected):
                    raise ValueError("reused normalization mismatch: " + k)
            if m["model_id"] != "window_replicated" or m["head_variant"] != "H1_shared2_11xhead2" or m["temporal_policy"] != "uniform":
                raise ValueError("reused downstream structure mismatch")
            reuse.append({"source": source, "seed": seed, "directory": str(directory),
                "sha256": {n: a.sha(directory / n) for n in ("metrics.json", "event_predictions.npz", "best_checkpoint.pt")}})
    overlap = json.loads((a.R1 / "inputs/eeg/cross_day/filter_report.json").read_text())
    if overlap["status"] != "pass" or overlap["canonical_train_holdout_overlap_count"] or overlap["canonical_train_holdout_preprocessing_context_overlap_count"]:
        raise ValueError("cross_day support isolation changed")
    for n, sha in overlap["provenance"].items():
        if a.sha(n) != sha:
            raise ValueError("canonical input provenance changed")
    contract = {"scope": "experiment_C_only", "variants": VARIANTS, "seeds": list(a.SEEDS),
        "bag_paths": {k: str(v) for k, v in bag_paths.items()}, "bag_sha256": {k: a.sha(v) for k, v in bag_paths.items()},
        "reuse": reuse, "within_subject_day": "stopped_protocol_overlap", "cross_day": "pass",
        "label_names": list(LABEL_NAMES), "source_initialization": "original_frozen_C_and_B0__A_B_outputs_not_used",
        "val_selection": "fixed_joint_gate_then_mean_sRMSE_tie_1e-6_prefer_weak",
        "epochs": 80, "patience": 15, "smoke_epochs": 3, "batch_size": 64}
    path = root / "inputs/c_fixed_config.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("C frozen contract changed")
    a.write(path, contract)
    (root / "C_STAGE0_COMPLETE").write_text("pass\n")
    return datasets, targets

def verify_control_path(root, datasets, targets, device):
    if (root / "C_CONTROL_EQUIVALENCE_VERIFIED").exists():
        return
    records = []
    for source, ds in datasets.items():
        new = root / f"control_equivalence/{source}/new"
        legacy = root / f"control_equivalence/{source}/legacy"
        variant = "C_EVENT" if source == "C" else "B0_EVENT"
        train_event_loss(dataset=ds, targets=targets[source], variant=variant, seed=240800, out_dir=new,
            smoke=True, device=device, weight_override=1.)
        # The original trainer's test alias contains validation events only.
        run_condition(dataset=replace(ds, test_index=ds.val_index.copy()), targets=targets[source], protocol="cross_day",
            condition_id="window_attention_regression_full_mean", model_id="window_replicated", temporal_policy="uniform",
            seed=240800, out_dir=legacy, epochs=3, patience=3, device=device)
        nc = torch.load(new / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        lc = torch.load(legacy / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        nm, lm = json.loads((new / "metrics.json").read_text()), json.loads((legacy / "metrics.json").read_text())
        if state_hash(nc["state_dict"]) != state_hash(lc["state_dict"]) or nm["history"] != lm["history"]:
            raise ValueError("lambda1 legacy training/RNG equivalence failed: " + source)
        records.append({"source": source, "epochs": 3, "checkpoint_state_sha256": state_hash(nc["state_dict"]),
            "checkpoint_and_history_exactly_equal": True, "test_forward": "validation_alias_only"})
    a.write(root / "inputs/c_control_equivalence.json", records)
    (root / "C_CONTROL_EQUIVALENCE_VERIFIED").write_text("pass\n")

def matching(root, phase):
    records = []
    for seed in ((240800,) if phase == "smoke" else a.SEEDS):
        configs, audits = [], []
        for variant in VARIANTS:
            d = root / f"fusion/{variant}/{phase}/cross_day/seed_{seed}"
            configs.append(json.loads((d / "config.json").read_text()))
            audits.append(json.loads((d / "matching_audit.json").read_text()))
        for k in ("head_initial_sha256", "head_cpu_rng_sha256", "head_cuda_rng_sha256", "train_events", "val_events", "mask_sha256", "train_target_mean", "train_target_std"):
            if any(v[k] != configs[0][k] for v in configs[1:]):
                raise ValueError("C matched initialization changed: " + k)
        if any(v["first_batch"] != audits[0]["first_batch"] for v in audits[1:]):
            raise ValueError("C first shuffle/dropout differs")
        records.append({"seed": seed, "matched_all_four_heads_rng_members_masks": True})
    a.write(root / f"inputs/c_{phase}_matching_audit.json", records)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    r = args.root
    try:
        torch.set_num_threads(4)
        datasets, targets = stage0(r)
        verify_control_path(r, datasets, targets, args.device)
        for phase in ("smoke", "formal"):
            for variant, (source, _) in VARIANTS.items():
                for seed in ((240800,) if phase == "smoke" else a.SEEDS):
                    d = r / f"fusion/{variant}/{phase}/cross_day/seed_{seed}"
                    if (d / "EVENT_LOSS_COMPLETE").exists():
                        continue
                    train_event_loss(dataset=datasets[source], targets=targets[source], variant=variant, seed=seed,
                        out_dir=d, smoke=phase == "smoke", device=args.device)
            matching(r, phase)
            (r / f"C_{phase.upper()}_COMPLETE").write_text("pass\n")
    except Exception:
        a.write(r / "C_FAILURE.json", {"traceback": traceback.format_exc()})
        raise

if __name__ == "__main__":
    main()
