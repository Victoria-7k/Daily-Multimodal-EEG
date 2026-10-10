#!/usr/bin/env python3
"""Execute only round-2 experiment A; preserve first-round queues and controls."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_jsonl, load_window_split
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.mae_emotion_adaptation import train_adaptation
from daily_multimodal.training.structure_emotion import event_targets, run_condition

install_numpy_core_pickle_aliases()
ALIGNED = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DATA = Path("/vePFS-0x0d/DailyEEG")
V4 = Path("/home/wangzw/mae_norm_channel_20261009")
R1 = Path("/home/wangzw/mae_round1_input_20261010")
REFERENCE = ALIGNED / "outputs/multiemotion_20260913/structure_matrix_A1"
SOURCES = {"C": V4 / "outputs/stage_a_train_channel/formal/cross_day/eeg_seed_240800",
           "P": R1 / "stage_a/E_POOL/formal/cross_day/eeg_seed_240800"}
SEEDS = (240800, 240801, 240802)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def runner():
    p = Path(__file__).with_name("118_run_mae_mt11_event_ablation.py")
    spec = importlib.util.spec_from_file_location("round2_frozen_bag", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def adaptation_dir(root, source, mode, phase):
    return root / f"adaptation/A_{source}_{mode}/{phase}/cross_day/seed_240800"


def fusion_dir(root, variant, phase, seed):
    return root / f"fusion/{variant}/{phase}/cross_day/seed_{seed}"


def controls(source, seed):
    if source == "C":
        return V4 / f"outputs/downstream_train_channel/single/formal/runs/cross_day/E1/seed_{seed}"
    return R1 / f"downstream/E_POOL/formal/runs/cross_day/E_POOL/seed_{seed}"


def make_bag(root, source, token, variant):
    m = runner()
    path = root / f"bags/cross_day/{variant}/ema_bags.npz"
    reference = m._reference_bag(REFERENCE, "cross_day")
    m._write_replaced_bag(reference=reference, mae_paths={"eeg": token}, destination=path, condition="E1")
    with np.load(path, allow_pickle=True) as z:
        payload = {k: z[k] for k in z.files}
    payload["route_id"] = np.asarray(variant)
    adapted = variant.startswith("F_A_")
    payload["supervision_boundary"] = np.asarray("11label_supervised_eeg_adaptation__frozen_fusion" if adapted else "label_free_SSL_EEG__frozen_fusion")
    sources = json.loads(str(payload["source_npz_json"].item()))
    sources["eeg_11label_supervised_adaptation" if adapted else "eeg_mae_stage_a_label_free"] = str(token)
    if adapted:
        sources.pop("eeg_mae_stage_a_label_free", None)
    payload["source_npz_json"] = np.asarray(json.dumps(sources, sort_keys=True))
    np.savez_compressed(path, **payload)
    with np.load(reference, allow_pickle=True) as b:
        for k in ("sample_id_matrix", "event_id", "subject_id", "day_id", "split", "label"):
            if k in b.files and not np.array_equal(b[k], payload[k]):
                raise ValueError("replacement membership changed: " + k)
        if not np.array_equal(b["tokens"][:, :, 1:], payload["tokens"][:, :, 1:]) or not np.array_equal(b["modality_mask"], payload["modality_mask"]):
            raise ValueError("replacement non-EEG tokens or native mask changed")
    return path


def stage0(root):
    rows = load_jsonl(ALIGNED / "index/eeg_aligned_window_index.jsonl")
    expected = np.asarray([str(r["sample_id"]) for r in rows])
    if len(rows) != 28819 or len(set(expected)) != 28819:
        raise ValueError("canonical identity changed")
    y = np.load(DATA / "processed_cadt_addtime_new/y.npy", mmap_mode="r")
    raw = np.load(DATA / "processed_cadt_addtime_new/X.npy", mmap_mode="r")
    if y.shape != (28819, 11) or raw.shape != (28819, 2000, 59):
        raise ValueError("canonical signal/label shape changed")
    index_y = np.asarray([r["labels"] for r in rows], np.float32)
    raw_indices = np.asarray([r["eeg_sample_index"] for r in rows])
    if not np.array_equal(raw_indices, np.arange(28819)) or not np.array_equal(y, index_y):
        raise ValueError("raw/index row or 11-label order mismatch")
    if any(tuple(r["label_names"]) != LABEL_NAMES for r in rows):
        raise ValueError("label order changed")
    audit_path = R1 / "inputs/eeg/cross_day/filter_report.json"
    overlap = json.loads(audit_path.read_text())
    if overlap["status"] != "pass" or overlap["canonical_train_holdout_overlap_count"] != 0 or overlap["canonical_train_holdout_preprocessing_context_overlap_count"] != 0:
        raise ValueError("cross_day support/context isolation failed")
    for path, h in overlap["provenance"].items():
        if sha(path) != h:
            raise ValueError("input provenance changed: " + path)
    for marker in (R1 / "EEG_STAGE_A_FORMAL_COMPLETE", R1 / "EEG_STAGE_B_FORMAL_COMPLETE"):
        if not marker.exists():
            raise ValueError("P source has not completed upstream validation")
    leaf = load_window_split(DATA / "splits_new/cross_day", len(rows))
    norms, registry = [], {}
    reuse = {"status": "pass", "implementation_sha256": {}, "cells": []}
    for rel in ("src/daily_multimodal/training/structure_emotion.py",
                "src/daily_multimodal/training/multihead_regression.py",
                "src/daily_multimodal/daily_affect/regression.py",
                "src/daily_multimodal/daily_affect/training.py"):
        if (root / rel).read_bytes() != (V4 / rel).read_bytes() or (root / rel).read_bytes() != (R1 / rel).read_bytes():
            raise ValueError("fusion implementation mismatch prevents exact reuse: " + rel)
        reuse["implementation_sha256"][rel] = sha(root / rel)
    for source, directory in SOURCES.items():
        cp = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        manifest = cp["manifest"]
        cfg = json.loads((directory / "config.json").read_text())
        if manifest["protocol"] != "cross_day" or manifest["embedding_seed"] != 240800 or cfg["model"] != {"patch_seconds": 1, "patch_shape": [59, 200], "embedding_dim": 256, "encoder_layers": 6, "heads": 8, "decoder_layers": 2}:
            raise ValueError("SSL source model/protocol mismatch")
        norms.append(manifest["raw_normalization"])
        token = directory / "window_embeddings.npz"
        with np.load(token, allow_pickle=False) as t:
            if not np.array_equal(t["sample_id"].astype(str), expected) or t["embedding"].shape != (28819, 256) or not t["valid_mask"].all() or not np.isfinite(t["embedding"]).all():
                raise ValueError("invalid source canonical token")
        bag = make_bag(root, source, token, "F_" + source)
        ds = load_bag_dataset(bag)
        targets = event_targets(ds, rows)
        sp = ds.split_indices()
        if {k: len(sp[k]) for k in ("train", "val", "test")} != {"train": 731, "val": 269, "test": 253}:
            raise ValueError("event split counts changed")
        positions = {sid: i for i, sid in enumerate(expected)}
        for k in ("train", "val", "test"):
            idx = sp[k]
            wr = np.asarray([positions[sid] for sid in ds.sample_id_matrix[idx].ravel()])
            allowed = ("pretrain", "finetune") if k == "train" else (k,)
            if not np.isin(leaf[wr], allowed).all():
                raise ValueError("bag differs from canonical split")
        daysets = [{(rows[i]["subject_id"], rows[i]["day_id"]) for i in np.flatnonzero(np.isin(leaf, v))}
                   for v in (("pretrain", "finetune"), ("val",), ("test",))]
        if any(daysets[a] & daysets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("cross_day preprocessing session overlap")
        with np.load(bag, allow_pickle=True) as b:
            control_bag = controls(source, 240800).parents[3] / f"bags/cross_day/{'E1' if source == 'C' else 'E_POOL'}/seed_240800/ema_bags.npz"
            # parents[3] is the formal root of the historical runs tree.
            with np.load(control_bag, allow_pickle=True) as c:
                for k in b.files:
                    if k in ("route_id", "source_npz_json", "supervision_boundary"):
                        continue
                    if k not in c.files or not np.array_equal(b[k], c[k]):
                        raise ValueError("reused carrier mismatch: " + k)
        for seed in SEEDS:
            d = controls(source, seed)
            m = json.loads((d / "metrics.json").read_text())
            if m["status"] != "ok" or m["downstream_seed"] != seed or m["selection_metric"] != "val_macro_standardized_rmse_min":
                raise ValueError("reused control status changed")
            if m["train_target_mean"] != targets[sp["train"]].mean(axis=0).tolist():
                raise ValueError("reused target normalization mismatch")
            with np.load(d / "event_predictions.npz", allow_pickle=False) as z:
                for k in ("val", "test"):
                    for field, expected_value in (("event_id", ds.event_id[sp[k]]), ("target", targets[sp[k]]),
                                                  ("subject_id", ds.subject_id[sp[k]]), ("day_id", ds.day_id[sp[k]])):
                        if not np.array_equal(z[k + "_" + field], expected_value):
                            raise ValueError("reused predictions membership mismatch")
            reuse["cells"].append({"source": source, "seed": seed, "directory": str(d),
                                    "metrics_sha256": sha(d / "metrics.json"), "predictions_sha256": sha(d / "event_predictions.npz")})
        registry[source] = {"directory": str(directory), "sha256": {name: sha(directory / name) for name in ("checkpoint.pt", "config.json", "window_embeddings.npz")},
                            "normalization": manifest["raw_normalization"], "first_round_gate_passed": False if source == "P" else None}
    if norms[0] != norms[1]:
        raise ValueError("C/P raw normalization differs")
    config = {"scope": "experiment_A_only", "protocol": "cross_day", "adaptation_seed": 240800,
              "downstream_seeds": list(SEEDS), "label_names": list(LABEL_NAMES),
              "encoder_blocks": [4, 5], "encoder_lr": 1e-5, "head_lr": 1e-3,
              "max_epochs": 80, "patience": 15, "event_batch_size": 64,
              "epochs_smoke": 10, "fusion_epochs_smoke": 3, "loss": "window_mse_event_equal",
              "lambda_reserved_B_C": [1, 0, 0.1], "B_C_started": False,
              "expected_new_adaptation_cells": 4, "expected_new_fusion_cells": 6,
              "expected_reused_fusion_cells": 6, "historical_test_seen": True,
              "evidence_status": "exploratory_fixed_plan", "git_source": json.loads((root / "inputs/local_source_snapshot.json").read_text())}
    for name, value in (("fixed_config.json", config), ("source_registry.json", registry),
                        ("reuse_audit.json", reuse), ("protocol_audit.json", {"cross_day": {"status": "pass", "event_counts": [731, 269, 253], "signal_context_audit": str(audit_path), "audit_sha256": sha(audit_path)},
                        "within_subject_day": {"status": "stopped_protocol_overlap", "next": "pending_protocol_revision", "overlapping_windows": 107}})):
        path = root / "inputs" / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("frozen Stage-0 contract changed: " + name)
        write(path, value)
    (root / "STAGE0_COMPLETE").write_text("pass\n")
    print("stage0_pass C/P identity, split, labels, raw normalization and six controls", flush=True)


def execute(root, phase, device):
    if not (root / "STAGE0_COMPLETE").exists():
        raise ValueError("Stage 0 required")
    rows = load_jsonl(ALIGNED / "index/eeg_aligned_window_index.jsonl")
    for source, directory in SOURCES.items():
        prefix = root / f"prefixes/{source}/cross_day/last2"
        if not (prefix / "PREFIX_COMPLETE").exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name("123_export_mae_frozen_prefix.py")),
                            "--modality", "eeg", "--initialization", "pretrained", "--checkpoint", str(directory / "checkpoint.pt"),
                            "--token-file", str(directory / "window_embeddings.npz"), "--out-dir", str(prefix),
                            "--eeg-source", str(DATA / "processed_cadt_addtime_new/X.npy"),
                            "--match-stage-a-batch-size", "--device", device], check=True)
        ds = load_bag_dataset(root / f"bags/cross_day/F_{source}/ema_bags.npz")
        targets = event_targets(ds, rows)
        for mode in ("PROBE", "ADAPT"):
            out = adaptation_dir(root, source, mode, phase)
            if (out / "ADAPTATION_COMPLETE").exists():
                print("retaining " + str(out), flush=True)
                continue
            train_adaptation(dataset=ds, targets=targets, prefix_dir=prefix,
                             source_checkpoint=directory / "checkpoint.pt", source_token=directory / "window_embeddings.npz",
                             source_id=source, mode=mode, out_dir=out, smoke=phase == "smoke", device=device)
    for source in SOURCES:
        probe = json.loads((adaptation_dir(root, source, "PROBE", phase) / "config.json").read_text())
        adapt = json.loads((adaptation_dir(root, source, "ADAPT", phase) / "config.json").read_text())
        for key in ("head_initial_sha256", "train_events", "val_events", "feature_mean", "feature_std", "train_target_mean", "train_target_std"):
            if probe[key] != adapt[key]:
                raise ValueError("PROBE/ADAPT matching failed: " + key)
    (root / ("A_ADAPTATION_SMOKE_COMPLETE" if phase == "smoke" else "A_ADAPTATION_FORMAL_COMPLETE")).write_text("pass\n")


def fusion_worker(root, source, phase, seed, device):
    torch.set_num_threads(4)
    token = adaptation_dir(root, source, "ADAPT", "formal") / "window_embeddings.npz"
    variant = "F_A_" + source
    out = fusion_dir(root, variant, phase, seed)
    if (out / "FUSION_COMPLETE").exists():
        return
    bag = make_bag(root, source, token, variant)
    ds = load_bag_dataset(bag)
    targets = event_targets(ds, load_jsonl(ALIGNED / "index/eeg_aligned_window_index.jsonl"))
    smoke = phase == "smoke"
    if smoke:
        # Existing trainer evaluates two leaves; alias both to validation for
        # interface smoke, then remove the alias artifacts. No test forward.
        ds = replace(ds, test_index=ds.val_index.copy())
    run_condition(dataset=ds, targets=targets, protocol="cross_day",
                  condition_id="window_attention_regression_full_mean", model_id="window_replicated",
                  temporal_policy="uniform", seed=seed, out_dir=out, epochs=3 if smoke else 80,
                  patience=3 if smoke else 15, batch_size=64, hidden_dim=128, learning_rate=1e-3,
                  weight_decay=1e-4, dropout=0.1, modality_dropout_prob=0.1, device=device)
    m = json.loads((out / "metrics.json").read_text())
    m.update(candidate_condition=variant, route_id=variant, ssl_source=source,
             adaptation_seed=240800, source_token=str(token), smoke=smoke,
             supervision_boundary="11label_supervised_eeg_adaptation__frozen_fusion")
    if smoke:
        m.pop("test", None)
        with np.load(out / "event_predictions.npz", allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if not k.startswith("test_")}
        np.savez_compressed(out / "event_predictions.npz", **arrays)
    for leaf in (("val",) if smoke else ("val", "test")):
        for label in LABEL_NAMES:
            if not all(np.isfinite(m[leaf]["per_label"][label][k]) for k in ("raw_r", "within_subject_centered_r", "rmse", "standardized_rmse", "mae")):
                raise ValueError("fusion metrics are not finite")
    cp = torch.load(out / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    cp.update(modality_token_sources=json.loads(ds.source_npz_json), ssl_source=source,
              adaptation_seed=240800, adapted_encoder_checkpoint=str(token.with_name("best_checkpoint.pt")),
              encoder_training_scope="all_frozen", variant=variant)
    torch.save(cp, out / "best_checkpoint.pt")
    write(out / "metrics.json", m)
    (out / "FUSION_COMPLETE").write_text("pass\n")
    print(f"fusion_complete {variant} {phase} seed={seed}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--phase", choices=("all", "stage0", "smoke", "formal", "fusion_worker"), default="all")
    p.add_argument("--source", choices=("C", "P"))
    p.add_argument("--fusion-phase", choices=("smoke", "formal"))
    p.add_argument("--seed", type=int, default=240800)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    try:
        if a.phase == "fusion_worker":
            fusion_worker(a.root, a.source, a.fusion_phase, a.seed, a.device)
            return
        # Linux flock guards the entire experiment-A queue from duplicate launch.
        import fcntl
        with (a.root / "experiment_a.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if a.phase in ("all", "stage0"):
                stage0(a.root)
            if a.phase in ("all", "smoke"):
                execute(a.root, "smoke", a.device)
            if a.phase in ("all", "formal"):
                if not (a.root / "A_ADAPTATION_SMOKE_COMPLETE").exists():
                    raise ValueError("all four adaptation smoke gates required")
                execute(a.root, "formal", a.device)
                for phase in ("smoke", "formal"):
                    for source in SOURCES:
                        for seed in ((240800,) if phase == "smoke" else SEEDS):
                            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--root", str(a.root),
                                            "--phase", "fusion_worker", "--source", source, "--fusion-phase", phase,
                                            "--seed", str(seed), "--device", a.device], check=True)
                    (a.root / f"A_FUSION_{phase.upper()}_COMPLETE").write_text("pass\n")
                subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("140_report_round2_experiment_a.py")),
                                "--root", str(a.root)], check=True)
    except Exception as e:
        write(a.root / "A_FAILURE.json", {"phase": a.phase, "error": str(e), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
