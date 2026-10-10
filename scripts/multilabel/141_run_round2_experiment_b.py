#!/usr/bin/env python3
"""Execute only round-2 B; original C initialization, exact frozen controls."""
from __future__ import annotations
import argparse
import copy
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
from daily_multimodal.daily_affect.ema_bags import load_jsonl
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.structure_emotion import event_targets
from daily_multimodal.training.mae_eeg_partial_ft import VARIANTS, train_ft

install_numpy_core_pickle_aliases()

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

a = module("round2_a_contract", "139_run_round2_experiment_a.py")

def last1_prefix(root, device):
    """Extend the immutable original-C cut4 through its original frozen block4."""
    src = root / "prefixes/C/cross_day/last2"
    dst = root / "prefixes/C/cross_day/last1"
    if (dst / "PREFIX_COMPLETE").exists():
        pm = json.loads((dst / "manifest.json").read_text())
        if a.sha(dst / "prefix.npy") != pm["prefix_sha256"]:
            raise ValueError("last1 prefix changed")
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    pm = json.loads((src / "manifest.json").read_text())
    if a.sha(src / "prefix.npy") != pm["prefix_sha256"]:
        raise ValueError("original frozen prefix changed")
    payload = torch.load(src / "tail.pt", map_location="cpu", weights_only=False)
    layer = torch.nn.TransformerEncoderLayer(256, 8, 1024, batch_first=True, activation="gelu")
    original = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
    original.load_state_dict(payload["state_dict"], strict=True)
    first, last = copy.deepcopy(original.layers[0]).to(device).eval(), copy.deepcopy(original.layers[1]).to(device).eval()
    old = np.load(src / "prefix.npy", mmap_mode="r")
    new = np.lib.format.open_memmap(dst / "prefix.npy", mode="w+", dtype=np.float32, shape=old.shape)
    with np.load(a.SOURCES["C"] / "window_embeddings.npz", allow_pickle=False) as z:
        tokens = z["embedding"]
    maximum = 0.
    with torch.no_grad():
        for start in range(0, len(old), 128):
            end = min(start + 128, len(old))
            hidden = first(torch.as_tensor(np.array(old[start:end]), device=device))
            new[start:end] = hidden.cpu().numpy()
            live = last(hidden).mean(dim=1).cpu().numpy()
            maximum = max(maximum, float(np.abs(live - tokens[start:end]).max()))
    new.flush()
    if maximum > 2e-4:
        raise ValueError(f"last1 composition error: {maximum}")
    tail = torch.nn.TransformerEncoder(copy.deepcopy(original.layers[1]), 1, enable_nested_tensor=False)
    torch.save({"heads": 8, "blocks": 1, "state_dict": tail.state_dict()}, dst / "tail.pt")
    shutil.copy2(src / "window_embeddings.npz", dst / "window_embeddings.npz")
    pm.update(frozen_encoder_blocks=5, trainable_tail_blocks=1, prefix_sha256=a.sha(dst / "prefix.npy"),
        tail_sha256=a.sha(dst / "tail.pt"), parent_prefix=str(src), parent_prefix_sha256=a.sha(src / "prefix.npy"),
        derivation="original_C_frozen_block4_on_original_C_cut4", stage_a_embedding_max_abs_error=maximum,
        equivalence_batch_size=128, prefix_shape=list(old.shape))
    a.write(dst / "manifest.json", pm)
    (dst / "PREFIX_COMPLETE").write_text("complete\n")
    return dst

def stage0(root):
    for marker in ("EXPERIMENT_A_COMPLETE_WITH_PROTOCOL_STOP", "EXPERIMENT_A_REPLAY_VERIFIED"):
        if not (root / marker).exists():
            raise ValueError("missing A prerequisite: " + marker)
    registry = json.loads((root / "inputs/source_registry.json").read_text())["C"]
    for name, h in registry["sha256"].items():
        if a.sha(Path(registry["directory"]) / name) != h:
            raise ValueError("C source changed")
    reuse = json.loads((root / "inputs/reuse_audit.json").read_text())
    for row in reuse["cells"]:
        if row["source"] == "C":
            p = Path(row["directory"])
            if a.sha(p / "metrics.json") != row["metrics_sha256"] or a.sha(p / "event_predictions.npz") != row["predictions_sha256"]:
                raise ValueError("frozen C control changed")
    for name, h in reuse["implementation_sha256"].items():
        if a.sha(root / name) != h:
            raise ValueError("matched fusion source changed")
    overlap = json.loads((a.R1 / "inputs/eeg/cross_day/filter_report.json").read_text())
    if overlap["status"] != "pass" or overlap["canonical_train_holdout_preprocessing_context_overlap_count"] or overlap["canonical_train_holdout_overlap_count"]:
        raise ValueError("strict cross-day contract changed")
    for name, h in overlap["provenance"].items():
        if a.sha(name) != h:
            raise ValueError("strict input provenance changed")
    a.write(root / "inputs/b_fixed_config.json", {"scope": "experiment_B_only", "variants": VARIANTS,
        "seeds": list(a.SEEDS), "ssl_source": registry, "cross_day": "pass", "within_subject_day": "stopped_protocol_overlap",
        "C_started": False, "frozen_control": "F_C_reuse", "batch_size": 64, "epochs": 80, "patience": 15,
        "window_chunk": 128, "selection": "val_only", "A_selection_sha256": a.sha(root / "reports/selection_val_only.json")})
    (root / "B_STAGE0_COMPLETE").write_text("complete\n")

def check_matching(root, phase):
    seeds = (240800,) if phase == "smoke" else a.SEEDS
    for seed in seeds:
        configs, audits = [], []
        for v in VARIANTS:
            d = root / f"fusion/{v}/{phase}/cross_day/seed_{seed}"
            configs.append(json.loads((d / "config.json").read_text()))
            audits.append(json.loads((d / "trainability_audit.json").read_text()))
        for key in ("head_initial_sha256", "post_head_cpu_rng_sha256", "post_head_cuda_rng_sha256", "fixed_non_eeg_sha256", "native_mask_sha256", "feature_mean_sha256", "feature_std_sha256", "train_events", "val_events"):
            if any(c[key] != configs[0][key] for c in configs[1:]):
                raise ValueError("matched B initialization failed: " + key)
        if any(v["first_dropout_audit"] != audits[0]["first_dropout_audit"] for v in audits[1:]):
            raise ValueError("first batch/dropout RNG mismatch")
    a.write(root / f"inputs/b_{phase}_matching_audit.json", {"status": "pass", "seeds": list(seeds),
        "matched_heads_rng_batch_dropout_inputs_normalization": True})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    r = args.root
    try:
        torch.set_num_threads(4)
        torch.backends.mha.set_fastpath_enabled(True)
        stage0(r)
        one = last1_prefix(r, args.device)
        (r / "B_PREFIX_COMPLETE").write_text("complete\n")
        ds = load_bag_dataset(r / "bags/cross_day/F_C/ema_bags.npz")
        targets = event_targets(ds, load_jsonl(a.ALIGNED / "index/eeg_aligned_window_index.jsonl"))
        if {k: len(ds.split_indices()[k]) for k in ("train", "val", "test")} != {"train": 731, "val": 269, "test": 253}:
            raise ValueError("B event membership changed")
        for phase in ("smoke", "formal"):
            for variant, (blocks, _) in VARIANTS.items():
                prefix = one if blocks == 1 else r / "prefixes/C/cross_day/last2"
                for seed in ((240800,) if phase == "smoke" else a.SEEDS):
                    d = r / f"fusion/{variant}/{phase}/cross_day/seed_{seed}"
                    if (d / "FT_COMPLETE").exists():
                        continue
                    train_ft(dataset=ds, targets=targets, prefix_dir=prefix, source_checkpoint=a.SOURCES["C"] / "checkpoint.pt",
                        variant=variant, seed=seed, out_dir=d, smoke=phase == "smoke", device=args.device)
            check_matching(r, phase)
            (r / f"B_{phase.upper()}_COMPLETE").write_text("complete\n")
    except Exception:
        a.write(r / "B_FAILURE.json", {"traceback": traceback.format_exc()})
        raise
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
