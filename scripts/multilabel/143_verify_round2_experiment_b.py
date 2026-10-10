#!/usr/bin/env python3
"""Independent raw-encoder and fused prediction replay; no test metric disclosure."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.training import load_bag_dataset, fit_token_normalization
from daily_multimodal.training.modality_mae import TemporalMaskedAutoencoder, eeg_to_patches, embedding_health, require_healthy
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    args = p.parse_args()
    r = args.root
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(True)
    if not (r / "EXPERIMENT_B_COMPLETE_WITH_PROTOCOL_STOP").exists():
        raise ValueError("B report/val freeze missing")
    manifest = json.loads((r / "inputs/b_execution_source_manifest.json").read_text())
    for n, h in manifest["files"].items():
        if sha(r / n) != h:
            raise ValueError("B source changed: " + n)
    registry = json.loads((r / "inputs/source_registry.json").read_text())["C"]
    for n, h in registry["sha256"].items():
        if sha(Path(registry["directory"]) / n) != h:
            raise ValueError("original C source changed")
    original = torch.load(Path(registry["directory"]) / "checkpoint.pt", map_location="cpu", weights_only=False)
    ds = load_bag_dataset(r / "bags/cross_day/F_C/ema_bags.npz")
    expected = np.load('/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new/X.npy', mmap_mode="r")
    xm, xs = fit_token_normalization(ds.tokens, ds.modality_mask, ds.split_indices()["train"], scope="per_modality")
    audit = {"status": "pass", "scope": "B_only", "source_hashes_verified": True,
        "selection_sha256": sha(r / "reports/experiment_b/selection_val_only.json"),
        "test_performance_disclosed": False, "within_subject_day": "stopped_protocol_overlap", "cells": []}
    dev = torch.device("cuda")
    for blocks in (1, 2):
        d = r / f"prefixes/C/cross_day/last{blocks}"
        pm = json.loads((d / "manifest.json").read_text())
        if sha(d / "prefix.npy") != pm["prefix_sha256"]:
            raise ValueError("immutable prefix changed")
    for variant in ("B_T2_STD", "B_T2_LOW", "B_T1_LOW"):
        for seed in (240800, 240801, 240802):
            d = r / f"fusion/{variant}/formal/cross_day/seed_{seed}"
            cp = torch.load(d / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            blocks = len(cp["config"]["trainable_encoder_blocks"])
            cut = 6 - blocks
            updated = []
            for n, v in cp["encoder_state_dict"].items():
                tail_parameter = n.startswith(tuple(f"encoder.layers.{b}." for b in range(cut, 6)))
                if not torch.equal(v, original["state_dict"][n]):
                    if not tail_parameter:
                        raise ValueError("frozen encoder weight changed: " + n)
                    updated.append(n)
            if not updated or cp["raw_normalization"] != original["manifest"]["raw_normalization"]:
                raise ValueError("tail update/raw normalization failure")
            if not np.array_equal(cp["x_mean"], xm) or not np.array_equal(cp["x_std"], xs):
                raise ValueError("fixed normalization mismatch")
            encoder = TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=256, encoder_layers=6, decoder_layers=2, heads=8)
            missing, unexpected = encoder.load_state_dict(cp["encoder_state_dict"], strict=False)
            if unexpected or any(not n.startswith(("decoder.", "reconstruction.", "mask_token")) for n in missing):
                raise ValueError("standalone encoder incomplete")
            encoder.to(dev).eval()
            # Build an independent tail directly from the full selected encoder.
            import copy
            tail = torch.nn.TransformerEncoder(copy.deepcopy(encoder.encoder.layers[cut]), blocks, enable_nested_tensor=False)
            for b in range(blocks):
                tail.layers[b].load_state_dict(encoder.encoder.layers[cut + b].state_dict())
            tail.to(dev).eval()
            head = MultiEmotionStructureModel(DailyAffectRegressionConfig(model_id="window_replicated", hidden_dim=128, dropout=.1,
                adapter_mode="per_modality", temporal_policy="uniform", lambda_d=.25)).to(dev).eval()
            head.load_state_dict({k.removeprefix("head."): v for k, v in cp["state_dict"].items() if k.startswith("head.")}, strict=True)
            for n, v in tail.state_dict().items():
                if not torch.equal(v.cpu(), cp["state_dict"]["tail." + n]):
                    raise ValueError("inference encoder / trained tail mismatch")
            pd = r / f"prefixes/C/cross_day/last{blocks}"
            prefix = np.load(pd / "prefix.npy", mmap_mode="r")
            with np.load(pd / "window_embeddings.npz", allow_pickle=False) as z:
                ids = z["sample_id"].astype(str)
            lookup = {v: i for i, v in enumerate(ids)}
            rows = np.asarray([lookup[v] for v in ds.sample_id_matrix.astype(str).ravel()]).reshape(1253, 23)
            maximum = 0.
            with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
                for start in (0, 128, 4096, 8192, 12288, 16384, 20480, 24576, 28672):
                    idx = np.arange(start, start + 128)
                    patch = torch.as_tensor(eeg_to_patches(expected[idx], cp["raw_normalization"]), device=dev)
                    raw = encoder.encode(patch).mean(dim=1)
                    cached = tail(torch.as_tensor(np.array(prefix[idx]), device=dev)).mean(dim=1)
                    maximum = max(maximum, float((raw - cached).abs().max()))
                if maximum > 2e-4:
                    raise ValueError(f"raw encoder equivalence failed: {maximum}")
                errors = {}
                with np.load(d / "event_predictions.npz", allow_pickle=False) as saved:
                    for leaf in ("val", "test"):
                        ix = ds.split_indices()[leaf]
                        predictions = []
                        for start in range(0, len(ix), 64):
                            batch = ix[start:start + 64]
                            rr = rows[batch].ravel()
                            eeg = torch.cat([tail(torch.as_tensor(np.array(prefix[rr[s:s + 128]]), device=dev)).mean(dim=1) for s in range(0, len(rr), 128)])
                            tokens = torch.as_tensor(ds.tokens[batch].copy(), device=dev)
                            tokens[:, :, 0] = eeg.reshape(len(batch), 23, 256)
                            normalized = (tokens - torch.as_tensor(cp["x_mean"], device=dev)) / torch.as_tensor(cp["x_std"], device=dev)
                            output = head(normalized, torch.as_tensor(ds.modality_mask[batch], device=dev))["prediction"].cpu().numpy()
                            predictions.append((output * cp["y_std"] + cp["y_mean"]).astype(np.float32))
                        actual = np.concatenate(predictions)
                        err = float(np.abs(actual - saved[leaf + "_prediction"]).max())
                        if not np.isfinite(actual).all() or err > 1e-6:
                            raise ValueError(f"independent fused replay failed: {variant} {seed} {leaf} {err}")
                        for k, v in (("event_id", ds.event_id), ("subject_id", ds.subject_id), ("day_id", ds.day_id)):
                            if not np.array_equal(v[ix], saved[leaf + "_" + k]):
                                raise ValueError("prediction membership mismatch")
                        errors[leaf] = {"count": len(ix), "max_abs_error": err}
                vr = rows[ds.split_indices()["val"]].ravel()
                vr = vr[np.linspace(0, len(vr) - 1, 256, dtype=int)]
                live = torch.cat([tail(torch.as_tensor(np.array(prefix[vr[s:s + 128]]), device=dev)).mean(dim=1) for s in range(0, len(vr), 128)]).cpu().numpy()
                health = embedding_health(live, robust=True)
                require_healthy(health, f"B {variant} {seed}")
            audit["cells"].append({"variant": variant, "seed": seed, "frozen_parameters_verified": True, "updated_parameter_count": len(updated),
                "raw_replay_windows": 1152, "raw_replay_max_abs_error": maximum, "prediction_replay": errors,
                "embedding_health": health, "checkpoint_sha256": sha(d / "best_checkpoint.pt")})
            print(json.dumps(audit["cells"][-1]), flush=True)
            del head, tail, encoder
    (r / "reports/experiment_b/completion_audit.json").write_text(json.dumps(audit, indent=2))
    (r / "EXPERIMENT_B_REPLAY_VERIFIED").write_text("pass\n")

if __name__ == "__main__":
    main()
