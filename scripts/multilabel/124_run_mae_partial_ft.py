#!/usr/bin/env python3
"""Matched last-two-block supervised adaptation: M5-FT versus random-init R0.

Frozen prefix activations are losslessly cached in float32, including all video
spatial tokens. Gradients pass through the encoder tails and the unchanged 0814
11-head downstream model. R0 has no reconstruction-pretrained parameters.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[2]
TRAINING_VERSION = "last2_ft_encoder_eval_math_v3"
sys.path.insert(0, str(ROOT / "src"))
from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, load_jsonl
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.samplers import shuffled_batches
from daily_multimodal.daily_affect.training import (
    _seed_everything, apply_modality_dropout, fit_token_normalization, load_bag_dataset,
)
from daily_multimodal.training.multihead_regression import evaluate_event_level
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, event_targets


def frozen_runner():
    path = Path(__file__).with_name("118_run_mae_mt11_event_ablation.py")
    spec = importlib.util.spec_from_file_location("matched_frozen_mae", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csv(value):
    return tuple(x.strip() for x in value.split(",") if x.strip())


def require_finite_tensor(value, context):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"non-finite tensor: {context}; shape={tuple(value.shape)}")


def require_finite_parameters(model, *, gradients, context):
    tensors = [(name, p.grad if gradients else p) for name, p in model.named_parameters()
               if not gradients or p.grad is not None]
    if not tensors:
        raise RuntimeError(f"no gradients: {context}")
    finite = torch.stack([torch.isfinite(value).all() for _, value in tensors])
    if not bool(finite.all()):
        names = [name for (name, _), good in zip(tensors, finite.cpu().tolist()) if not good]
        raise RuntimeError(f"non-finite {'gradients' if gradients else 'parameters'}: {context}; {names}")


class PartialFTModel(torch.nn.Module):
    def __init__(self, dataset, directories, x_mean, x_std, device, chunk_size=128):
        super().__init__()
        # Instantiate first, keeping the downstream initialization matched to 118.
        self.head = MultiEmotionStructureModel(DailyAffectRegressionConfig(
            model_id="window_replicated", hidden_dim=128, dropout=0.1,
            adapter_mode="per_modality", temporal_policy="uniform", lambda_d=0.25,
        )).to(device)
        self.tails = torch.nn.ModuleDict()
        self.prefixes, self.row_maps, self.manifests = {}, {}, {}
        self.chunk_size = chunk_size
        self.register_buffer("x_mean", torch.as_tensor(x_mean, device=device), persistent=False)
        self.register_buffer("x_std", torch.as_tensor(x_std, device=device), persistent=False)
        self.register_buffer("native_mask", torch.as_tensor(dataset.modality_mask, device=device), persistent=False)
        self.dev = device
        for slot, modality in enumerate(("eeg", "wear", "video")):
            directory = directories[modality]
            if not (directory / "PREFIX_COMPLETE").is_file():
                raise RuntimeError(f"prefix not complete: {directory}")
            manifest = json.loads((directory / "manifest.json").read_text())
            payload = torch.load(directory / "tail.pt", map_location="cpu", weights_only=False)
            layer = torch.nn.TransformerEncoderLayer(256, payload["heads"], 1024,
                                                      batch_first=True, activation="gelu")
            tail = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            tail.load_state_dict(payload["state_dict"], strict=True)
            self.tails[modality] = tail.to(device)
            with np.load(directory / "window_embeddings.npz", allow_pickle=False) as token:
                ids = token["sample_id"].astype(str)
                valid_indices = token["valid_indices"].astype(np.int64)
                valid = token["valid_mask"].astype(bool)
            if not np.array_equal(np.flatnonzero(valid), valid_indices):
                raise ValueError("prefix valid-index contract failed")
            position = {value: index for index, value in enumerate(ids.tolist())}
            canonical_rows = np.asarray([position[x] for x in dataset.sample_id_matrix.astype(str).ravel()]).reshape(dataset.sample_id_matrix.shape)
            if not np.array_equal(valid[canonical_rows], dataset.modality_mask[:, :, slot]):
                raise ValueError(f"bag/prefix availability mismatch: {modality}")
            compact_rows = np.full(len(ids), -1, dtype=np.int64)
            compact_rows[valid_indices] = np.arange(len(valid_indices))
            self.row_maps[modality] = torch.as_tensor(compact_rows[canonical_rows], device=device)
            values = np.load(directory / "prefix.npy", mmap_mode="r")
            if list(values.shape) != manifest["prefix_shape"] or values.dtype != np.float32:
                raise ValueError("prefix shape/dtype contract failed")
            # GPU-resident read-only data; deliberately excluded from checkpoints.
            self.prefixes[modality] = torch.from_numpy(np.array(values)).to(device)
            if self.prefixes[modality].requires_grad:
                raise AssertionError("frozen prefix must not require gradients")
            self.manifests[modality] = manifest

    def train(self, mode=True):
        super().train(mode)
        # eval() only disables stochastic encoder dropout; encoder-tail weights
        # remain trainable and autograd stays enabled. The downstream head keeps
        # its original training mode and dropout/modality-dropout settings.
        self.tails.eval()
        return self

    def encode_events(self, indices):
        event_rows = torch.as_tensor(indices, dtype=torch.long, device=self.dev)
        slots = []
        for modality in ("eeg", "wear", "video"):
            rows = self.row_maps[modality][event_rows].reshape(-1)
            valid = rows >= 0
            hidden = self.prefixes[modality][rows[valid]]
            chunks = []
            for start in range(0, len(hidden), self.chunk_size):
                x = hidden[start:start + self.chunk_size]
                if self.training and torch.is_grad_enabled():
                    encoded = checkpoint(self.tails[modality], x, use_reentrant=False)
                else:
                    encoded = self.tails[modality](x)
                chunks.append(encoded.mean(dim=1))
            values = torch.zeros((len(rows), 256), dtype=torch.float32, device=self.dev)
            if chunks:
                values = values.index_copy(0, torch.nonzero(valid).flatten(), torch.cat(chunks))
            slots.append(values.reshape(len(indices), 23, 256))
        slots.append(torch.zeros_like(slots[0]))
        tokens = torch.stack(slots, dim=2)
        normalized = (tokens - self.x_mean) / self.x_std
        require_finite_tensor(normalized, "normalized encoder tokens")
        return normalized, self.native_mask[event_rows]

    def forward(self, indices, rng=None):
        tokens, mask = self.encode_events(indices)
        if rng is not None:
            tokens, mask = apply_modality_dropout(tokens, mask, probability=0.1, rng=rng)
        # Tiny initial video variance can produce large normalized FT features.
        # Keep the same attention equation and fp32 inputs, using the stable
        # math implementation for the four-modality fusion forward/backward.
        with sdpa_kernel([SDPBackend.MATH]):
            return self.head(tokens, mask)


@torch.no_grad()
def predict(model, indices, y_mean, y_std, batch_size):
    model.eval()
    chunks = []
    for start in range(0, len(indices), batch_size):
        prediction = model(indices[start:start + batch_size])["prediction"]
        require_finite_tensor(prediction, "evaluation predictions")
        chunks.append(prediction.cpu().numpy())
    return (np.concatenate(chunks) * y_std + y_mean).astype(np.float32)


def train(args, dataset, targets, directories, condition, protocol, seed, out_dir):
    started = time.time()
    _seed_everything(seed)
    split = dataset.split_indices()
    train_rows, val_rows = split["train"], split["val"]
    x_mean, x_std = fit_token_normalization(dataset.tokens, dataset.modality_mask, train_rows, scope="per_modality")
    y_mean = targets[train_rows].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[train_rows].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std >= 1e-6, y_std, 1.0).astype(np.float32)
    dev = torch.device(args.device)
    model = PartialFTModel(dataset, directories, x_mean, x_std, dev, args.window_chunk_size)
    initialization = "pretrained" if condition == "M5-FT" else "random"
    if any(m["initialization"] != initialization for m in model.manifests.values()):
        raise ValueError("initialization provenance mismatch")
    model.eval()
    # Check the live encoder tail reconstructs the initial frozen-token bag.
    with torch.no_grad():
        check_rows = train_rows[:8]
        live, _ = model.encode_events(check_rows)
        expected = torch.as_tensor((dataset.tokens[check_rows] - x_mean) / x_std, device=dev)
        valid = model.native_mask[check_rows][..., None].expand_as(expected)
        initial_error = float((live - expected).abs()[valid].max().cpu())
        if initial_error > 2e-4:
            raise ValueError(f"initial token equivalence failed: {initial_error}")
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": args.learning_rate},
        {"params": model.tails.parameters(), "lr": args.learning_rate * 0.1},
    ], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_score, best_epoch, stale, best_state = math.inf, 0, 0, None
    history, gradient_audit = [], None
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_number, batch_indices in enumerate(shuffled_batches(train_rows, args.batch_size, rng), 1):
            context = f"protocol={protocol} condition={condition} seed={seed} epoch={epoch + 1} batch={batch_number}"
            target = torch.as_tensor((targets[batch_indices] - y_mean) / y_std, device=dev)
            output = model(batch_indices, rng=rng)
            require_finite_tensor(output["window_prediction"], f"{context} window predictions")
            valid = output["window_mask"].to(target.dtype)
            error = (output["window_prediction"] - target[:, None, :]).square()
            loss = ((error * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]).mean()
            require_finite_tensor(loss, f"{context} loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            require_finite_parameters(model, gradients=True, context=context)
            if gradient_audit is None:
                gradient_audit = {}
                before = {}
                for modality, tail in model.tails.items():
                    norm = sum(float(p.grad.detach().abs().sum().cpu()) for p in tail.parameters() if p.grad is not None)
                    if not math.isfinite(norm) or norm <= 0:
                        raise RuntimeError(f"encoder tail has no finite nonzero gradient: {modality}")
                    gradient_audit[modality] = {"first_backward_grad_abs_sum": norm, "frozen_prefix_requires_grad": False}
                    before[modality] = next(tail.parameters()).detach().clone()
            else:
                before = None
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            require_finite_parameters(model, gradients=False, context=context)
            if before is not None:
                for modality, tail in model.tails.items():
                    delta = float((next(tail.parameters()).detach() - before[modality]).abs().max().cpu())
                    if delta <= 0:
                        raise RuntimeError(f"encoder tail was not updated: {modality}")
                    gradient_audit[modality]["first_step_max_abs_parameter_delta"] = delta
                (out_dir / "gradient_audit.json").write_text(json.dumps(gradient_audit, indent=2))
            losses.append(float(loss.detach().cpu()))
        val_pred = predict(model, val_rows, y_mean, y_std, args.batch_size)
        score = float(np.sqrt(np.mean(((val_pred - targets[val_rows]) / y_std) ** 2, axis=0)).mean())
        if not math.isfinite(score):
            raise RuntimeError(f"non-finite validation score: protocol={protocol} condition={condition} seed={seed} epoch={epoch + 1}")
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_macro_standardized_rmse": score,
                  "elapsed_seconds": time.time() - started}
        history.append(record)
        print(json.dumps({"protocol": protocol, "condition": condition, "seed": seed, **record}), flush=True)
        if score < best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        (out_dir / "progress.json").write_text(json.dumps({"history": history, "best_epoch": best_epoch,
                                                          "best_val_macro_standardized_rmse": best_score}, indent=2))
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(best_state)
    metrics, arrays = {}, {"label_names": np.asarray(LABEL_NAMES)}
    for leaf in ("val", "test"):
        idx = split[leaf]
        pred = predict(model, idx, y_mean, y_std, args.batch_size)
        metrics[leaf] = evaluate_event_level({"target": targets[idx], "prediction": pred,
                                             "subject_id": dataset.subject_id[idx]}, y_std)
        if not np.isfinite(pred).all() or any(not math.isfinite(value["raw_r"])
                                             for value in metrics[leaf]["per_label"].values()):
            raise RuntimeError(f"non-finite {leaf} prediction or raw-r; completion gate failed")
        for key, value in {"event_id": dataset.event_id[idx], "subject_id": dataset.subject_id[idx],
                           "day_id": dataset.day_id[idx], "target": targets[idx], "prediction": pred}.items():
            arrays[f"{leaf}_{key}"] = value
    np.savez_compressed(out_dir / "event_predictions.npz", **arrays)
    torch.save({"state_dict": best_state, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std,
                "prefix_directories": {k: str(v) for k, v in directories.items()}, "prefix_manifests": model.manifests,
                "downstream_seed": seed, "condition": condition, "label_names": LABEL_NAMES,
                "training_version": TRAINING_VERSION, "encoder_dropout_mode": "eval_weights_trainable",
                "fusion_attention_backend": "math_fp32"}, out_dir / "best_checkpoint.pt")
    result = {"status": "ok", "protocol": protocol, "candidate_condition": condition,
              "route_id": f"A1_MT11_reference__{condition}_three_modality_last2_supervised",
              "condition_id": "window_attention_regression_full_mean", "model_id": "window_replicated",
              "temporal_policy": "uniform", "head_variant": "H1_shared2_11xhead2", "normalization": "per_modality",
              "embedding_seed": 240800, "downstream_seed": seed, "bag_path": str(dataset.bag_path),
              "supervision_boundary": f"{initialization}_MAE_init__last2_encoder_blocks_and_11label_head_supervised",
              "encoder_initialization": initialization, "mt11_eeg_retained": False,
              "training_version": TRAINING_VERSION, "encoder_dropout_mode": "eval_weights_trainable",
              "fusion_attention_backend": "math_fp32",
              "numeric_checks": "every_batch_tokens_predictions_loss_gradients_parameters_and_validation",
              "encoder_training_scope": "last_two_transformer_blocks_per_modality; stems_positions_early_blocks_frozen",
              "encoder_lr": args.learning_rate * 0.1, "downstream_lr": args.learning_rate,
              "batch_size": args.batch_size, "window_chunk_size": args.window_chunk_size,
              "selection_metric": "val_macro_standardized_rmse_min", "best_epoch": best_epoch,
              "best_val_macro_standardized_rmse": best_score, "epochs_ran": len(history),
              "duration_seconds": time.time() - started, "normalization_fit": "initial_tokens_train_events_only_fixed_during_ft",
              "train_target_mean": y_mean.ravel().tolist(), "train_target_std": y_std.ravel().tolist(),
              "initial_normalized_token_max_abs_error": initial_error, "gradient_audit": gradient_audit,
              "prefix_manifests": model.manifests, "trainable_params": sum(p.numel() for p in model.parameters()),
              "val": metrics["val"], "test": metrics["test"], "history": history}
    (out_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--prefix-root", required=True, type=Path)
    parser.add_argument("--random-prefix-per-protocol", action="store_true", help="Use protocol-specific random prefixes when raw train statistics differ.")
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--conditions", default="M5-FT,R0")
    parser.add_argument("--protocols", default="cross_day,within_subject_day")
    parser.add_argument("--seeds", default="240800,240801,240802")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-chunk-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    selected, protocols = csv(args.conditions), csv(args.protocols)
    seeds = tuple(map(int, csv(args.seeds)))
    if not seeds or set(selected) - {"M5-FT", "R0"} or set(protocols) - {"cross_day", "within_subject_day"}:
        raise ValueError("unsupported condition, protocol, or empty seeds")
    reference = frozen_runner()
    rows = load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    results = []
    for protocol in protocols:
        for condition in selected:
            initialization = "pretrained" if condition == "M5-FT" else "random"
            leaf = protocol if initialization == "pretrained" or args.random_prefix_per_protocol else "shared"
            directories = {modality: args.prefix_root / initialization / leaf / modality for modality in ("eeg", "wear", "video")}
            tokens = {modality: path / "window_embeddings.npz" for modality, path in directories.items()}
            bag_path = args.out_root / "bags" / protocol / condition / "ema_bags.npz"
            reference._write_replaced_bag(reference=reference._reference_bag(args.reference_root, protocol),
                                          mae_paths=tokens, destination=bag_path, condition="M5-F")
            with np.load(bag_path, allow_pickle=True) as loaded:
                payload = {k: loaded[k] for k in loaded.files}
            payload["route_id"] = np.asarray(f"A1_MT11_reference__{condition}_three_modality_last2_supervised")
            payload["supervision_boundary"] = np.asarray(f"{initialization}_MAE_init__last2_encoder_blocks_and_11label_head_supervised")
            payload["source_npz_json"] = np.asarray(json.dumps({f"{m}_{initialization}_mae_initial_tokens": str(p) for m, p in tokens.items()}))
            np.savez_compressed(bag_path, **payload)
            dataset = load_bag_dataset(bag_path)
            targets = event_targets(dataset, rows)
            if dataset.modality_mask[:, :, 3].any():
                raise ValueError("audio must remain disabled")
            for seed in seeds:
                out_dir = args.out_root / "runs" / protocol / condition / f"seed_{seed}"
                print(f"starting protocol={protocol} condition={condition} seed={seed}", flush=True)
                try:
                    metrics = train(args, dataset, targets, directories, condition, protocol, seed, out_dir)
                except Exception as exc:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    failure = {"protocol": protocol, "condition": condition, "seed": seed,
                               "training_version": TRAINING_VERSION, "error": str(exc),
                               "traceback": traceback.format_exc()}
                    (out_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                    raise
                results.append({"protocol": protocol, "condition": condition, "seed": seed, "metrics": metrics})
        for seed in seeds:
            results.append({"protocol": protocol, "condition": "B0", "seed": seed,
                            "metrics": reference._read_reference(args.reference_root, protocol, seed)})
    summary = {}
    for protocol in protocols:
        summary[protocol] = {}
        for condition in ("B0",) + selected:
            matching = [r["metrics"] for r in results if r["protocol"] == protocol and r["condition"] == condition]
            summary[protocol][condition] = {}
            for label in LABEL_NAMES:
                values = np.asarray([r["test"]["per_label"][label]["raw_r"] for r in matching])
                spread = values.std(ddof=1) if len(values) > 1 else 0.0
                summary[protocol][condition][label] = f"{values.mean():.4f} ± {spread:.4f}"
    output = {"reference_route": reference.REFERENCE_ROUTE, "protocols": list(protocols), "seeds": list(seeds),
              "table_conditions": ["B0", *selected], "results": results, "summary": summary}
    (args.out_root / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    reference._table(output, args.out_root / "raw_r_tables.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
