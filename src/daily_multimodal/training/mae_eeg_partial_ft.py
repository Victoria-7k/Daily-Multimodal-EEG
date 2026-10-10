"""Round-2 B: matched EEG-only supervised tail fine-tuning."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.samplers import shuffled_batches
from daily_multimodal.daily_affect.training import _seed_everything, apply_modality_dropout, fit_token_normalization
from daily_multimodal.training.mae_emotion_adaptation import atomic_save, finite, rng_state, restore_rng, state_hash, window_loss
from daily_multimodal.training.multihead_regression import evaluate_event_level
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel

VARIANTS = {"B_T2_STD": (2, 1e-4), "B_T2_LOW": (2, 1e-5), "B_T1_LOW": (1, 1e-5)}
VERSION = "round2_B_eeg_only_matched_v1"


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class EEGPartialFTModel(nn.Module):
    def __init__(self, dataset, prefix_dir, x_mean, x_std, device):
        super().__init__()
        self.dev = torch.device(device)
        # Match the archived frozen fusion initialization before constructing tails.
        self.head = MultiEmotionStructureModel(DailyAffectRegressionConfig(
            model_id="window_replicated", hidden_dim=128, dropout=0.1,
            adapter_mode="per_modality", temporal_policy="uniform", lambda_d=0.25)).to(self.dev)
        self.head_initial_hash = state_hash(self.head.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        payload = torch.load(prefix_dir / "tail.pt", map_location="cpu", weights_only=False)
        self.blocks = payload["blocks"]
        layer = nn.TransformerEncoderLayer(256, payload["heads"], 1024, batch_first=True, activation="gelu")
        self.tail = nn.TransformerEncoder(layer, self.blocks, enable_nested_tensor=False).to(self.dev)
        self.tail.load_state_dict(payload["state_dict"], strict=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng:
            torch.cuda.set_rng_state_all(cuda_rng)
        self.post_head_rng_hash = array_hash(cpu_rng.numpy())
        self.post_head_cuda_rng_hash = [array_hash(s.cpu().numpy()) for s in cuda_rng]
        self.prefix_manifest = json.loads((prefix_dir / "manifest.json").read_text())
        if not (prefix_dir / "PREFIX_COMPLETE").is_file() or self.prefix_manifest["trainable_tail_blocks"] != self.blocks:
            raise ValueError("prefix cut / tail contract mismatch")
        with np.load(prefix_dir / "window_embeddings.npz", allow_pickle=False) as z:
            ids, valid = z["sample_id"].astype(str), z["valid_mask"].astype(bool)
        if not valid.all():
            raise ValueError("B requires the audited complete canonical EEG tokens")
        lookup = {sid: i for i, sid in enumerate(ids)}
        rows = np.asarray([lookup[sid] for sid in dataset.sample_id_matrix.astype(str).ravel()]).reshape(dataset.sample_id_matrix.shape)
        if not np.array_equal(valid[rows], dataset.modality_mask[:, :, 0]):
            raise ValueError("EEG availability changed")
        prefix = np.load(prefix_dir / "prefix.npy", mmap_mode="r")
        if list(prefix.shape) != self.prefix_manifest["prefix_shape"] or prefix.dtype != np.float32:
            raise ValueError("prefix shape/dtype changed")
        self.register_buffer("prefix", torch.from_numpy(np.array(prefix)).to(self.dev), persistent=False)
        self.register_buffer("rows", torch.as_tensor(rows, device=self.dev), persistent=False)
        self.register_buffer("fixed_tokens", torch.as_tensor(dataset.tokens, device=self.dev), persistent=False)
        self.register_buffer("native_mask", torch.as_tensor(dataset.modality_mask, device=self.dev), persistent=False)
        self.register_buffer("x_mean", torch.as_tensor(x_mean, device=self.dev))
        self.register_buffer("x_std", torch.as_tensor(x_std, device=self.dev))
        self.first_dropout_audit = None

    def train(self, mode=True):
        super().train(mode)
        self.tail.eval()
        return self

    def encode_rows(self, rows):
        chunks = []
        with sdpa_kernel([SDPBackend.MATH]):
            for start in range(0, len(rows), 128):
                chunks.append(self.tail(self.prefix[rows[start:start + 128]]).mean(dim=1))
        return torch.cat(chunks)

    def encode_events(self, indices):
        idx = torch.as_tensor(indices, device=self.dev)
        eeg = self.encode_rows(self.rows[idx].reshape(-1)).reshape(len(indices), 23, 256)
        tokens = torch.cat((eeg[:, :, None], self.fixed_tokens[idx, :, 1:]), dim=2)
        normalized = (tokens - self.x_mean) / self.x_std
        finite(normalized, "normalized fusion tokens")
        return normalized, self.native_mask[idx]

    def forward(self, indices, rng=None):
        tokens, mask = self.encode_events(indices)
        if rng is not None:
            tokens, mask = apply_modality_dropout(tokens, mask, probability=0.1, rng=rng)
            if self.first_dropout_audit is None:
                self.first_dropout_audit = {"batch_sha256": array_hash(np.asarray(indices)),
                    "mask_sha256": array_hash(mask.detach().cpu().numpy()),
                    "torch_rng_sha256": array_hash(torch.get_rng_state().numpy()),
                    "cuda_rng_sha256": [array_hash(s.cpu().numpy()) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
        with sdpa_kernel([SDPBackend.MATH]):
            return self.head(tokens, mask)


@torch.no_grad()
def predict(model, indices, mean, std):
    model.eval()
    chunks = []
    for start in range(0, len(indices), 64):
        p = model(indices[start:start + 64])["prediction"]
        finite(p, "event predictions")
        chunks.append(p.cpu().numpy())
    return (np.concatenate(chunks) * std + mean).astype(np.float32)


def macro(y, p, subjects, std):
    m = evaluate_event_level({"target": y, "prediction": p, "subject_id": subjects}, std)["per_label"]
    return {k: float(np.mean([m[l][k] for l in LABEL_NAMES])) for k in ("raw_r", "within_subject_centered_r", "standardized_rmse")}


@torch.no_grad()
def diagnostics(model, rows, initial_tokens, initial_tail):
    model.eval()
    live = model.encode_rows(rows).cpu().numpy().astype(np.float64)
    singular = np.linalg.svd(live - live.mean(axis=0), compute_uv=False)
    prob = singular / max(float(singular.sum()), 1e-12)
    rank = float(np.exp(-np.sum(prob[prob > 0] * np.log(prob[prob > 0]))))
    delta = sum(float((v.detach().cpu() - initial_tail[k]).double().square().sum()) for k, v in model.tail.state_dict().items())
    return {"tail_parameter_l2_delta": math.sqrt(delta), "val_token_relative_l2_change": float(np.linalg.norm(live - initial_tokens) / max(np.linalg.norm(initial_tokens), 1e-12)), "val_token_effective_rank": rank}


def train_ft(*, dataset, targets, prefix_dir, source_checkpoint, variant, seed, out_dir, smoke=False, device="cuda", interrupt_after_batches=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(True)
    _seed_everything(seed)
    blocks, lr = VARIANTS[variant]
    sp = dataset.split_indices()
    full_train = sp["train"]
    subset = lambda a: a[np.linspace(0, len(a) - 1, min(64, len(a)), dtype=int)]
    train, val = (subset(full_train), subset(sp["val"])) if smoke else (full_train, sp["val"])
    xm, xs = fit_token_normalization(dataset.tokens, dataset.modality_mask, full_train, scope="per_modality")
    ym = targets[full_train].mean(axis=0, keepdims=True).astype(np.float32)
    ys = targets[full_train].std(axis=0, keepdims=True).astype(np.float32)
    ys = np.where(ys >= 1e-6, ys, 1).astype(np.float32)
    model = EEGPartialFTModel(dataset, prefix_dir, xm, xs, device)
    if model.blocks != blocks:
        raise ValueError("variant has incorrect frozen cut")
    initial_tail = copy.deepcopy({k: v.detach().cpu() for k, v in model.tail.state_dict().items()})
    diagnostic_rows = model.rows[val].reshape(-1)[torch.as_tensor(np.linspace(0, len(val) * 23 - 1, 256, dtype=int), device=model.dev)]
    # Use the original token bag to define all fixed diagnostics and normalization.
    initial_tokens = dataset.tokens[val, :, 0].reshape(-1, 256)[np.linspace(0, len(val) * 23 - 1, 256, dtype=int)].astype(np.float64)
    model.eval()
    with torch.no_grad():
        raw = model.encode_rows(model.rows[full_train[:8]].reshape(-1)).cpu().numpy()
        error = float(np.max(np.abs(raw - dataset.tokens[full_train[:8], :, 0].reshape(-1, 256))))
        if error > 2e-4:
            raise ValueError(f"initial raw token equivalence failed: {error}")
    config = {"version": VERSION, "protocol": "cross_day", "variant": variant, "downstream_seed": seed,
        "ssl_source": "C", "ssl_seed": 240800, "source_checkpoint": str(source_checkpoint),
        "prefix_directory": str(prefix_dir), "prefix_manifest": model.prefix_manifest,
        "trainable_encoder_blocks": list(range(6 - blocks, 6)), "encoder_lr": lr, "head_lr": 1e-3,
        "epochs": 10 if smoke else 80, "patience": 10 if smoke else 15,
        "batch_size": 64, "window_chunk_size": 128, "weight_decay": 1e-4, "gradient_clip": 1.0,
        "head_initial_sha256": model.head_initial_hash, "post_head_cpu_rng_sha256": model.post_head_rng_hash,
        "post_head_cuda_rng_sha256": model.post_head_cuda_rng_hash,
        "initial_tail_sha256": state_hash(initial_tail), "initial_token_max_abs_error": error,
        "train_events": train.tolist(), "val_events": val.tolist(), "smoke": smoke,
        "label_names": list(LABEL_NAMES), "normalization_fit": "original_C_full_train_tokens_fixed",
        "train_target_mean": ym.ravel().tolist(), "train_target_std": ys.ravel().tolist(),
        "selection_metric": "val_macro_standardized_rmse_min", "supervision_boundary": "C_SSL_init__EEG_tail_and_11label_fusion_supervised",
        "rng_rule": "archived_shared_numpy_shuffle_modality_rng__independent_of_encoder_construction",
        "attention": "train_math_fp32_eval_existing_fastpath", "encoder_dropout": "eval_weights_trainable",
        "fixed_non_eeg_sha256": array_hash(dataset.tokens[:, :, 1:]), "native_mask_sha256": array_hash(dataset.modality_mask),
        "feature_mean_sha256": array_hash(xm), "feature_std_sha256": array_hash(xs)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": 1e-3}, {"params": model.tail.parameters(), "lr": lr}], weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    recovery = out_dir / "recovery.pt"
    history, audit, best_state = [], {}, None
    best, best_epoch, stale, epoch, next_batch = math.inf, 0, 0, 0, 0
    batches, losses = None, []
    if recovery.exists():
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("B recovery contract changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, audit, best_state = saved["history"], saved["audit"], saved["best_state"]
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        epoch, next_batch, batches, losses = saved["epoch"], saved["next_batch"], saved["batches"], saved["losses"]
        model.first_dropout_audit = saved["first_dropout_audit"]
        restore_rng(saved["rng"], rng)
    started, executed = time.time(), 0
    def save():
        atomic_save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "audit": audit, "best_state": best_state, "best": best, "best_epoch": best_epoch,
            "stale": stale, "epoch": epoch, "next_batch": next_batch, "batches": batches, "losses": losses,
            "rng": rng_state(rng), "first_dropout_audit": model.first_dropout_audit}, recovery)
    while epoch < config["epochs"] and stale < config["patience"]:
        if batches is None:
            batches = list(shuffled_batches(train, 64, rng))
            next_batch, losses = 0, []
        model.train()
        for b in range(next_batch, len(batches)):
            batch = batches[b]
            target = torch.as_tensor((targets[batch] - ym) / ys, device=model.dev)
            output = model(batch, rng=rng)
            loss = window_loss(output["window_prediction"], target, output["window_mask"])
            finite(loss, "window loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    finite(p.grad, "gradient " + name)
            first = not audit
            if first:
                before = {k: v.detach().clone() for k, v in model.tail.named_parameters()}
                grad = sum(float(p.grad.detach().abs().sum()) for p in model.tail.parameters() if p.grad is not None)
                if not math.isfinite(grad) or grad <= 0:
                    raise RuntimeError("EEG tail lacks finite nonzero gradients")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for name, p in model.named_parameters():
                finite(p, "parameter " + name)
            if first:
                delta = max(float((p.detach() - before[k]).abs().max()) for k, p in model.tail.named_parameters())
                if delta <= 0:
                    raise RuntimeError("EEG tail did not update")
                audit.update(first_tail_gradient_abs_sum=grad, first_tail_parameter_max_abs_delta=delta)
            losses.append(float(loss.detach()))
            next_batch = b + 1
            save()
            executed += 1
            if interrupt_after_batches is not None and executed >= interrupt_after_batches:
                raise InterruptedError("test interruption after saved B batch")
        vp = predict(model, val, ym, ys)
        vm = macro(targets[val], vp, dataset.subject_id[val], ys)
        tp = predict(model, train, ym, ys)
        tm = macro(targets[train], tp, dataset.subject_id[train], ys)
        score = vm["standardized_rmse"]
        if not math.isfinite(score):
            raise RuntimeError("nonfinite val selector")
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        epoch += 1
        history.append({"epoch": epoch, "train_window_mse": float(np.mean(losses)),
            "train_event_standardized_rmse": tm["standardized_rmse"],
            "val_macro_standardized_rmse": score, "val_raw_r": vm["raw_r"],
            "val_centered_r": vm["within_subject_centered_r"],
            **diagnostics(model, diagnostic_rows, initial_tokens, initial_tail)})
        batches, next_batch, losses = None, 0, []
        save()
        (out_dir / "progress.json").write_text(json.dumps({"history": history, "best_epoch": best_epoch}, indent=2))
        print(json.dumps({"variant": variant, "seed": seed, **history[-1]}), flush=True)
    if best_state is None:
        raise RuntimeError("no selected B model")
    model.load_state_dict(best_state)
    if state_hash(model.tail.state_dict()) == state_hash(initial_tail):
        raise ValueError("selected tail unchanged")
    audit.update(first_dropout_audit=model.first_dropout_audit,
        fixed_non_eeg_unchanged=array_hash(model.fixed_tokens[:, :, 1:].cpu().numpy()) == config["fixed_non_eeg_sha256"],
        native_mask_unchanged=array_hash(model.native_mask.cpu().numpy()) == config["native_mask_sha256"],
        normalization_unchanged=array_hash(model.x_mean.cpu().numpy()) == config["feature_mean_sha256"] and array_hash(model.x_std.cpu().numpy()) == config["feature_std_sha256"],
        prefix_requires_grad=model.prefix.requires_grad, raw_pool_in_training=False,
        selected_tail_sha256=state_hash(model.tail.state_dict()),
        selected_diagnostics=diagnostics(model, diagnostic_rows, initial_tokens, initial_tail))
    if not all(audit[k] for k in ("fixed_non_eeg_unchanged", "native_mask_unchanged", "normalization_unchanged")):
        raise ValueError("fixed B inputs changed")
    arrays, metrics = {"label_names": np.asarray(LABEL_NAMES)}, {}
    for leaf in (("val",) if smoke else ("val", "test")):
        idx = val if leaf == "val" else sp[leaf]
        p = predict(model, idx, ym, ys)
        metrics[leaf] = evaluate_event_level({"target": targets[idx], "prediction": p, "subject_id": dataset.subject_id[idx]}, ys)
        for k, v in {"event_id": dataset.event_id[idx], "subject_id": dataset.subject_id[idx], "day_id": dataset.day_id[idx], "target": targets[idx], "prediction": p}.items():
            arrays[leaf + "_" + k] = v
    np.savez_compressed(out_dir / "event_predictions.npz", **arrays)
    original = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    encoder = {k: v for k, v in original["state_dict"].items() if k.startswith(("patch_embed.", "encoder.")) or k == "position"}
    for k, v in model.tail.state_dict().items():
        parts = k.split(".")
        parts[1] = str(int(parts[1]) + 6 - blocks)
        encoder["encoder." + ".".join(parts)] = v.detach().cpu()
    atomic_save({"state_dict": best_state, "encoder_state_dict": encoder, "raw_normalization": original["manifest"]["raw_normalization"],
        "config": config, "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
        "bag_path": str(dataset.bag_path)}, out_dir / "best_checkpoint.pt")
    result = {"status": "ok", **config, "best_epoch": best_epoch, "epochs_ran": len(history), "best_val_macro_standardized_rmse": best,
        "history": history, "audit": audit, "duration_seconds": time.time() - started, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "trainability_audit.json").write_text(json.dumps(audit, indent=2))
    (out_dir / "FT_COMPLETE").write_text("complete\n")
    return result
