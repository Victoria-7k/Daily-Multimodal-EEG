"""EEG-only event-supervised MAE readout/adaptation for round-2 experiment A.

Cached, immutable activations end before the final two encoder blocks. The
decoder is excluded. This module does not change the archived fusion trainer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.samplers import shuffled_batches
from daily_multimodal.daily_affect.training import _seed_everything, fit_token_normalization
from daily_multimodal.training.modality_mae import embedding_health, require_healthy
from daily_multimodal.training.multihead_regression import evaluate_event_level
from daily_multimodal.training.structure_emotion import SharedTwoLayerElevenHeads


VERSION = "eeg_mae_single_modality_emotion_v1"


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def window_loss(prediction, target, valid):
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("EEG-only event has no valid window")
    error = (prediction - target[:, None, :]).square()
    return ((error * valid[:, :, None]).sum(dim=1) / counts[:, None]).mean()


def rng_state(shuffle):
    return {"shuffle": copy.deepcopy(shuffle.bit_generator.state),
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved, shuffle):
    shuffle.bit_generator.state = saved["shuffle"]
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def finite(value, context):
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError("non-finite " + context)


class EEGEmotionModel(nn.Module):
    def __init__(self, prefix, tail_payload, original_tokens, event_window_rows,
                 mean, std, *, adapt, device, chunk_size=128):
        super().__init__()
        # Head creation and dropout streams match across PROBE/ADAPT and C/P.
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(),
                                  SharedTwoLayerElevenHeads(128, 0.1)).to(device)
        self.head_initial_sha256 = state_hash(self.head.state_dict())
        # Tail construction must not consume the task head's random stream.
        saved = torch.get_rng_state()
        cuda_saved = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        layer = nn.TransformerEncoderLayer(256, tail_payload["heads"],
                                           dim_feedforward=1024, dropout=0.1,
                                           batch_first=True, activation="gelu")
        self.tail = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).to(device)
        self.tail.load_state_dict(tail_payload["state_dict"], strict=True)
        torch.set_rng_state(saved)
        if cuda_saved:
            torch.cuda.set_rng_state_all(cuda_saved)
        self.tail.requires_grad_(adapt)
        self.tail.eval()
        self.prefix = prefix
        self.original_tokens = original_tokens
        self.rows = event_window_rows
        self.adapt, self.dev, self.chunk_size = adapt, device, chunk_size
        self.register_buffer("feature_mean", torch.as_tensor(mean, device=device))
        self.register_buffer("feature_std", torch.as_tensor(std, device=device))

    def train(self, mode=True):
        super().train(mode)
        self.tail.eval()
        return self

    def encode_rows(self, rows):
        if not self.adapt:
            return torch.as_tensor(self.original_tokens[rows], device=self.dev)
        chunks = []
        for start in range(0, len(rows), self.chunk_size):
            x = torch.as_tensor(np.array(self.prefix[rows[start:start + self.chunk_size]], copy=True), device=self.dev)
            with sdpa_kernel([SDPBackend.MATH]):
                chunks.append(self.tail(x).mean(dim=1))
        tokens = torch.cat(chunks)
        finite(tokens, "EEG tokens")
        return tokens

    def forward(self, indices):
        rows = self.rows[indices]
        tokens = self.encode_rows(rows.ravel()).reshape(len(indices), 23, 256)
        normalized = (tokens - self.feature_mean) / self.feature_std
        finite(normalized, "fixed-normalized EEG tokens")
        return self.head(normalized)


@torch.no_grad()
def predict(model, indices, mean, std, batch_size=64):
    model.eval()
    result = []
    for start in range(0, len(indices), batch_size):
        p = model(indices[start:start + batch_size]).mean(dim=1)
        finite(p, "event prediction")
        result.append(p.cpu().numpy())
    return (np.concatenate(result) * std + mean).astype(np.float32)


def train_adaptation(*, dataset, targets, prefix_dir, source_checkpoint,
                     source_token, source_id, mode, out_dir, smoke=False,
                     device="cuda", interrupt_after_batches=None):
    """Exact batch-boundary recovery; smoke uses fixed complete event subsets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    # Preserve the archived eval/no-grad inference kernel for token equivalence.
    # Trainable-tail autograd disables that inference fastpath; sdpa_kernel in
    # encode_rows then explicitly selects FP32 math attention for adaptation.
    torch.backends.mha.set_fastpath_enabled(True)
    _seed_everything(240800)
    dev = torch.device(device)
    split = dataset.split_indices()
    full_train = split["train"]
    train = full_train[np.linspace(0, len(full_train) - 1, min(64, len(full_train)), dtype=int)] if smoke else full_train
    val = split["val"][np.linspace(0, len(split["val"]) - 1, min(64, len(split["val"])), dtype=int)] if smoke else split["val"]
    y_mean = targets[full_train].mean(axis=0, keepdims=True).astype(np.float32)
    y_std = targets[full_train].std(axis=0, keepdims=True).astype(np.float32)
    y_std = np.where(y_std >= 1e-6, y_std, 1).astype(np.float32)
    xm, xs = fit_token_normalization(dataset.tokens, dataset.modality_mask, full_train, scope="per_modality")
    mean, std = xm[0, 0, 0], xs[0, 0, 0]
    with np.load(source_token, allow_pickle=False) as z:
        ids, embeddings = z["sample_id"].astype(str), z["embedding"].astype(np.float32)
    lookup = {sid: i for i, sid in enumerate(ids)}
    rows = np.asarray([lookup[s] for s in dataset.sample_id_matrix.astype(str).ravel()]).reshape(1253, 23)
    prefix = np.load(prefix_dir / "prefix.npy", mmap_mode="r")
    pm = json.loads((prefix_dir / "manifest.json").read_text())
    if prefix.shape != (28819, 10, 256) or pm["valid_count"] != 28819 or pm["trainable_tail_blocks"] != 2:
        raise ValueError("incomplete or incompatible EEG prefix")
    tail = torch.load(prefix_dir / "tail.pt", map_location="cpu", weights_only=False)
    model = EEGEmotionModel(prefix, tail, embeddings, rows, mean, std,
                            adapt=mode == "ADAPT", device=dev)
    initial_tail = state_hash(model.tail.state_dict())
    frozen_prefix_hash = pm["prefix_sha256"]
    with torch.no_grad():
        # Test the live split even for PROBE, which trains on original frozen tokens.
        check = rows[full_train[:8]].ravel()
        previous = model.adapt
        model.adapt = True
        error = float((model.encode_rows(check) - torch.as_tensor(embeddings[check], device=dev)).abs().max())
        model.adapt = previous
        if error > 2e-4:
            raise ValueError(f"initial tail token equivalence failed: {error}")
    groups = [{"params": model.head.parameters(), "lr": 1e-3}]
    if mode == "ADAPT":
        groups.append({"params": model.tail.parameters(), "lr": 1e-5})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    epochs, patience = (10, 10) if smoke else (80, 15)
    config = {"version": VERSION, "ssl_source": source_id, "mode": mode,
              "protocol": "cross_day", "adaptation_seed": 240800, "ssl_seed": 240800,
              "supervision_boundary": "11label_supervised_eeg_adaptation" if mode == "ADAPT" else "frozen_encoder_nonlinear_11label_readout",
              "source_checkpoint": str(source_checkpoint), "source_token": str(source_token),
              "prefix_manifest": pm, "train_events": train.tolist(), "val_events": val.tolist(),
              "label_names": list(LABEL_NAMES), "epochs": epochs, "patience": patience,
              "batch_size": 64, "encoder_lr": 1e-5 if mode == "ADAPT" else 0,
              "head_lr": 1e-3, "weight_decay": 1e-4, "gradient_clip": 1.0,
              "trainable_encoder_blocks": [4, 5] if mode == "ADAPT" else [],
              "encoder_dropout": "eval", "attention": "train_math_fp32_eval_existing_fastpath", "smoke": smoke,
              "normalization_fit": "full_canonical_train_events_only_fixed",
              "feature_mean": mean.tolist(), "feature_std": std.tolist(),
              "train_target_mean": y_mean.ravel().tolist(), "train_target_std": y_std.ravel().tolist(),
              "head_initial_sha256": model.head_initial_sha256,
              "initial_tail_sha256": initial_tail, "initial_token_max_abs_error": error,
              "selection_metric": "val_macro_standardized_rmse_min"}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(240800)
    recovery = out_dir / "recovery.pt"
    history, audit, best_state = [], {}, None
    best, best_epoch, stale, epoch, next_batch = math.inf, 0, 0, 0, 0
    batches, losses = None, []
    if recovery.exists():
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("adaptation recovery contract changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, audit, best_state = saved["history"], saved["audit"], saved["best_state"]
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        epoch, next_batch, batches, losses = saved["epoch"], saved["next_batch"], saved["batches"], saved["losses"]
        restore_rng(saved["rng"], rng)
    started, executed = time.time(), 0
    def save():
        atomic_save({"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "history": history, "audit": audit, "best_state": best_state,
                     "best": best, "best_epoch": best_epoch, "stale": stale,
                     "epoch": epoch, "next_batch": next_batch, "batches": batches,
                     "losses": losses, "rng": rng_state(rng)}, recovery)
    while epoch < epochs and stale < patience:
        if batches is None:
            batches = list(shuffled_batches(train, 64, rng))
            next_batch, losses = 0, []
        model.train()
        for batch_number in range(next_batch, len(batches)):
            batch = batches[batch_number]
            target = torch.as_tensor((targets[batch] - y_mean) / y_std, device=dev)
            output = model(batch)
            valid = torch.ones(output.shape[:2], dtype=output.dtype, device=dev)
            loss = window_loss(output, target, valid)
            finite(loss, "loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    finite(p.grad, "gradient " + name)
            first = not audit
            if first:
                before = {k: v.detach().clone() for k, v in model.tail.named_parameters()}
                grad = sum(float(p.grad.abs().sum()) for p in model.tail.parameters() if p.grad is not None)
                if mode == "ADAPT" and (not math.isfinite(grad) or grad <= 0):
                    raise RuntimeError("ADAPT tail lacks nonzero finite gradient")
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            for name, p in model.named_parameters():
                finite(p, "parameter " + name)
            if first:
                delta = max(float((p - before[name]).abs().max()) for name, p in model.tail.named_parameters())
                if (mode == "ADAPT" and delta <= 0) or (mode == "PROBE" and delta != 0):
                    raise RuntimeError("encoder trainability audit failed")
                audit.update(first_tail_gradient_abs_sum=grad, first_tail_parameter_delta=delta,
                             frozen_prefix_requires_grad=False, frozen_prefix_sha256=frozen_prefix_hash)
            losses.append(float(loss.detach()))
            next_batch = batch_number + 1
            save()
            executed += 1
            if interrupt_after_batches is not None and executed >= interrupt_after_batches:
                raise InterruptedError("test interruption after saved batch")
        pred = predict(model, val, y_mean, y_std)
        score = float(np.sqrt(np.mean(((pred - targets[val]) / y_std) ** 2, axis=0)).mean())
        if not math.isfinite(score):
            raise RuntimeError("non-finite validation selector")
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        epoch += 1
        history.append({"epoch": epoch, "train_window_mse": float(np.mean(losses)),
                        "val_macro_standardized_rmse": score})
        batches, next_batch, losses = None, 0, []
        save()
        (out_dir / "progress.json").write_text(json.dumps({"history": history, "best_epoch": best_epoch}, indent=2))
        print(json.dumps({"source": source_id, "mode": mode, **history[-1]}), flush=True)
    if best_state is None:
        raise RuntimeError("no selected model")
    model.load_state_dict(best_state)
    final_tail = state_hash(model.tail.state_dict())
    if (mode == "PROBE") != (final_tail == initial_tail):
        raise ValueError("selected encoder trainability failed")
    with torch.no_grad():
        health_rows = rows[val].ravel()
        health_tokens = model.encode_rows(health_rows[np.linspace(0, len(health_rows) - 1, 256, dtype=int)]).cpu().numpy()
    health = embedding_health(health_tokens, robust=True)
    require_healthy(health, "selected EEG emotion representation")
    audit.update(initial_tail_sha256=initial_tail, selected_tail_sha256=final_tail,
                 probe_encoder_unchanged=final_tail == initial_tail,
                 embedding_health=health, prefix_read_only=True,
                 raw_pool_in_supervised_training=False)
    (out_dir / "trainability_audit.json").write_text(json.dumps(audit, indent=2))
    arrays, metrics = {"label_names": np.asarray(LABEL_NAMES)}, {}
    # Smoke never evaluates the test leaf. Formal saved test predictions are not
    # read by selection tools; report publication follows the frozen val record.
    for leaf in (("val",) if smoke else ("val", "test")):
        idx = val if leaf == "val" else split[leaf]
        p = predict(model, idx, y_mean, y_std)
        metrics[leaf] = evaluate_event_level({"target": targets[idx], "prediction": p,
                                              "subject_id": dataset.subject_id[idx]}, y_std)
        for key, value in {"event_id": dataset.event_id[idx], "subject_id": dataset.subject_id[idx],
                           "day_id": dataset.day_id[idx], "target": targets[idx], "prediction": p}.items():
            arrays[f"{leaf}_{key}"] = value
    np.savez_compressed(out_dir / "event_predictions.npz", **arrays)
    # Include the immutable full initialization and learned tail for standalone
    # raw-signal reconstruction, without the MAE reconstruction decoder.
    original = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    encoder = {k: v for k, v in original["state_dict"].items()
               if k.startswith(("patch_embed.", "encoder.")) or k == "position"}
    if mode == "ADAPT":
        for k, v in model.tail.cpu().state_dict().items():
            parts = k.split(".")
            parts[1] = str(int(parts[1]) + 4)
            encoder["encoder." + ".".join(parts)] = v
        model.tail.to(dev)
    atomic_save({"state_dict": best_state, "encoder_state_dict": encoder,
                 "raw_normalization": original["manifest"]["raw_normalization"],
                 "config": config, "y_mean": y_mean, "y_std": y_std}, out_dir / "best_checkpoint.pt")
    if mode == "ADAPT" and not smoke:
        embedding = np.empty((28819, 256), np.float32)
        model.eval()
        with torch.no_grad():
            for start in range(0, 28819, 128):
                embedding[start:start + 128] = model.encode_rows(np.arange(start, min(start + 128, 28819))).cpu().numpy()
        if not np.isfinite(embedding).all():
            raise ValueError("non-finite adapted canonical token")
        np.savez_compressed(out_dir / "window_embeddings.npz", sample_id=ids,
                            embedding=embedding, valid_mask=np.ones(28819, bool),
                            ssl_source=np.asarray(source_id), adaptation_seed=np.asarray(240800),
                            supervision_boundary=np.asarray("11label_supervised_eeg_adaptation"),
                            protocol=np.asarray("cross_day"))
    result = {"status": "ok", **config, "best_epoch": best_epoch,
              "best_val_macro_standardized_rmse": best, "epochs_ran": len(history),
              "history": history, "audit": audit, "duration_seconds": time.time() - started, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "ADAPTATION_COMPLETE").write_text("complete\n")
    return result
