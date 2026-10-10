"""Round-2 C: fixed-token event loss with explicit window consistency."""
from __future__ import annotations
import copy
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.regression import DailyAffectRegressionConfig
from daily_multimodal.daily_affect.samplers import shuffled_batches
from daily_multimodal.daily_affect.training import _seed_everything, apply_modality_dropout, difficulty_lambda_for_epoch, fit_token_normalization
from daily_multimodal.training.mae_emotion_adaptation import atomic_save, finite, rng_state, restore_rng, state_hash
from daily_multimodal.training.mae_eeg_partial_ft import array_hash
from daily_multimodal.training.multihead_regression import evaluate_event_level
from daily_multimodal.training.structure_emotion import MultiEmotionStructureModel, _batch, _predict

VARIANTS = {"C_EVENT": ("C", 0.0), "C_EVENT_WEAK": ("C", 0.1), "B0_EVENT": ("B0", 0.0), "B0_EVENT_WEAK": ("B0", 0.1)}
VERSION = "round2_C_fixed_tokens_event_consistency_v1"

def event_consistency_loss(output, target, weight):
    if weight not in (0., 0.1, 1.):
        raise ValueError("only predeclared consistency weights are supported")
    valid = output["window_mask"].to(target.dtype)
    if not bool((valid.sum(dim=1) > 0).all()):
        raise ValueError("event without valid windows")
    if weight == 1.:
        # Preserve the archived operation order exactly for the reusable control.
        error = (output["window_prediction"] - target[:, None, :]).square()
        return ((error * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1).clamp_min(1)[:, None]).mean()
    event = (output["prediction"] - target).square().mean()
    if weight == 0.:
        return event
    deviation = (output["window_prediction"] - output["prediction"][:, None, :]).square()
    consistency = ((deviation * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1)[:, None]).mean()
    return event + weight * consistency

@torch.no_grad()
def window_diagnostics(model, dataset, indices, xm, xs, ym, ys, device):
    model.eval()
    chunks = {k: [] for k in ("window_variance", "window_min", "window_max", "valid_count", "event_prediction")}
    for start in range(0, len(indices), 128):
        batch = indices[start:start + 128]
        x, mask, _ = _batch(dataset, np.zeros((dataset.row_count, 11), np.float32), batch, xm, xs, ym, ys, device)
        output = model(x, mask)
        valid = output["window_mask"].bool()
        pred = output["prediction"]
        wp = output["window_prediction"]
        variance = ((wp - pred[:, None]).square() * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1)[:, None]
        minimum = wp.masked_fill(~valid[:, :, None], float("inf")).amin(dim=1)
        maximum = wp.masked_fill(~valid[:, :, None], float("-inf")).amax(dim=1)
        for k, v in {"window_variance": variance.cpu().numpy() * ys ** 2,
                     "window_min": minimum.cpu().numpy() * ys + ym,
                     "window_max": maximum.cpu().numpy() * ys + ym,
                     "valid_count": valid.sum(dim=1).cpu().numpy(),
                     "event_prediction": pred.cpu().numpy() * ys + ym}.items():
            if not np.isfinite(v).all():
                raise ValueError("nonfinite window diagnostics")
            chunks[k].append(v)
    return {k: np.concatenate(v) for k, v in chunks.items()}

def train_event_loss(*, dataset, targets, variant, seed, out_dir, smoke=False, device="cuda", weight_override=None, interrupt_after_batches=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(True)
    _seed_everything(seed)
    source, weight = VARIANTS[variant]
    if weight_override is not None:
        if weight_override != 1.:
            raise ValueError("override is reserved for exact control replay")
        weight = weight_override
    sp = dataset.split_indices()
    train, val = sp["train"], sp["val"]
    if not dataset.modality_mask.any(axis=2).any(axis=1).all():
        raise ValueError("event without any available window")
    xm, xs = fit_token_normalization(dataset.tokens, dataset.modality_mask, train, scope="per_modality")
    ym = targets[train].mean(axis=0, keepdims=True).astype(np.float32)
    ys = targets[train].std(axis=0, keepdims=True).astype(np.float32)
    ys = np.where(np.isfinite(ys) & (ys >= 1e-6), ys, 1.).astype(np.float32)
    dev = torch.device(device)
    model = MultiEmotionStructureModel(DailyAffectRegressionConfig(model_id="window_replicated", hidden_dim=128,
        dropout=.1, adapter_mode="per_modality", temporal_policy="uniform", lambda_d=.25)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    initial_hash = state_hash(model.state_dict())
    config = {"version": VERSION, "variant": variant, "source": source, "consistency_weight": weight,
        "protocol": "cross_day", "downstream_seed": seed, "embedding_seed": 240800,
        "head_initial_sha256": initial_hash, "head_cpu_rng_sha256": array_hash(torch.get_rng_state().numpy()),
        "head_cuda_rng_sha256": [array_hash(v.cpu().numpy()) for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "tokens_sha256": array_hash(dataset.tokens), "mask_sha256": array_hash(dataset.modality_mask),
        "batch_size": 64, "epochs": 3 if smoke else 80, "patience": 3 if smoke else 15,
        "learning_rate": 1e-3, "weight_decay": 1e-4, "dropout": .1, "modality_dropout": .1,
        "gradient_clip": 1., "model_id": "window_replicated", "head_variant": "H1_shared2_11xhead2",
        "normalization": "per_modality", "adapter_mode": "per_modality", "temporal_policy": "uniform",
        "supervision_boundary": dataset.supervision_boundary, "encoder_training_scope": "all_frozen",
        "bag_path": str(dataset.bag_path), "train_target_mean": ym.ravel().tolist(), "train_target_std": ys.ravel().tolist(),
        "selection_metric": "val_macro_standardized_rmse_min", "smoke": smoke,
        "label_names": list(LABEL_NAMES), "train_events": train.tolist(), "val_events": val.tolist()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    recovery = out_dir / "recovery.pt"
    history, audit, best_state = [], {}, None
    best, best_epoch, stale, epoch, next_batch = math.inf, 0, 0, 0, 0
    batches, losses = None, []
    if recovery.exists():
        saved = torch.load(recovery, map_location="cpu", weights_only=False)
        if config != saved["config"]:
            raise ValueError("C recovery contract changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history, audit, best_state = saved["history"], saved["audit"], saved["best_state"]
        best, best_epoch, stale = saved["best"], saved["best_epoch"], saved["stale"]
        epoch, next_batch, batches, losses = saved["epoch"], saved["next_batch"], saved["batches"], saved["losses"]
        restore_rng(saved["rng"], rng)
    started, executed = time.time(), 0
    def save():
        atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
            "history": history, "audit": audit, "best_state": best_state, "best": best, "best_epoch": best_epoch,
            "stale": stale, "epoch": epoch, "next_batch": next_batch, "batches": batches, "losses": losses, "rng": rng_state(rng)}, recovery)
    while epoch < config["epochs"] and stale < config["patience"]:
        if batches is None:
            batches = shuffled_batches(train, 64, rng)
            next_batch, losses = 0, []
        model.train()
        scheduled_lambda = difficulty_lambda_for_epoch(epoch, target=.25, warmup_epochs=5, ramp_epochs=5)
        for b in range(next_batch, len(batches)):
            batch = batches[b]
            x, mask, target = _batch(dataset, targets, batch, xm, xs, ym, ys, dev)
            x, mask = apply_modality_dropout(x, mask, probability=.1, rng=rng)
            if not audit:
                first = {"batch_sha256": array_hash(batch), "mask_sha256": array_hash(mask.cpu().numpy()),
                    "cpu_rng_sha256": array_hash(torch.get_rng_state().numpy()),
                    "cuda_rng_sha256": [array_hash(v.cpu().numpy()) for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []}
            output = model(x, mask, lambda_d_override=scheduled_lambda)
            loss = event_consistency_loss(output, target, weight)
            finite(loss, "C loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    finite(p.grad, "C gradient " + n)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for n, p in model.named_parameters():
                finite(p, "C parameter " + n)
            if not audit:
                audit["first_batch"] = first
            losses.append(float(loss.detach().cpu()))
            next_batch = b + 1
            save()
            executed += 1
            if interrupt_after_batches is not None and executed >= interrupt_after_batches:
                raise InterruptedError("test interruption after saved C batch")
        vp = _predict(model, dataset, val, xm, xs, ym, ys, dev)
        score = float(np.sqrt(np.mean(np.square((vp - targets[val]) / ys), axis=0)).mean())
        if not math.isfinite(score):
            raise ValueError("nonfinite C val selector")
        if score < best:
            best, best_epoch, stale = score, epoch + 1, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        epoch += 1
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_standardized_rmse": score})
        batches, next_batch, losses = None, 0, []
        save()
        (out_dir / "progress.json").write_text(json.dumps({"history": history, "best_epoch": best_epoch}, indent=2))
        print(json.dumps({"variant": variant, "seed": seed, **history[-1]}), flush=True)
    if best_state is None:
        raise ValueError("no selected C checkpoint")
    model.load_state_dict(best_state)
    audit.update(tokens_unchanged=array_hash(dataset.tokens) == config["tokens_sha256"],
        mask_unchanged=array_hash(dataset.modality_mask) == config["mask_sha256"])
    if not audit["tokens_unchanged"] or not audit["mask_unchanged"]:
        raise ValueError("C fixed inputs changed")
    metrics, arrays, diagnostic_arrays = {}, {"label_names": np.asarray(LABEL_NAMES)}, {"label_names": np.asarray(LABEL_NAMES)}
    for leaf in (("val",) if smoke else ("val", "test")):
        idx = sp[leaf]
        pred = _predict(model, dataset, idx, xm, xs, ym, ys, dev)
        finite(torch.as_tensor(pred), "C final predictions")
        metrics[leaf] = evaluate_event_level({"target": targets[idx], "prediction": pred, "subject_id": dataset.subject_id[idx]}, ys)
        for k, v in {"event_id": dataset.event_id[idx], "subject_id": dataset.subject_id[idx], "day_id": dataset.day_id[idx], "target": targets[idx], "prediction": pred}.items():
            arrays[leaf + "_" + k] = v
        diag = window_diagnostics(model, dataset, idx, xm, xs, ym, ys, dev)
        if np.max(np.abs(diag["event_prediction"] - pred)) > 1e-6:
            raise ValueError("C diagnostics / predictions disagree")
        for k, v in diag.items():
            diagnostic_arrays[leaf + "_" + k] = v
    np.savez_compressed(out_dir / "event_predictions.npz", **arrays)
    np.savez_compressed(out_dir / "window_diagnostics.npz", **diagnostic_arrays)
    atomic_save({"state_dict": best_state, "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys,
        "config": config, "bag_path": str(dataset.bag_path), "downstream_seed": seed,
        "modality_token_sources": json.loads(dataset.source_npz_json)}, out_dir / "best_checkpoint.pt")
    result = {"status": "ok", **config, "best_epoch": best_epoch, "best_val_macro_standardized_rmse": best,
        "epochs_ran": len(history), "duration_seconds": time.time() - started, "audit": audit, "history": history, **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "matching_audit.json").write_text(json.dumps(audit, indent=2))
    (out_dir / "EVENT_LOSS_COMPLETE").write_text("complete\n")
    return result
