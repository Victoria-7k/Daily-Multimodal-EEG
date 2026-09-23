#!/usr/bin/env python3
"""Run a small W3FM ACC replacement screen with handcrafted ACC features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered
from daily_multimodal.split_paths import resolve_protocol_split_root


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_WEAR_FM_ROOT = DEFAULT_ROOT / "outputs/wear_fm"
DEFAULT_PHASE2_ROOT = DEFAULT_WEAR_FM_ROOT / "phase2"
DEFAULT_INTERNAL_ROOT = DEFAULT_WEAR_FM_ROOT / "internal_ablation"
DEFAULT_OUT_ROOT = DEFAULT_WEAR_FM_ROOT / "acc_replacement_screen"
DEFAULT_PROTOCOLS = ("cross_day", "date_in_order")
DEFAULT_SEEDS = (240729, 240730, 240731)
DEFAULT_ROUTES = ("W3FM_acc_handcrafted",)
LABEL_NAMES = (
    "inspired",
    "alert",
    "determined",
    "attentive",
    "active",
    "hostile",
    "nervous",
    "upset",
    "afraid",
    "ashamed",
    "fatigue",
    "stress",
    "happy",
)
ROUTE_COMPONENTS = {
    "W3FM_acc_handcrafted": ("ppg", "acc_handcrafted", "gsr"),
}
REFERENCE_ROUTES = ("W3FM_frozen", "W3FM_no_acc")


@dataclass(frozen=True)
class Dataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    event_id: np.ndarray
    target: np.ndarray
    complete_mask: np.ndarray
    ppg: np.ndarray
    gsr: np.ndarray
    acc_handcrafted: np.ndarray
    acc_feature_names: tuple[str, ...]


class WearHead(torch.nn.Module):
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1)


class W3FMReplacementModel(torch.nn.Module):
    def __init__(self, components: tuple[str, ...], input_dims: dict[str, int], *, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.components = tuple(components)
        self.projections = torch.nn.ModuleDict({name: _projector(input_dims[name], dropout) for name in self.components})
        self.gate = torch.nn.Linear(256, 1)
        self.head = WearHead(input_dim=256, hidden_dim=hidden_dim, dropout=dropout)

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = [self.projections[name](inputs[name]) for name in self.components]
        stacked = torch.stack(tokens, dim=1)
        weights = torch.softmax(self.gate(stacked).squeeze(-1), dim=1)
        return (weights.unsqueeze(-1) * stacked).sum(dim=1)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.encode(inputs))


def _projector(input_dim: int, dropout: float) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, 256),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--wear-fm-root", type=Path, default=DEFAULT_WEAR_FM_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL_ROOT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--selection-metric", choices=("rmse", "raw_r", "centered_r"), default="rmse")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")

    dataset = _load_dataset(args)
    protocols = _split_csv(args.protocols)
    routes = _split_csv(args.routes)
    seeds = [int(value) for value in _split_csv(args.seeds)]
    _validate_routes(routes)

    args.out_root.mkdir(parents=True, exist_ok=True)
    _write_preflight(dataset, args.out_root / "acc_replacement_preflight.json")
    _write_feature_manifest(dataset, args.out_root)

    results: list[dict[str, Any]] = []
    started = time.time()
    run_number = 0
    for protocol in protocols:
        split = _filter_split(
            _load_split(resolve_protocol_split_root(args.splits_root, protocol), len(dataset.sample_id)),
            dataset.complete_mask,
        )
        for seed in seeds:
            for route in routes:
                run_number += 1
                if args.limit_runs and run_number > args.limit_runs:
                    break
                run_dir = args.out_root / "runs" / protocol / route / f"seed_{seed}"
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.is_file():
                    results.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                    print(f"skipped existing protocol={protocol} route={route} seed={seed}", flush=True)
                    continue
                print(f"starting protocol={protocol} route={route} seed={seed}", flush=True)
                result = _run_one(args, dataset, split, protocol=protocol, route=route, seed=seed, run_dir=run_dir)
                results.append(result)
                print(
                    f"completed protocol={protocol} route={route} seed={seed} "
                    f"rmse={_fmt(result['test']['rmse'])} raw_r={_fmt(result['test']['raw_r'])} "
                    f"centered_r={_fmt(result['test']['within_subject_centered_r'])}",
                    flush=True,
                )
            if args.limit_runs and run_number >= args.limit_runs:
                break
        if args.limit_runs and run_number >= args.limit_runs:
            break

    references = _load_reference_results(args, protocols, seeds)
    comparison_rows = _summarize(results + references)
    output = {
        "script": Path(__file__).name,
        "stage": "wear_only_fm_acc_replacement_screen",
        "diagnostic_boundary": "post_phase3_acc_replacement_diagnostic_not_a_formal_promotion_gate",
        "target_label": args.target_label,
        "protocols": list(protocols),
        "routes": list(routes),
        "reference_routes": list(REFERENCE_ROUTES),
        "seeds": seeds,
        "run_count": len(results),
        "reference_count": len(references),
        "elapsed_seconds": float(time.time() - started),
        "mask_count": int(dataset.complete_mask.sum()),
        "acc_feature_count": int(dataset.acc_handcrafted.shape[1]),
        "acc_feature_names": list(dataset.acc_feature_names),
        "results": results,
        "references": references,
        "summary": comparison_rows,
    }
    report_json = args.out_root / "w3fm_acc_replacement_screen_report.json"
    report_md = args.out_root / "w3fm_acc_replacement_screen_report.md"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(comparison_rows, args.out_root / "w3fm_acc_replacement_summary.csv")
    _write_markdown(output, report_md)
    print(f"run_count={len(results)}")
    print(f"reference_count={len(references)}")
    print(f"out_json={report_json}")
    print(f"out_md={report_md}")
    return 0


def _load_dataset(args: argparse.Namespace) -> Dataset:
    rows = _load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([_norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    target = np.asarray([_target_value(row, args.target_label) for row in rows], dtype=np.float32)
    complete_mask = np.load(args.wear_fm_root / "staged_inputs/wear_complete_mask.npy").astype(bool)
    if complete_mask.shape != (len(rows),):
        raise ValueError(f"invalid wear_complete_mask shape: {complete_mask.shape}")

    ppg = _load_array(args.wear_fm_root / "embeddings/ppg_papagei_s_512d.npy", (len(rows), 512), "PPG")
    gsr = _load_array(args.wear_fm_root / "embeddings/gsr_normwear_768d.npy", (len(rows), 768), "GSR")
    for name, path in (
        ("PPG", args.wear_fm_root / "embeddings/ppg_papagei_s_valid_mask.npy"),
        ("GSR", args.wear_fm_root / "embeddings/gsr_normwear_valid_mask.npy"),
    ):
        mask = np.load(path).astype(bool)
        if not np.array_equal(mask, complete_mask):
            raise ValueError(f"{name} embedding mask does not equal wear_complete_mask")

    with np.load(args.wear_fm_root / "staged_inputs/acc_10s.npz", allow_pickle=True) as loaded:
        ids = loaded["sample_id"].astype(str)
        if not np.array_equal(ids, sample_id):
            raise ValueError("staged ACC sample_id does not match canonical index")
        acc_raw = loaded["acc"].astype(np.float32)
        acc_mask = loaded["valid_mask"].astype(bool)
        acc_rate = float(np.asarray(loaded["sample_rate_hz"]).reshape(-1)[0])
    if acc_raw.shape != (len(rows), 3, 300):
        raise ValueError(f"invalid staged ACC shape: {acc_raw.shape}")
    if not np.array_equal(acc_mask, complete_mask):
        raise ValueError("staged ACC valid_mask does not equal wear_complete_mask")
    acc_handcrafted, acc_feature_names = _acc_handcrafted_features(acc_raw, sample_rate_hz=acc_rate)
    if not np.isfinite(acc_handcrafted).all():
        raise ValueError("handcrafted ACC features contain non-finite values")
    return Dataset(sample_id, subject_id, day_id, event_id, target, complete_mask, ppg, gsr, acc_handcrafted, tuple(acc_feature_names))


def _acc_handcrafted_features(acc: np.ndarray, *, sample_rate_hz: float) -> tuple[np.ndarray, list[str]]:
    x = acc[:, 0, :].astype(np.float32)
    y = acc[:, 1, :].astype(np.float32)
    z = acc[:, 2, :].astype(np.float32)
    mag = np.sqrt(x * x + y * y + z * z).astype(np.float32)
    jerk = np.diff(acc, axis=2)
    jerk_mag = np.sqrt(np.sum(jerk * jerk, axis=1)).astype(np.float32)
    arrays = {
        "x": x,
        "y": y,
        "z": z,
        "mag": mag,
        "jerk_mag": jerk_mag,
    }
    features: list[np.ndarray] = []
    names: list[str] = []
    for prefix, values in arrays.items():
        _append_stats(features, names, prefix, values)
    for left_name, left, right_name, right in (("x", x, "y", y), ("x", x, "z", z), ("y", y, "z", z)):
        features.append(_row_corr(left, right))
        names.append(f"corr_{left_name}_{right_name}")
    median_mag = np.median(mag, axis=1)
    positive_centered = np.maximum(mag - median_mag[:, None], 0.0)
    features.append(positive_centered.mean(axis=1).astype(np.float32))
    names.append("mag_positive_centered_mean")
    features.extend(_spectral_features(mag, sample_rate_hz, names))
    out = np.stack(features, axis=1).astype(np.float32)
    return out, names


def _append_stats(features: list[np.ndarray], names: list[str], prefix: str, values: np.ndarray) -> None:
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95], axis=1).astype(np.float32)
    diff = np.diff(values, axis=1)
    stats = {
        "mean": values.mean(axis=1),
        "std": values.std(axis=1),
        "min": values.min(axis=1),
        "max": values.max(axis=1),
        "p05": quantiles[0],
        "p25": quantiles[1],
        "p50": quantiles[2],
        "p75": quantiles[3],
        "p95": quantiles[4],
        "iqr": quantiles[3] - quantiles[1],
        "range": values.max(axis=1) - values.min(axis=1),
        "abs_mean": np.mean(np.abs(values), axis=1),
        "rms": np.sqrt(np.mean(values * values, axis=1)),
        "diff_abs_mean": np.mean(np.abs(diff), axis=1),
        "diff_std": diff.std(axis=1),
    }
    for suffix, arr in stats.items():
        features.append(arr.astype(np.float32))
        names.append(f"{prefix}_{suffix}")


def _row_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = left.astype(np.float64) - left.astype(np.float64).mean(axis=1, keepdims=True)
    b = right.astype(np.float64) - right.astype(np.float64).mean(axis=1, keepdims=True)
    denom = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    out = np.zeros((left.shape[0],), dtype=np.float64)
    ok = denom > 0.0
    out[ok] = np.sum(a[ok] * b[ok], axis=1) / denom[ok]
    return out.astype(np.float32)


def _spectral_features(mag: np.ndarray, sample_rate_hz: float, names: list[str]) -> list[np.ndarray]:
    centered = mag.astype(np.float64) - mag.astype(np.float64).mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(mag.shape[1], d=1.0 / float(sample_rate_hz))
    total = power[:, 1:].sum(axis=1)
    total_safe = np.where(total <= 1e-12, 1.0, total)
    out: list[np.ndarray] = []
    bands = (
        (0.10, 0.50),
        (0.50, 1.50),
        (1.50, 3.00),
        (3.00, 6.00),
        (6.00, 12.00),
        (12.00, min(15.00, float(sample_rate_hz) / 2.0)),
    )
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        value = power[:, mask].sum(axis=1) / total_safe
        out.append(value.astype(np.float32))
        names.append(f"mag_power_ratio_{lo:g}_{hi:g}hz")
    nonzero = power[:, 1:]
    nonzero_freqs = freqs[1:]
    peak = np.argmax(nonzero, axis=1)
    out.append(nonzero_freqs[peak].astype(np.float32))
    names.append("mag_dominant_frequency_hz")
    prob = nonzero / total_safe[:, None]
    entropy = -np.sum(np.where(prob > 0.0, prob * np.log(prob + 1e-12), 0.0), axis=1) / math.log(max(2, prob.shape[1]))
    out.append(entropy.astype(np.float32))
    names.append("mag_spectral_entropy")
    out.append(np.log1p(total).astype(np.float32))
    names.append("mag_log_total_power")
    return out


def _run_one(
    args: argparse.Namespace,
    dataset: Dataset,
    split: dict[str, np.ndarray],
    *,
    protocol: str,
    route: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    model, audit = _fit_route(args, dataset, split, route=route, seed=seed)
    predictions = {
        name: _predict_route(model, dataset, indices=indices, device=args.device, eval_batch_size=args.eval_batch_size)
        for name, indices in _eval_splits(split).items()
    }
    result = {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "source": str(args.wear_fm_root / "staged_inputs/acc_10s.npz"),
        "target_label": args.target_label,
        "mask_count": int(dataset.complete_mask.sum()),
        "split_counts": {name: int(len(indices)) for name, indices in _eval_splits(split).items()},
        "components": list(ROUTE_COMPONENTS[route]),
        "acc_feature_count": int(dataset.acc_handcrafted.shape[1]),
        "train": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["train"]], predictions["train"], dataset.subject_id[split["train"]])),
        "val": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["val"]], predictions["val"], dataset.subject_id[split["val"]])),
        "test": _metric_aliases(evaluate_regression_with_centered(dataset.target[split["test"]], predictions["test"], dataset.subject_id[split["test"]])),
        "train_audit": audit,
    }
    pred_npz = run_dir / "predictions.npz"
    np.savez_compressed(
        pred_npz,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
        event_id=dataset.event_id,
        target=dataset.target,
        train_index=split["train"],
        val_index=split["val"],
        test_index=split["test"],
        train_prediction=predictions["train"],
        val_prediction=predictions["val"],
        test_prediction=predictions["test"],
    )
    _write_predictions_table(run_dir / "test_predictions.csv", dataset, split["test"], predictions["test"])
    torch.save({"state_dict": audit.pop("_best_state"), "route": route, "protocol": protocol, "seed": int(seed)}, run_dir / "best_checkpoint.pt")
    (run_dir / "val_history.csv").write_text(_history_csv(audit["history"]), encoding="utf-8")
    result["prediction_path"] = str(pred_npz)
    result["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")
    result["test_predictions_csv"] = str(run_dir / "test_predictions.csv")
    result["metrics_path"] = str(run_dir / "metrics.json")
    result["config_path"] = str(run_dir / "config.json")
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_config_snapshot(args, protocol, route, seed, dataset), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_route(args: argparse.Namespace, dataset: Dataset, split: dict[str, np.ndarray], *, route: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(args.device)
    train = split["train"]
    components = ROUTE_COMPONENTS[route]
    arrays = {"ppg": dataset.ppg, "acc_handcrafted": dataset.acc_handcrafted, "gsr": dataset.gsr}
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for component in components:
        arr = arrays[component]
        means[component] = arr[train].mean(axis=0, keepdims=True).astype(np.float32)
        std = arr[train].std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        stds[component] = std
    y_mean = float(dataset.target[train].mean())
    y_std = float(dataset.target[train].std()) or 1.0
    input_dims = {name: arrays[name].shape[1] for name in components}
    module = W3FMReplacementModel(components, input_dims, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    def inputs(indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            component: ((arrays[component][indices] - means[component]) / stds[component]).astype(np.float32)
            for component in components
        }

    audit = _train_loop(args, module, train, split["val"], seed, inputs, dataset.target, dataset.subject_id, y_mean, y_std)
    audit["components"] = list(components)
    audit["input_dims"] = input_dims
    return {"module": module, "components": components, "means": means, "stds": stds, "y_mean": y_mean, "y_std": y_std}, audit


def _train_loop(
    args: argparse.Namespace,
    module: torch.nn.Module,
    train: np.ndarray,
    val: np.ndarray,
    seed: int,
    input_fn,
    target: np.ndarray,
    subject_id: np.ndarray,
    y_mean: float,
    y_std: float,
) -> dict[str, Any]:
    device = torch.device(args.device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_val_rmse = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    rng = np.random.default_rng(seed)
    for epoch in range(max(1, args.epochs)):
        module.train()
        losses: list[float] = []
        shuffled = train.copy()
        rng.shuffle(shuffled)
        for start in range(0, len(shuffled), int(args.batch_size)):
            batch = shuffled[start : start + int(args.batch_size)]
            x = _to_torch(input_fn(batch), device)
            y = torch.as_tensor((target[batch] - y_mean) / y_std, dtype=torch.float32, device=device)
            pred = module(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_pred = []
            for start in range(0, len(val), int(args.eval_batch_size)):
                batch = val[start : start + int(args.eval_batch_size)]
                pred = module(_to_torch(input_fn(batch), device)).detach().cpu().numpy()
                val_pred.append((pred * y_std + y_mean).astype(np.float32))
            val_values = np.concatenate(val_pred) if val_pred else np.zeros((0,), dtype=np.float32)
        val_metrics = _metric_aliases(evaluate_regression_with_centered(target[val], val_values, subject_id[val]))
        score = _selection_score(val_metrics, args.selection_metric)
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else math.nan,
            "val_rmse": float(val_metrics["rmse"]),
            "val_mae": float(val_metrics["mae"]),
            "val_raw_r": float(val_metrics["raw_r"]),
            "val_within_subject_centered_r": float(val_metrics["within_subject_centered_r"]),
            "val_prediction_std": float(np.std(val_values)) if len(val_values) else math.nan,
            "selection_score": float(score),
        }
        history.append(row)
        if score > best_score:
            best_score = score
            best_val_rmse = float(val_metrics["rmse"])
            best_epoch = epoch + 1
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no best state")
    module.load_state_dict(best_state)
    return {
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_val_rmse),
        "best_selection_score": float(best_score),
        "selection_metric": str(args.selection_metric),
        "epoch_count": int(len(history)),
        "history": history,
        "trainable_params": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in module.parameters())),
        "_best_state": best_state,
    }


def _predict_route(model: dict[str, Any], dataset: Dataset, *, indices: np.ndarray, device: str, eval_batch_size: int) -> np.ndarray:
    module = model["module"]
    module.eval()
    arrays = {"ppg": dataset.ppg, "acc_handcrafted": dataset.acc_handcrafted, "gsr": dataset.gsr}
    values: list[np.ndarray] = []
    dev = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(indices), int(eval_batch_size)):
            batch = indices[start : start + int(eval_batch_size)]
            x = {
                component: ((arrays[component][batch] - model["means"][component]) / model["stds"][component]).astype(np.float32)
                for component in model["components"]
            }
            pred = module(_to_torch(x, dev)).detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _load_reference_results(args: argparse.Namespace, protocols: tuple[str, ...], seeds: list[int]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for protocol in protocols:
        for seed in seeds:
            for path in (
                args.phase2_root / "runs" / protocol / "W3FM_frozen" / f"seed_{seed}" / "metrics.json",
                args.internal_root / "runs" / protocol / "W3FM_no_acc" / f"seed_{seed}" / "metrics.json",
            ):
                row = _maybe_metric(path)
                if row is not None:
                    refs.append(row)
    return refs


def _maybe_metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    row = json.loads(path.read_text(encoding="utf-8"))
    row = dict(row)
    row["reference_source"] = str(path)
    return row


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key = {(row["protocol"], row["route"], int(row["seed"])): row for row in rows}
    order = ("W3FM_frozen", "W3FM_no_acc", "W3FM_acc_handcrafted")
    for protocol in sorted({row["protocol"] for row in rows}):
        for route in order:
            route_rows = [row for row in rows if row["protocol"] == protocol and row["route"] == route]
            if not route_rows:
                continue
            deltas_w3fm = [_delta(row, by_key.get((protocol, "W3FM_frozen", int(row["seed"])))) for row in route_rows]
            deltas_no_acc = [_delta(row, by_key.get((protocol, "W3FM_no_acc", int(row["seed"])))) for row in route_rows]
            out.append(
                {
                    "protocol": protocol,
                    "route": route,
                    "seed_count": int(len(route_rows)),
                    "test_rmse_mean": _mean_metric(route_rows, "rmse"),
                    "test_rmse_std": _std_metric(route_rows, "rmse"),
                    "test_raw_r_mean": _mean_metric(route_rows, "raw_r"),
                    "test_raw_r_std": _std_metric(route_rows, "raw_r"),
                    "test_centered_r_mean": _mean_metric(route_rows, "within_subject_centered_r"),
                    "test_centered_r_std": _std_metric(route_rows, "within_subject_centered_r"),
                    "delta_raw_r_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "raw_r"),
                    "delta_rmse_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "rmse"),
                    "delta_centered_r_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "within_subject_centered_r"),
                    "delta_raw_r_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "raw_r"),
                    "delta_rmse_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "rmse"),
                    "delta_centered_r_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "within_subject_centered_r"),
                    "best_epoch_mean": _mean_audit(route_rows, "best_epoch"),
                    "trainable_params_mean": _mean_audit(route_rows, "trainable_params"),
                }
            )
    return out


def _delta(row: dict[str, Any], base: dict[str, Any] | None) -> dict[str, float | None]:
    if base is None:
        return {"raw_r": None, "rmse": None, "within_subject_centered_r": None}
    return {
        "raw_r": float(row["test"]["raw_r"]) - float(base["test"]["raw_r"]),
        "rmse": float(row["test"]["rmse"]) - float(base["test"]["rmse"]),
        "within_subject_centered_r": float(row["test"]["within_subject_centered_r"]) - float(base["test"]["within_subject_centered_r"]),
    }


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.mean([float(row["test"][metric]) for row in rows])) if rows else math.nan


def _std_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.std([float(row["test"][metric]) for row in rows], ddof=0)) if rows else math.nan


def _mean_delta(rows: list[dict[str, float | None]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def _mean_audit(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row["train_audit"][field]) for row in rows])) if rows else math.nan


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# W3FM ACC Replacement Screen",
        "",
        f"- diagnostic_boundary: `{output['diagnostic_boundary']}`",
        f"- run_count: `{output['run_count']}`",
        f"- reference_count: `{output['reference_count']}`",
        f"- mask_count: `{output['mask_count']}`",
        f"- acc_feature_count: `{output['acc_feature_count']}`",
        "",
        "## Route Summary",
        "",
        "| protocol | route | seeds | RMSE | raw r | centered r | d raw r vs W3FM | d RMSE vs W3FM | d centered r vs W3FM | d raw r vs no-ACC | d RMSE vs no-ACC | d centered r vs no-ACC | best epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed_count']} | "
            f"{_fmt(row['test_rmse_mean'])} +/- {_fmt(row['test_rmse_std'])} | "
            f"{_fmt(row['test_raw_r_mean'])} +/- {_fmt(row['test_raw_r_std'])} | "
            f"{_fmt(row['test_centered_r_mean'])} +/- {_fmt(row['test_centered_r_std'])} | "
            f"{_fmt(row['delta_raw_r_vs_W3FM_frozen_mean'])} | "
            f"{_fmt(row['delta_rmse_vs_W3FM_frozen_mean'])} | "
            f"{_fmt(row['delta_centered_r_vs_W3FM_frozen_mean'])} | "
            f"{_fmt(row['delta_raw_r_vs_W3FM_no_acc_mean'])} | "
            f"{_fmt(row['delta_rmse_vs_W3FM_no_acc_mean'])} | "
            f"{_fmt(row['delta_centered_r_vs_W3FM_no_acc_mean'])} | "
            f"{_fmt(row['best_epoch_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Per Run",
            "",
            "| protocol | route | seed | test RMSE | test MAE | test raw r | test centered r | best epoch | trainable params |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in output["references"] + output["results"]:
        test = row["test"]
        audit = row["train_audit"]
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed']} | {_fmt(test['rmse'])} | {_fmt(test['mae'])} | "
            f"{_fmt(test['raw_r'])} | {_fmt(test['within_subject_centered_r'])} | "
            f"{audit['best_epoch']} | {audit['trainable_params']} |"
        )
    lines.extend(
        [
            "",
            "## ACC Feature Families",
            "",
            "Handcrafted ACC features are deterministic label-free statistics from the 10-second staged ACC window. Downstream train-only normalization is still fitted separately for each protocol and seed.",
            "",
            "```json",
            json.dumps(output["acc_feature_names"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_preflight(dataset: Dataset, path: Path) -> None:
    payload = {
        "row_count": int(dataset.sample_id.shape[0]),
        "wear_complete_count": int(dataset.complete_mask.sum()),
        "target_finite": bool(np.isfinite(dataset.target).all()),
        "ppg_embedding_shape": list(dataset.ppg.shape),
        "gsr_embedding_shape": list(dataset.gsr.shape),
        "acc_handcrafted_shape": list(dataset.acc_handcrafted.shape),
        "acc_handcrafted_finite": bool(np.isfinite(dataset.acc_handcrafted).all()),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_feature_manifest(dataset: Dataset, out_root: Path) -> None:
    payload = {
        "source": "outputs/wear_fm/staged_inputs/acc_10s.npz",
        "feature_policy": "label_free_deterministic_window_statistics_no_test_selection",
        "shape": list(dataset.acc_handcrafted.shape),
        "feature_names": list(dataset.acc_feature_names),
        "global_mean": float(np.mean(dataset.acc_handcrafted)),
        "global_std": float(np.std(dataset.acc_handcrafted)),
        "finite": bool(np.isfinite(dataset.acc_handcrafted).all()),
    }
    (out_root / "acc_handcrafted_feature_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    if not summary:
        path.write_text("", encoding="utf-8")
        return
    fields = list(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _write_predictions_table(path: Path, dataset: Dataset, indices: np.ndarray, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "sample_id", "subject_id", "day_id", "event_id", "target", "prediction"])
        for row_id, pred in zip(indices.tolist(), prediction.tolist()):
            writer.writerow([row_id, dataset.sample_id[row_id], dataset.subject_id[row_id], dataset.day_id[row_id], dataset.event_id[row_id], float(dataset.target[row_id]), float(pred)])


def _history_csv(history: list[dict[str, Any]]) -> str:
    fields = [
        "epoch",
        "train_loss",
        "val_rmse",
        "val_mae",
        "val_raw_r",
        "val_within_subject_centered_r",
        "val_prediction_std",
        "selection_score",
    ]
    rows = [",".join(fields)]
    rows.extend(",".join(str(item.get(field, "")) for field in fields) for item in history)
    return "\n".join(rows) + "\n"


def _config_snapshot(args: argparse.Namespace, protocol: str, route: str, seed: int, dataset: Dataset) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "components": list(ROUTE_COMPONENTS[route]),
        "acc_feature_count": int(dataset.acc_handcrafted.shape[1]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "hidden_dim": int(args.hidden_dim),
        "patience": int(args.patience),
        "selection_metric": str(args.selection_metric),
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "mask_policy": "same wear_complete_mask intersected with split indices",
        "diagnostic_boundary": "post_phase3_acc_replacement_diagnostic_not_a_formal_promotion_gate",
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_value(row: dict[str, Any], target_label: str) -> float:
    if "label_columns" in row and target_label in row["label_columns"]:
        return float(row["label_columns"][target_label])
    label_names = tuple(str(value) for value in row.get("label_names") or LABEL_NAMES)
    if target_label not in label_names:
        raise ValueError(f"target label {target_label!r} missing from row")
    return float(row["labels"][label_names.index(target_label)])


def _load_array(path: Path, shape: tuple[int, int], name: str) -> np.ndarray:
    values = np.load(path).astype(np.float32)
    if values.shape != shape:
        raise ValueError(f"{name} shape {values.shape} is not {shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _load_split(path: Path, n_rows: int) -> dict[str, np.ndarray]:
    split = {name: _load_indices(path / f"{name}.json", n_rows) for name in ("pretrain", "finetune", "val", "test")}
    split["train"] = np.asarray(split["pretrain"].tolist() + split["finetune"].tolist(), dtype=np.int64)
    return split


def _load_indices(path: Path, n_rows: int) -> np.ndarray:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("indices", value.get("index", value.get("rows")))
    values = np.asarray(value, dtype=np.int64).reshape(-1)
    if values.size and (values.min() < 0 or values.max() >= n_rows):
        raise ValueError(f"{path} contains out-of-range indices")
    return values


def _filter_split(split: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    out = {name: indices[mask[indices]].astype(np.int64) for name, indices in split.items()}
    for name in ("train", "val", "test"):
        if len(out[name]) == 0:
            raise ValueError(f"{name} split is empty after wear_complete_mask")
    return out


def _eval_splits(split: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: split[name] for name in ("train", "val", "test")}


def _metric_aliases(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["pearson_r"] = result.get("raw_r")
    result["per_subject_r_mean"] = result.get("per_subject_r", {}).get("mean")
    result["per_subject_r_std"] = result.get("per_subject_r", {}).get("std")
    return result


def _selection_score(metrics: dict[str, Any], metric: str) -> float:
    if metric == "rmse":
        return -float(metrics["rmse"])
    if metric == "raw_r":
        return float(metrics["raw_r"])
    if metric == "centered_r":
        return float(metrics["within_subject_centered_r"])
    raise ValueError(f"unsupported selection metric: {metric}")


def _to_torch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: torch.as_tensor(item, dtype=torch.float32, device=device) for key, item in value.items()}
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _validate_routes(routes: tuple[str, ...]) -> None:
    unknown = set(routes) - set(DEFAULT_ROUTES)
    if unknown:
        raise ValueError(f"unsupported routes: {sorted(unknown)}")


def _norm_subject(value: Any) -> str:
    text = str(value)
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(float(text)):02d}"
    except (TypeError, ValueError):
        return text


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
