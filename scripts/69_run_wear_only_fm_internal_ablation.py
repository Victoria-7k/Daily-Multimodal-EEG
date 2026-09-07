#!/usr/bin/env python3
"""Run diagnostic W3FM internal ablations for the wear-only FM experiment."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered


DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_WEAR_FM_ROOT = DEFAULT_ROOT / "outputs/wear_fm"
DEFAULT_PHASE2_ROOT = DEFAULT_ROOT / "outputs/wear_fm/phase2"
DEFAULT_PROTOCOLS = ("cross_day", "within_subject_day")
DEFAULT_ROUTES = ("W3FM_no_ppg", "W3FM_no_acc", "W3FM_no_gsr", "W3FM_ppg_partial_ft")
DEFAULT_SEEDS = (240729, 240730, 240731)
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
)
COMPONENT_DIMS = {"ppg": 512, "acc": 1024, "gsr": 768}
ROUTE_COMPONENTS = {
    "W3FM_no_ppg": ("acc", "gsr"),
    "W3FM_no_acc": ("ppg", "gsr"),
    "W3FM_no_gsr": ("ppg", "acc"),
    "W3FM_ppg_partial_ft": ("ppg", "acc", "gsr"),
}


@dataclass(frozen=True)
class Dataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    event_id: np.ndarray
    target: np.ndarray
    complete_mask: np.ndarray
    ppg: np.ndarray
    acc: np.ndarray
    gsr: np.ndarray
    ppg_raw: np.ndarray


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


class W3FMDiagnosticModel(torch.nn.Module):
    def __init__(
        self,
        components: tuple[str, ...],
        *,
        hidden_dim: int,
        dropout: float,
        ppg_encoder: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.components = tuple(components)
        self.ppg_encoder = ppg_encoder
        self.projections = torch.nn.ModuleDict({name: _projector(COMPONENT_DIMS[name], dropout) for name in components})
        self.gate = torch.nn.Linear(256, 1)
        self.head = WearHead(input_dim=256, hidden_dim=hidden_dim, dropout=dropout)

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        if self.ppg_encoder is not None:
            self.ppg_encoder.eval()
        return self

    def _ppg_features(self, ppg_raw: torch.Tensor) -> torch.Tensor:
        if self.ppg_encoder is None:
            return ppg_raw
        y = self.ppg_encoder(ppg_raw.unsqueeze(1))
        return y[0] if isinstance(y, tuple) else y

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []
        for component in self.components:
            value = inputs[component]
            if component == "ppg":
                value = self._ppg_features(value)
            tokens.append(self.projections[component](value))
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
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--target-label", default="fatigue")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppg-partial-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ppg-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--selection-metric", choices=("rmse", "raw_r", "centered_r"), default="rmse")
    parser.add_argument("--ppg-trainable-tail-params", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT / "outputs/wear_fm/internal_ablation")
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
    _write_preflight(dataset, args.out_root / "internal_ablation_preflight.json")

    results: list[dict[str, Any]] = []
    started = time.time()
    run_number = 0
    for protocol in protocols:
        split = _filter_split(_load_split(args.splits_root / protocol, len(dataset.sample_id)), dataset.complete_mask)
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

    phase2 = _load_phase2_results(args.phase2_root)
    summary = _summarize(results, phase2)
    output = {
        "script": Path(__file__).name,
        "stage": "wear_only_fm_internal_ablation",
        "diagnostic_boundary": "post_phase3_failure_diagnostic_not_a_formal_promotion_gate",
        "target_label": args.target_label,
        "protocols": list(protocols),
        "routes": list(routes),
        "seeds": seeds,
        "run_count": len(results),
        "elapsed_seconds": float(time.time() - started),
        "mask_count": int(dataset.complete_mask.sum()),
        "results": results,
        "summary": summary,
    }
    report_json = args.out_root / "wear_only_fm_internal_ablation_report.json"
    report_md = args.out_root / "wear_only_fm_internal_ablation_report.md"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary, args.out_root / "internal_ablation_summary.csv")
    _write_markdown(output, report_md)
    print(f"run_count={len(results)}")
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
    acc = _load_array(args.wear_fm_root / "embeddings/acc_harnet10.npy", (len(rows), 1024), "ACC")
    gsr = _load_array(args.wear_fm_root / "embeddings/gsr_normwear_768d.npy", (len(rows), 768), "GSR")
    for name, path in (
        ("PPG", args.wear_fm_root / "embeddings/ppg_papagei_s_valid_mask.npy"),
        ("ACC", args.wear_fm_root / "embeddings/acc_harnet10_valid_mask.npy"),
        ("GSR", args.wear_fm_root / "embeddings/gsr_normwear_valid_mask.npy"),
    ):
        mask = np.load(path).astype(bool)
        if not np.array_equal(mask, complete_mask):
            raise ValueError(f"{name} embedding mask does not equal wear_complete_mask")

    with np.load(args.wear_fm_root / "staged_inputs/ppg_10s.npz", allow_pickle=True) as loaded:
        ppg_ids = loaded["sample_id"].astype(str)
        if not np.array_equal(ppg_ids, sample_id):
            raise ValueError("staged PPG sample_id does not match canonical index")
        ppg_raw = loaded["ppg"].astype(np.float32)
        ppg_mask = loaded["valid_mask"].astype(bool)
    if ppg_raw.shape != (len(rows), 1250):
        raise ValueError(f"invalid staged PPG shape: {ppg_raw.shape}")
    if not np.array_equal(ppg_mask, complete_mask):
        raise ValueError("staged PPG valid_mask does not equal wear_complete_mask")
    return Dataset(sample_id, subject_id, day_id, event_id, target, complete_mask, ppg, acc, gsr, ppg_raw)


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
        "source": str(args.wear_fm_root / "embeddings/embedding_manifest.json"),
        "target_label": args.target_label,
        "mask_count": int(dataset.complete_mask.sum()),
        "split_counts": {name: int(len(indices)) for name, indices in _eval_splits(split).items()},
        "components": list(ROUTE_COMPONENTS[route]),
        "ppg_partial_ft": bool(route == "W3FM_ppg_partial_ft"),
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
    (run_dir / "config.json").write_text(json.dumps(_config_snapshot(args, protocol, route, seed), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_route(args: argparse.Namespace, dataset: Dataset, split: dict[str, np.ndarray], *, route: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    device = torch.device(args.device)
    train = split["train"]
    val = split["val"]
    components = ROUTE_COMPONENTS[route]
    partial_ppg = route == "W3FM_ppg_partial_ft"
    frozen_arrays = {"ppg": dataset.ppg, "acc": dataset.acc, "gsr": dataset.gsr}
    means: dict[str, np.ndarray] = {}
    stds: dict[str, np.ndarray] = {}
    for component in components:
        if component == "ppg" and partial_ppg:
            continue
        arr = frozen_arrays[component]
        means[component] = arr[train].mean(axis=0, keepdims=True).astype(np.float32)
        std = arr[train].std(axis=0, keepdims=True).astype(np.float32)
        std[std < 1e-6] = 1.0
        stds[component] = std

    y_mean = float(dataset.target[train].mean())
    y_std = float(dataset.target[train].std()) or 1.0
    ppg_encoder = None
    ppg_trainable_names: list[str] = []
    if partial_ppg:
        ppg_encoder = _papagei_model(args.wear_fm_root, args.device)
        ppg_trainable_names = _configure_tail_trainable(ppg_encoder, args.ppg_trainable_tail_params)
    module = W3FMDiagnosticModel(components, hidden_dim=args.hidden_dim, dropout=args.dropout, ppg_encoder=ppg_encoder).to(device)

    def inputs(indices: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for component in components:
            if component == "ppg" and partial_ppg:
                out[component] = _zscore_rows(dataset.ppg_raw[indices])
            else:
                arr = frozen_arrays[component]
                out[component] = ((arr[indices] - means[component]) / stds[component]).astype(np.float32)
        return out

    batch_size = int(args.ppg_partial_batch_size if partial_ppg else args.batch_size)
    audit = _train_loop(
        args,
        module,
        train,
        val,
        seed,
        inputs,
        dataset.target,
        dataset.subject_id,
        y_mean,
        y_std,
        batch_size=batch_size,
        ppg_partial=partial_ppg,
    )
    audit["components"] = list(components)
    audit["ppg_partial_ft"] = bool(partial_ppg)
    audit["ppg_trainable_tail_params"] = int(args.ppg_trainable_tail_params if partial_ppg else 0)
    audit["ppg_trainable_parameter_names"] = ppg_trainable_names
    return {"module": module, "components": components, "means": means, "stds": stds, "y_mean": y_mean, "y_std": y_std, "ppg_partial_ft": partial_ppg}, audit


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
    *,
    batch_size: int,
    ppg_partial: bool,
) -> dict[str, Any]:
    device = torch.device(args.device)
    optimizer = _build_optimizer(module, head_lr=args.learning_rate, encoder_lr=args.ppg_encoder_lr, weight_decay=args.weight_decay)
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
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            x = _to_torch(input_fn(batch), device)
            y = torch.as_tensor((target[batch] - y_mean) / y_std, dtype=torch.float32, device=device)
            pred = module(x)
            loss = torch.mean((pred - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for param in module.parameters() if param.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        module.eval()
        with torch.no_grad():
            val_pred = []
            eval_batch = min(int(args.eval_batch_size), batch_size) if ppg_partial else int(args.eval_batch_size)
            for start in range(0, len(val), eval_batch):
                batch = val[start : start + eval_batch]
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
    dev = torch.device(device)
    frozen_arrays = {"ppg": dataset.ppg, "acc": dataset.acc, "gsr": dataset.gsr}
    values: list[np.ndarray] = []
    batch_size = min(eval_batch_size, 128) if model["ppg_partial_ft"] else eval_batch_size
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            x: dict[str, np.ndarray] = {}
            for component in model["components"]:
                if component == "ppg" and model["ppg_partial_ft"]:
                    x[component] = _zscore_rows(dataset.ppg_raw[batch])
                else:
                    arr = frozen_arrays[component]
                    x[component] = ((arr[batch] - model["means"][component]) / model["stds"][component]).astype(np.float32)
            pred = module(_to_torch(x, dev)).detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _build_optimizer(module: torch.nn.Module, *, head_lr: float, encoder_lr: float, weight_decay: float) -> torch.optim.Optimizer:
    encoder_params = []
    head_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("ppg_encoder."):
            encoder_params.append(param)
        else:
            head_params.append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    if encoder_params:
        groups.append({"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay})
    if not groups:
        raise ValueError("no trainable parameters")
    return torch.optim.AdamW(groups)


def _papagei_model(base: Path, device_name: str) -> torch.nn.Module:
    repo = base / "third_party/papagei-foundation-model"
    sys.path.insert(0, str(repo))
    from models.resnet import ResNet1DMoE

    model = ResNet1DMoE(
        in_channels=1,
        base_filters=32,
        kernel_size=3,
        stride=2,
        groups=1,
        n_block=18,
        n_classes=512,
        n_experts=3,
    )
    checkpoint = torch.load(base / "weights/papagei_s.pt", map_location="cpu")
    state = {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint.items()}
    model.load_state_dict(state)
    return model.to(torch.device(device_name)).eval()


def _configure_tail_trainable(module: torch.nn.Module, tail_params: int) -> list[str]:
    for param in module.parameters():
        param.requires_grad = False
    named = list(module.named_parameters())
    selected = named[-max(0, int(tail_params)) :] if tail_params else []
    for _, param in selected:
        param.requires_grad = True
    return [name for name, _ in selected]


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32)


def _to_torch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: torch.as_tensor(item, dtype=torch.float32, device=device) for key, item in value.items()}
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _load_phase2_results(phase2_root: Path) -> list[dict[str, Any]]:
    report = phase2_root / "wear_only_fm_phase2_report.json"
    if not report.is_file():
        return []
    return json.loads(report.read_text(encoding="utf-8")).get("results", [])


def _summarize(results: list[dict[str, Any]], phase2: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase2_by_key = {(row["protocol"], row["route"], int(row["seed"])): row for row in phase2}
    out: list[dict[str, Any]] = []
    for protocol in sorted({row["protocol"] for row in results}):
        for route in sorted({row["route"] for row in results if row["protocol"] == protocol}):
            rows = [row for row in results if row["protocol"] == protocol and row["route"] == route]
            deltas = []
            for row in rows:
                base = phase2_by_key.get((protocol, "W3FM_frozen", int(row["seed"])))
                best = _best_phase2_baseline(phase2, protocol, int(row["seed"]))
                deltas.append(
                    {
                        "raw_r_vs_w3fm": _delta(row, base, "raw_r"),
                        "rmse_vs_w3fm": _delta(row, base, "rmse"),
                        "centered_r_vs_w3fm": _delta(row, base, "within_subject_centered_r"),
                        "raw_r_vs_best": _delta(row, best, "raw_r"),
                        "rmse_vs_best": _delta(row, best, "rmse"),
                        "centered_r_vs_best": _delta(row, best, "within_subject_centered_r"),
                    }
                )
            out.append(
                {
                    "protocol": protocol,
                    "route": route,
                    "seed_count": int(len(rows)),
                    "test_rmse_mean": _mean_metric(rows, "rmse"),
                    "test_rmse_std": _std_metric(rows, "rmse"),
                    "test_raw_r_mean": _mean_metric(rows, "raw_r"),
                    "test_raw_r_std": _std_metric(rows, "raw_r"),
                    "test_centered_r_mean": _mean_metric(rows, "within_subject_centered_r"),
                    "test_centered_r_std": _std_metric(rows, "within_subject_centered_r"),
                    "delta_raw_r_vs_W3FM_frozen_mean": _mean_delta(deltas, "raw_r_vs_w3fm"),
                    "delta_rmse_vs_W3FM_frozen_mean": _mean_delta(deltas, "rmse_vs_w3fm"),
                    "delta_centered_r_vs_W3FM_frozen_mean": _mean_delta(deltas, "centered_r_vs_w3fm"),
                    "delta_raw_r_vs_best_baseline_mean": _mean_delta(deltas, "raw_r_vs_best"),
                    "delta_rmse_vs_best_baseline_mean": _mean_delta(deltas, "rmse_vs_best"),
                    "delta_centered_r_vs_best_baseline_mean": _mean_delta(deltas, "centered_r_vs_best"),
                }
            )
    return out


def _best_phase2_baseline(phase2: list[dict[str, Any]], protocol: str, seed: int) -> dict[str, Any] | None:
    candidates = [
        row
        for row in phase2
        if row.get("protocol") == protocol and int(row.get("seed")) == seed and row.get("route") in {"Wphysio", "Wdeep", "Wmoment_frozen"}
    ]
    if not candidates:
        return None
    if protocol == "within_subject_day":
        return max(candidates, key=lambda row: (float(row["test"]["within_subject_centered_r"]), -float(row["test"]["rmse"])))
    return max(candidates, key=lambda row: (float(row["test"]["raw_r"]), -float(row["test"]["rmse"])))


def _delta(row: dict[str, Any], base: dict[str, Any] | None, metric: str) -> float | None:
    if base is None:
        return None
    return float(row["test"][metric]) - float(base["test"][metric])


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.mean([float(row["test"][metric]) for row in rows])) if rows else math.nan


def _std_metric(rows: list[dict[str, Any]], metric: str) -> float:
    return float(np.std([float(row["test"][metric]) for row in rows], ddof=0)) if rows else math.nan


def _mean_delta(rows: list[dict[str, float | None]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def _write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    if not summary:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# Wear-only FM Internal Ablation Report",
        "",
        f"- diagnostic_boundary: `{output['diagnostic_boundary']}`",
        f"- run_count: `{output['run_count']}`",
        f"- mask_count: `{output['mask_count']}`",
        "",
        "## Route Summary",
        "",
        "| protocol | route | seeds | RMSE | raw r | centered r | d raw r vs W3FM | d RMSE vs W3FM | d centered r vs W3FM |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed_count']} | "
            f"{_fmt(row['test_rmse_mean'])} +/- {_fmt(row['test_rmse_std'])} | "
            f"{_fmt(row['test_raw_r_mean'])} +/- {_fmt(row['test_raw_r_std'])} | "
            f"{_fmt(row['test_centered_r_mean'])} +/- {_fmt(row['test_centered_r_std'])} | "
            f"{_fmt(row['delta_raw_r_vs_W3FM_frozen_mean'])} | "
            f"{_fmt(row['delta_rmse_vs_W3FM_frozen_mean'])} | "
            f"{_fmt(row['delta_centered_r_vs_W3FM_frozen_mean'])} |"
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
    for row in output["results"]:
        test = row["test"]
        audit = row["train_audit"]
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed']} | {_fmt(test['rmse'])} | {_fmt(test['mae'])} | "
            f"{_fmt(test['raw_r'])} | {_fmt(test['within_subject_centered_r'])} | "
            f"{audit['best_epoch']} | {audit['trainable_params']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_predictions_table(path: Path, dataset: Dataset, indices: np.ndarray, prediction: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "sample_id", "subject_id", "day_id", "event_id", "target", "prediction"])
        for row_id, pred in zip(indices.tolist(), prediction.tolist()):
            writer.writerow([row_id, dataset.sample_id[row_id], dataset.subject_id[row_id], dataset.day_id[row_id], dataset.event_id[row_id], float(dataset.target[row_id]), float(pred)])


def _write_preflight(dataset: Dataset, path: Path) -> None:
    payload = {
        "row_count": int(dataset.sample_id.shape[0]),
        "wear_complete_count": int(dataset.complete_mask.sum()),
        "target_finite": bool(np.isfinite(dataset.target).all()),
        "ppg_embedding_shape": list(dataset.ppg.shape),
        "acc_embedding_shape": list(dataset.acc.shape),
        "gsr_embedding_shape": list(dataset.gsr.shape),
        "ppg_raw_shape": list(dataset.ppg_raw.shape),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def _config_snapshot(args: argparse.Namespace, protocol: str, route: str, seed: int) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "components": list(ROUTE_COMPONENTS[route]),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "ppg_partial_batch_size": int(args.ppg_partial_batch_size),
        "learning_rate": float(args.learning_rate),
        "ppg_encoder_lr": float(args.ppg_encoder_lr),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "hidden_dim": int(args.hidden_dim),
        "patience": int(args.patience),
        "selection_metric": str(args.selection_metric),
        "ppg_trainable_tail_params": int(args.ppg_trainable_tail_params if route == "W3FM_ppg_partial_ft" else 0),
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "mask_policy": "wear_complete_mask intersected with split indices for all routes",
        "diagnostic_boundary": "post_phase3_failure_diagnostic_not_a_formal_promotion_gate",
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
    return "NA" if value is None else f"{float(value):.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
