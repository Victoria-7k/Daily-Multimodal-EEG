#!/usr/bin/env python3
"""Run a W3FM diagnostic where PPG, ACC, and GSR all use NormWear embeddings."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError(f"could not locate repo root from {here}")


ROOT = _find_repo_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.training.centered_metrics import evaluate_regression_with_centered
from daily_multimodal.split_paths import resolve_protocol_split_root


def _load_base_helpers():
    path = Path(__file__).resolve().with_name("89_run_w3fm_acc_handcrafted_screen.py")
    if not path.is_file():
        raise FileNotFoundError(f"missing helper script: {path}")
    spec = importlib.util.spec_from_file_location("_w3fm_acc_handcrafted_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_helpers()

DEFAULT_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DEFAULT_SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")
DEFAULT_WEAR_FM_ROOT = DEFAULT_ROOT / "outputs/wear_fm"
DEFAULT_PHASE2_ROOT = DEFAULT_WEAR_FM_ROOT / "phase2"
DEFAULT_INTERNAL_ROOT = DEFAULT_WEAR_FM_ROOT / "internal_ablation"
DEFAULT_ACC_REPLACEMENT_ROOT = DEFAULT_WEAR_FM_ROOT / "acc_replacement_screen"
DEFAULT_OUT_ROOT = DEFAULT_WEAR_FM_ROOT / "normwear_all_screen"
DEFAULT_PROTOCOLS = ("cross_day", "date_in_order")
DEFAULT_SEEDS = (240729, 240730, 240731)
DEFAULT_ROUTES = ("W3FM_normwear_all",)
LABEL_NAMES = BASE.LABEL_NAMES
ROUTE_COMPONENTS = {
    "W3FM_normwear_all": ("ppg_normwear", "acc_normwear", "gsr_normwear"),
}
REFERENCE_ORDER = (
    "Wphysio",
    "Wdeep",
    "Wmoment_frozen",
    "W3FM_frozen",
    "W3FM_no_acc",
    "W3FM_acc_handcrafted",
    "W3FM_normwear_all",
)
MODALITY_SETTINGS = {
    "ppg": {"key": "ppg", "rate": 125.0, "path": "ppg_10s.npz", "axis_pool": "squeeze_or_mean"},
    "acc": {"key": "acc", "rate": 30.0, "path": "acc_10s.npz", "axis_pool": "mean"},
    "gsr": {"key": "gsr", "rate": 40.0, "path": "gsr_10s.npz", "axis_pool": "squeeze_or_mean"},
}


@dataclass(frozen=True)
class Dataset:
    sample_id: np.ndarray
    subject_id: np.ndarray
    day_id: np.ndarray
    event_id: np.ndarray
    target: np.ndarray
    complete_mask: np.ndarray
    ppg_normwear: np.ndarray
    acc_normwear: np.ndarray
    gsr_normwear: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--wear-fm-root", type=Path, default=DEFAULT_WEAR_FM_ROOT)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT)
    parser.add_argument("--internal-root", type=Path, default=DEFAULT_INTERNAL_ROOT)
    parser.add_argument("--acc-replacement-root", type=Path, default=DEFAULT_ACC_REPLACEMENT_ROOT)
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
    parser.add_argument("--ppg-normwear-batch-size", type=int, default=16)
    parser.add_argument("--acc-normwear-batch-size", type=int, default=16)
    parser.add_argument("--gsr-normwear-batch-size", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=2000)
    parser.add_argument("--limit-embedding-batches", type=int, default=0)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-cache", action="store_true")
    parser.add_argument("--cache-only", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")

    args.out_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    dataset, embedding_manifest = _load_dataset(args)
    _write_preflight(dataset, embedding_manifest, args.out_root / "normwear_all_preflight.json")
    if args.cache_only:
        print(f"cache_only=true out_root={args.out_root}")
        return 0

    protocols = BASE._split_csv(args.protocols)
    routes = BASE._split_csv(args.routes)
    seeds = [int(value) for value in BASE._split_csv(args.seeds)]
    _validate_routes(routes)

    results: list[dict[str, Any]] = []
    run_number = 0
    for protocol in protocols:
        split = BASE._filter_split(
            BASE._load_split(resolve_protocol_split_root(args.splits_root, protocol), len(dataset.sample_id)),
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
                    f"rmse={BASE._fmt(result['test']['rmse'])} raw_r={BASE._fmt(result['test']['raw_r'])} "
                    f"centered_r={BASE._fmt(result['test']['within_subject_centered_r'])}",
                    flush=True,
                )
            if args.limit_runs and run_number >= args.limit_runs:
                break
        if args.limit_runs and run_number >= args.limit_runs:
            break

    references = _load_reference_results(args, protocols, seeds)
    summary = _summarize(results + references)
    output = {
        "script": Path(__file__).name,
        "stage": "wear_only_fm_normwear_all_screen",
        "diagnostic_boundary": "post_phase3_normwear_all_diagnostic_not_a_formal_promotion_gate",
        "target_label": args.target_label,
        "protocols": list(protocols),
        "routes": list(routes),
        "reference_routes": [route for route in REFERENCE_ORDER if route not in routes],
        "seeds": seeds,
        "run_count": len(results),
        "reference_count": len(references),
        "elapsed_seconds": float(time.time() - started),
        "mask_count": int(dataset.complete_mask.sum()),
        "embedding_manifest": embedding_manifest,
        "results": results,
        "references": references,
        "summary": summary,
    }
    report_json = args.out_root / "w3fm_normwear_all_screen_report.json"
    report_md = args.out_root / "w3fm_normwear_all_screen_report.md"
    report_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(summary, args.out_root / "w3fm_normwear_all_summary.csv")
    _write_markdown(output, report_md)
    print(f"run_count={len(results)}")
    print(f"reference_count={len(references)}")
    print(f"out_json={report_json}")
    print(f"out_md={report_md}")
    return 0


def _load_dataset(args: argparse.Namespace) -> tuple[Dataset, dict[str, Any]]:
    rows = BASE._load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=str)
    subject_id = np.asarray([BASE._norm_subject(row.get("subject_id")) for row in rows], dtype=str)
    day_id = np.asarray([str(row.get("day_id", "")) for row in rows], dtype=str)
    event_id = np.asarray([str(row.get("event_id", "")) for row in rows], dtype=str)
    target = np.asarray([BASE._target_value(row, args.target_label) for row in rows], dtype=np.float32)
    complete_mask = np.load(args.wear_fm_root / "staged_inputs/wear_complete_mask.npy").astype(bool)
    if complete_mask.shape != (len(rows),):
        raise ValueError(f"invalid wear_complete_mask shape: {complete_mask.shape}")

    embeddings, manifest = _load_or_extract_normwear_embeddings(args, sample_id, complete_mask)
    dataset = Dataset(
        sample_id=sample_id,
        subject_id=subject_id,
        day_id=day_id,
        event_id=event_id,
        target=target,
        complete_mask=complete_mask,
        ppg_normwear=embeddings["ppg"],
        acc_normwear=embeddings["acc"],
        gsr_normwear=embeddings["gsr"],
    )
    for modality in ("ppg", "acc", "gsr"):
        values = embeddings[modality][complete_mask]
        if values.shape[1] != 768:
            raise ValueError(f"{modality} NormWear embedding dim is {values.shape[1]}, expected 768")
        if not np.isfinite(values).all():
            raise ValueError(f"{modality} NormWear embeddings contain non-finite values")
        if float(np.std(values)) <= 1e-8:
            raise ValueError(f"{modality} NormWear embeddings are near-constant")
    return dataset, manifest


def _load_or_extract_normwear_embeddings(
    args: argparse.Namespace,
    sample_id: np.ndarray,
    complete_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = args.out_root / "embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings: dict[str, np.ndarray] = {}
    manifest: dict[str, Any] = {
        "encoder": "NormWearModel",
        "weight_path": str(args.wear_fm_root / "weights/normwear_pretrain_ckpt.pth"),
        "source_root": str(args.wear_fm_root / "staged_inputs"),
        "complete_mask_count": int(complete_mask.sum()),
        "modalities": {},
    }
    model = None
    device = None
    for modality in ("ppg", "acc", "gsr"):
        emb_path = cache_dir / f"{modality}_normwear_768d.npy"
        mask_path = cache_dir / f"{modality}_normwear_valid_mask.npy"
        if args.skip_existing_cache and emb_path.is_file() and mask_path.is_file():
            emb = np.load(emb_path).astype(np.float32)
            mask = np.load(mask_path).astype(bool)
            if emb.shape != (len(sample_id), 768):
                raise ValueError(f"invalid cached {modality} shape: {emb.shape}")
            if not np.array_equal(mask, complete_mask):
                raise ValueError(f"cached {modality} mask does not equal wear_complete_mask")
            embeddings[modality] = emb
            manifest["modalities"][modality] = _embedding_manifest_row(modality, emb_path, mask_path, emb, mask, cached=True)
            continue
        if model is None or device is None:
            model, device = _normwear_model(args.wear_fm_root, args.device)
        emb, mask, source_shape = _extract_normwear_modality(args, modality, sample_id, complete_mask, model, device)
        np.save(emb_path, emb)
        np.save(mask_path, mask)
        embeddings[modality] = emb
        manifest["modalities"][modality] = _embedding_manifest_row(modality, emb_path, mask_path, emb, mask, cached=False)
        manifest["modalities"][modality]["source_shape"] = list(source_shape)
    (args.out_root / "normwear_all_embedding_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return embeddings, manifest


def _normwear_model(wear_fm_root: Path, device_name: str):
    parent = wear_fm_root / "third_party"
    sys.path.insert(0, str(parent))
    if "timm.models.layers" not in sys.modules:
        timm_mod = types.ModuleType("timm")
        models_mod = types.ModuleType("timm.models")
        layers_mod = types.ModuleType("timm.models.layers")

        def to_2tuple(value: Any) -> tuple[Any, Any]:
            if isinstance(value, tuple):
                return value
            return (value, value)

        layers_mod.to_2tuple = to_2tuple  # type: ignore[attr-defined]
        models_mod.layers = layers_mod  # type: ignore[attr-defined]
        timm_mod.models = models_mod  # type: ignore[attr-defined]
        sys.modules["timm"] = timm_mod
        sys.modules["timm.models"] = models_mod
        sys.modules["timm.models.layers"] = layers_mod
    from NormWear.main_model import NormWearModel

    device = torch.device(device_name)
    model = NormWearModel(weight_path=str(wear_fm_root / "weights/normwear_pretrain_ckpt.pth"), optimized_cwt=True).to(device)
    return model.eval(), device


def _extract_normwear_modality(
    args: argparse.Namespace,
    modality: str,
    sample_id: np.ndarray,
    complete_mask: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    settings = MODALITY_SETTINGS[modality]
    path = args.wear_fm_root / "staged_inputs" / str(settings["path"])
    with np.load(path, allow_pickle=True) as loaded:
        ids = loaded["sample_id"].astype(str)
        if not np.array_equal(ids, sample_id):
            raise ValueError(f"{modality} staged sample_id does not match canonical index")
        source = loaded[str(settings["key"])].astype(np.float32)
        mask = loaded["valid_mask"].astype(bool)
        rate = float(np.asarray(loaded["sample_rate_hz"]).reshape(-1)[0])
    expected_rate = float(settings["rate"])
    if abs(rate - expected_rate) > 1e-6:
        raise ValueError(f"{modality} rate {rate} does not match expected {expected_rate}")
    if not np.array_equal(mask, complete_mask):
        raise ValueError(f"{modality} staged valid_mask does not equal wear_complete_mask")
    out = np.zeros((len(sample_id), 768), dtype=np.float32)
    valid = np.zeros((len(sample_id),), dtype=bool)
    indices = np.flatnonzero(mask)
    batch_size = int(getattr(args, f"{modality}_normwear_batch_size"))
    started = time.time()
    for batch_no, batch in enumerate(_iter_batches(indices, batch_size), start=1):
        if args.limit_embedding_batches and batch_no > args.limit_embedding_batches:
            break
        emb = _normwear_embed_batch(model, source[batch], sampling_rate=rate, device=device, axis_pool=str(settings["axis_pool"]))
        out[batch] = emb
        valid[batch] = True
        done = int(valid.sum())
        if args.progress_interval and (done % args.progress_interval < batch_size or done == len(indices)):
            print(f"{modality}_normwear: embedded {done}/{indices.size} in {time.time() - started:.1f}s", flush=True)
    if args.limit_embedding_batches:
        missing = int((complete_mask & ~valid).sum())
        if missing:
            print(f"{modality}_normwear: limit_embedding_batches left {missing} complete-mask rows uncached", flush=True)
    if not np.array_equal(valid, complete_mask) and not args.limit_embedding_batches:
        raise ValueError(f"{modality} NormWear cache incomplete")
    return out, valid, tuple(source.shape)


def _normwear_embed_batch(
    model: torch.nn.Module,
    values: np.ndarray,
    *,
    sampling_rate: float,
    device: torch.device,
    axis_pool: str,
) -> np.ndarray:
    x = values.astype(np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    with torch.inference_mode():
        output = model.get_embedding(x, sampling_rate=sampling_rate, device=device)
        pooled = output.mean(dim=2)
        if pooled.shape[1] == 1:
            emb = pooled[:, 0, :]
        elif axis_pool == "mean":
            emb = pooled.mean(dim=1)
        else:
            emb = pooled.mean(dim=1)
    return emb.detach().cpu().numpy().astype(np.float32)


def _embedding_manifest_row(
    modality: str,
    emb_path: Path,
    mask_path: Path,
    emb: np.ndarray,
    mask: np.ndarray,
    *,
    cached: bool,
) -> dict[str, Any]:
    values = emb[mask]
    return {
        "modality": modality,
        "embedding_path": str(emb_path),
        "mask_path": str(mask_path),
        "shape": list(emb.shape),
        "valid_count": int(mask.sum()),
        "embedding_dim": int(emb.shape[1]),
        "cached": bool(cached),
        "finite": bool(np.isfinite(values).all()) if values.size else False,
        "std": float(np.std(values)) if values.size else 0.0,
        "axis_pool": str(MODALITY_SETTINGS[modality]["axis_pool"]),
        "sampling_rate_hz": float(MODALITY_SETTINGS[modality]["rate"]),
    }


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
        for name, indices in BASE._eval_splits(split).items()
    }
    result = {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "source": str(args.out_root / "normwear_all_embedding_manifest.json"),
        "target_label": args.target_label,
        "mask_count": int(dataset.complete_mask.sum()),
        "split_counts": {name: int(len(indices)) for name, indices in BASE._eval_splits(split).items()},
        "components": list(ROUTE_COMPONENTS[route]),
        "train": BASE._metric_aliases(evaluate_regression_with_centered(dataset.target[split["train"]], predictions["train"], dataset.subject_id[split["train"]])),
        "val": BASE._metric_aliases(evaluate_regression_with_centered(dataset.target[split["val"]], predictions["val"], dataset.subject_id[split["val"]])),
        "test": BASE._metric_aliases(evaluate_regression_with_centered(dataset.target[split["test"]], predictions["test"], dataset.subject_id[split["test"]])),
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
    BASE._write_predictions_table(run_dir / "test_predictions.csv", dataset, split["test"], predictions["test"])
    torch.save({"state_dict": audit.pop("_best_state"), "route": route, "protocol": protocol, "seed": int(seed)}, run_dir / "best_checkpoint.pt")
    (run_dir / "val_history.csv").write_text(BASE._history_csv(audit["history"]), encoding="utf-8")
    result["prediction_path"] = str(pred_npz)
    result["checkpoint_path"] = str(run_dir / "best_checkpoint.pt")
    result["test_predictions_csv"] = str(run_dir / "test_predictions.csv")
    result["metrics_path"] = str(run_dir / "metrics.json")
    result["config_path"] = str(run_dir / "config.json")
    (run_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(_config_snapshot(args, protocol, route, seed, dataset), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _fit_route(args: argparse.Namespace, dataset: Dataset, split: dict[str, np.ndarray], *, route: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    BASE._seed_everything(seed)
    train = split["train"]
    components = ROUTE_COMPONENTS[route]
    arrays = _component_arrays(dataset)
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
    module = BASE.W3FMReplacementModel(components, input_dims, hidden_dim=args.hidden_dim, dropout=args.dropout).to(torch.device(args.device))

    def inputs(indices: np.ndarray) -> dict[str, np.ndarray]:
        return {
            component: ((arrays[component][indices] - means[component]) / stds[component]).astype(np.float32)
            for component in components
        }

    audit = BASE._train_loop(args, module, train, split["val"], seed, inputs, dataset.target, dataset.subject_id, y_mean, y_std)
    audit["components"] = list(components)
    audit["input_dims"] = input_dims
    return {"module": module, "components": components, "means": means, "stds": stds, "y_mean": y_mean, "y_std": y_std}, audit


def _predict_route(model: dict[str, Any], dataset: Dataset, *, indices: np.ndarray, device: str, eval_batch_size: int) -> np.ndarray:
    module = model["module"]
    module.eval()
    arrays = _component_arrays(dataset)
    values: list[np.ndarray] = []
    dev = torch.device(device)
    with torch.no_grad():
        for start in range(0, len(indices), int(eval_batch_size)):
            batch = indices[start : start + int(eval_batch_size)]
            x = {
                component: ((arrays[component][batch] - model["means"][component]) / model["stds"][component]).astype(np.float32)
                for component in model["components"]
            }
            pred = module(BASE._to_torch(x, dev)).detach().cpu().numpy()
            values.append((pred * float(model["y_std"]) + float(model["y_mean"])).astype(np.float32))
    return np.concatenate(values) if values else np.zeros((0,), dtype=np.float32)


def _component_arrays(dataset: Dataset) -> dict[str, np.ndarray]:
    return {
        "ppg_normwear": dataset.ppg_normwear,
        "acc_normwear": dataset.acc_normwear,
        "gsr_normwear": dataset.gsr_normwear,
    }


def _load_reference_results(args: argparse.Namespace, protocols: tuple[str, ...], seeds: list[int]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for protocol in protocols:
        for seed in seeds:
            paths = [
                args.phase2_root / "runs" / protocol / route / f"seed_{seed}" / "metrics.json"
                for route in ("Wphysio", "Wdeep", "Wmoment_frozen", "W3FM_frozen")
            ]
            paths.append(args.internal_root / "runs" / protocol / "W3FM_no_acc" / f"seed_{seed}" / "metrics.json")
            paths.append(args.acc_replacement_root / "runs" / protocol / "W3FM_acc_handcrafted" / f"seed_{seed}" / "metrics.json")
            for path in paths:
                row = BASE._maybe_metric(path)
                if row is not None:
                    refs.append(row)
    return refs


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_key = {(row["protocol"], row["route"], int(row["seed"])): row for row in rows}
    for protocol in sorted({row["protocol"] for row in rows}):
        for route in REFERENCE_ORDER:
            route_rows = [row for row in rows if row["protocol"] == protocol and row["route"] == route]
            if not route_rows:
                continue
            out.append(_summary_row(protocol, route, route_rows, by_key))
    return out


def _summary_row(
    protocol: str,
    route: str,
    route_rows: list[dict[str, Any]],
    by_key: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    deltas_w3fm = [_delta(row, by_key.get((protocol, "W3FM_frozen", int(row["seed"])))) for row in route_rows]
    deltas_wdeep = [_delta(row, by_key.get((protocol, "Wdeep", int(row["seed"])))) for row in route_rows]
    deltas_no_acc = [_delta(row, by_key.get((protocol, "W3FM_no_acc", int(row["seed"])))) for row in route_rows]
    return {
        "protocol": protocol,
        "route": route,
        "seed_count": int(len(route_rows)),
        "test_rmse_mean": BASE._mean_metric(route_rows, "rmse"),
        "test_rmse_std": BASE._std_metric(route_rows, "rmse"),
        "test_raw_r_mean": BASE._mean_metric(route_rows, "raw_r"),
        "test_raw_r_std": BASE._std_metric(route_rows, "raw_r"),
        "test_centered_r_mean": BASE._mean_metric(route_rows, "within_subject_centered_r"),
        "test_centered_r_std": BASE._std_metric(route_rows, "within_subject_centered_r"),
        "delta_raw_r_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "raw_r"),
        "delta_rmse_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "rmse"),
        "delta_centered_r_vs_W3FM_frozen_mean": _mean_delta(deltas_w3fm, "within_subject_centered_r"),
        "delta_raw_r_vs_Wdeep_mean": _mean_delta(deltas_wdeep, "raw_r"),
        "delta_rmse_vs_Wdeep_mean": _mean_delta(deltas_wdeep, "rmse"),
        "delta_centered_r_vs_Wdeep_mean": _mean_delta(deltas_wdeep, "within_subject_centered_r"),
        "delta_raw_r_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "raw_r"),
        "delta_rmse_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "rmse"),
        "delta_centered_r_vs_W3FM_no_acc_mean": _mean_delta(deltas_no_acc, "within_subject_centered_r"),
        "best_epoch_mean": _mean_audit(route_rows, "best_epoch"),
        "trainable_params_mean": _mean_audit(route_rows, "trainable_params"),
    }


def _delta(row: dict[str, Any], base: dict[str, Any] | None) -> dict[str, float | None]:
    if base is None:
        return {"raw_r": None, "rmse": None, "within_subject_centered_r": None}
    return {
        "raw_r": float(row["test"]["raw_r"]) - float(base["test"]["raw_r"]),
        "rmse": float(row["test"]["rmse"]) - float(base["test"]["rmse"]),
        "within_subject_centered_r": float(row["test"]["within_subject_centered_r"]) - float(base["test"]["within_subject_centered_r"]),
    }


def _mean_delta(rows: list[dict[str, float | None]], metric: str) -> float | None:
    values = [row[metric] for row in rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def _mean_audit(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(row["train_audit"][field]) for row in rows if "train_audit" in row and field in row["train_audit"]]
    return float(np.mean(values)) if values else math.nan


def _write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = [
        "# W3FM NormWear-All Screen",
        "",
        f"- diagnostic_boundary: `{output['diagnostic_boundary']}`",
        f"- run_count: `{output['run_count']}`",
        f"- reference_count: `{output['reference_count']}`",
        f"- mask_count: `{output['mask_count']}`",
        "",
        "## Route Summary",
        "",
        "| protocol | route | seeds | RMSE | raw r | centered r | d raw r vs W3FM | d RMSE vs W3FM | d centered r vs W3FM | d raw r vs Wdeep | d RMSE vs Wdeep | d centered r vs Wdeep | d raw r vs no-ACC | d RMSE vs no-ACC | d centered r vs no-ACC | best epoch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in output["summary"]:
        lines.append(
            f"| {row['protocol']} | {row['route']} | {row['seed_count']} | "
            f"{BASE._fmt(row['test_rmse_mean'])} +/- {BASE._fmt(row['test_rmse_std'])} | "
            f"{BASE._fmt(row['test_raw_r_mean'])} +/- {BASE._fmt(row['test_raw_r_std'])} | "
            f"{BASE._fmt(row['test_centered_r_mean'])} +/- {BASE._fmt(row['test_centered_r_std'])} | "
            f"{BASE._fmt(row['delta_raw_r_vs_W3FM_frozen_mean'])} | "
            f"{BASE._fmt(row['delta_rmse_vs_W3FM_frozen_mean'])} | "
            f"{BASE._fmt(row['delta_centered_r_vs_W3FM_frozen_mean'])} | "
            f"{BASE._fmt(row['delta_raw_r_vs_Wdeep_mean'])} | "
            f"{BASE._fmt(row['delta_rmse_vs_Wdeep_mean'])} | "
            f"{BASE._fmt(row['delta_centered_r_vs_Wdeep_mean'])} | "
            f"{BASE._fmt(row['delta_raw_r_vs_W3FM_no_acc_mean'])} | "
            f"{BASE._fmt(row['delta_rmse_vs_W3FM_no_acc_mean'])} | "
            f"{BASE._fmt(row['delta_centered_r_vs_W3FM_no_acc_mean'])} | "
            f"{BASE._fmt(row['best_epoch_mean'])} |"
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
            f"| {row['protocol']} | {row['route']} | {row['seed']} | {BASE._fmt(test['rmse'])} | "
            f"{BASE._fmt(test['mae'])} | {BASE._fmt(test['raw_r'])} | "
            f"{BASE._fmt(test['within_subject_centered_r'])} | {audit['best_epoch']} | {audit['trainable_params']} |"
        )
    lines.extend(
        [
            "",
            "## Embedding Manifest",
            "",
            "All three submodalities use the same frozen NormWear checkpoint. ACC returns one embedding per axis, then uses mean axis pooling to keep one ACC token.",
            "",
            "```json",
            json.dumps(output["embedding_manifest"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_preflight(dataset: Dataset, embedding_manifest: dict[str, Any], path: Path) -> None:
    payload = {
        "row_count": int(dataset.sample_id.shape[0]),
        "wear_complete_count": int(dataset.complete_mask.sum()),
        "target_finite": bool(np.isfinite(dataset.target).all()),
        "ppg_normwear_shape": list(dataset.ppg_normwear.shape),
        "acc_normwear_shape": list(dataset.acc_normwear.shape),
        "gsr_normwear_shape": list(dataset.gsr_normwear.shape),
        "all_embeddings_finite": bool(
            np.isfinite(dataset.ppg_normwear[dataset.complete_mask]).all()
            and np.isfinite(dataset.acc_normwear[dataset.complete_mask]).all()
            and np.isfinite(dataset.gsr_normwear[dataset.complete_mask]).all()
        ),
        "embedding_manifest": embedding_manifest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    if not summary:
        path.write_text("", encoding="utf-8")
        return
    fields = list(summary[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def _config_snapshot(args: argparse.Namespace, protocol: str, route: str, seed: int, dataset: Dataset) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "route": route,
        "seed": int(seed),
        "components": list(ROUTE_COMPONENTS[route]),
        "component_dims": {name: 768 for name in ROUTE_COMPONENTS[route]},
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "dropout": float(args.dropout),
        "hidden_dim": int(args.hidden_dim),
        "patience": int(args.patience),
        "selection_metric": str(args.selection_metric),
        "train_supervision": "pretrain_plus_finetune_train_val_early_stop_test_once",
        "encoder_policy": "NormWear frozen checkpoint for ppg, acc, and gsr; ACC axis embeddings mean-pooled",
        "mask_policy": "same wear_complete_mask intersected with split indices",
        "diagnostic_boundary": "post_phase3_normwear_all_diagnostic_not_a_formal_promotion_gate",
        "mask_count": int(dataset.complete_mask.sum()),
    }


def _validate_routes(routes: tuple[str, ...]) -> None:
    unknown = set(routes) - set(DEFAULT_ROUTES)
    if unknown:
        raise ValueError(f"unsupported routes: {sorted(unknown)}")


def _iter_batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), int(batch_size)):
        yield indices[start : start + int(batch_size)]


if __name__ == "__main__":
    raise SystemExit(main())
