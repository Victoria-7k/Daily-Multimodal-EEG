#!/usr/bin/env python
"""Phase 1 cache builder for wear-only external foundation models.

This script is deliberately label-free. It reads the canonical window index and
wear source metadata only, builds fixed-row-order model inputs, then extracts
frozen embeddings from PaPaGei-S, HARNet10, and NormWear.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.alignment.time_utils import parse_absolute_time
from daily_multimodal.embeddings.wear_real import _read_wear_series, _resample_series
from daily_multimodal.training.wear_moment_matrix import (
    DEFAULT_INDEX_PATH,
    DEFAULT_WEAR_META_NPZ,
    DEFAULT_WEAR_RAW_ROOT,
    _load_jsonl,
    _load_wear_meta,
    _parse_source_wear_file,
    _relocate_wear_path,
)

PPG_RATE_HZ = 125
ACC_RATE_HZ = 30
GSR_RATE_HZ = 40
WINDOW_SECONDS = 10
EXPECTED = {
    "papagei_s.pt": 23339216,
    "mtl_best.mdl": 42014098,
    "normwear_pretrain_ckpt.pth": 544579503,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _zscore_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean(axis=1, keepdims=True)
    std = arr.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((arr - mean) / std).astype(np.float32)


def _check_weight(path: Path, expected_size: int) -> dict[str, Any]:
    status = {"path": str(path), "exists": path.is_file(), "expected_size": int(expected_size)}
    if not path.is_file():
        status["ok"] = False
        return status
    size = path.stat().st_size
    status["size"] = int(size)
    status["ok"] = bool(size == expected_size)
    if size == expected_size:
        status["sha256"] = _sha256(path)
    return status


def _ensure_unpacked(base: Path) -> dict[str, str]:
    """Assumes zip files were already downloaded by the Phase 1 download script."""
    import shutil
    import zipfile

    third_party = base / "third_party"
    targets = {
        "papagei": (third_party / "papagei-main.zip", third_party / "papagei-foundation-model", "papagei-foundation-model-main"),
        "normwear": (third_party / "normwear-main.zip", third_party / "NormWear", "NormWear-main"),
    }
    out: dict[str, str] = {}
    for name, (zip_path, target, inner_name) in targets.items():
        if not target.is_dir():
            if not zip_path.is_file():
                raise FileNotFoundError(f"missing {zip_path}")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(third_party)
            extracted = third_party / inner_name
            if target.exists():
                shutil.rmtree(target)
            extracted.rename(target)
        out[name] = str(target)
    ssl = third_party / "ssl-wearables"
    if not (ssl / "hubconf.py").is_file() or not (ssl / "sslearning/models/accNet.py").is_file():
        raise FileNotFoundError("missing targeted ssl-wearables source files")
    out["ssl_wearables"] = str(ssl)
    return out


def stage_inputs(args: argparse.Namespace) -> dict[str, Any]:
    base = Path(args.base)
    staged = base / "staged_inputs"
    staged.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(Path(args.index_path))
    meta = _load_wear_meta(Path(args.wear_meta_npz))
    sample_id = np.asarray([str(row["sample_id"]) for row in rows], dtype=object)
    if not np.array_equal(meta["sample_id"][: len(rows)].astype(str), sample_id.astype(str)):
        raise ValueError("wear meta sample_id order does not match canonical index")

    complete_mask_path = staged / "wear_complete_mask.npy"
    if complete_mask_path.is_file():
        complete_mask = np.load(complete_mask_path).astype(bool)
    else:
        complete_mask = meta["wear_mask"][: len(rows)].astype(bool)

    ppg = np.zeros((len(rows), PPG_RATE_HZ * WINDOW_SECONDS), dtype=np.float32)
    acc = np.zeros((len(rows), 3, ACC_RATE_HZ * WINDOW_SECONDS), dtype=np.float32)
    gsr = np.zeros((len(rows), 1, GSR_RATE_HZ * WINDOW_SECONDS), dtype=np.float32)
    ppg_mask = np.zeros((len(rows),), dtype=bool)
    acc_mask = np.zeros((len(rows),), dtype=bool)
    gsr_mask = np.zeros((len(rows),), dtype=bool)
    failures: list[dict[str, Any]] = []

    raw_root = Path(args.wear_raw_root)
    started = time.time()
    for index in range(len(rows)):
        if not complete_mask[index]:
            continue
        try:
            start = parse_absolute_time(str(meta["window_start_time"][index]))
            end = parse_absolute_time(str(meta["window_end_time"][index]))
            duration = (end - start).total_seconds()
            if duration <= 0:
                raise ValueError("non-positive window duration")
            sources = _parse_source_wear_file(meta["source_wear_file"][index])
            for modality in ("ppg", "acc", "gsr"):
                if not sources.get(modality):
                    raise FileNotFoundError(f"missing {modality} source in metadata")
            ppg_series = _read_wear_series(_relocate_wear_path(sources["ppg"], raw_root), modality="ppg", window_start=start, window_end=end)
            acc_series = _read_wear_series(_relocate_wear_path(sources["acc"], raw_root), modality="acc", window_start=start, window_end=end)
            gsr_series = _read_wear_series(_relocate_wear_path(sources["gsr"], raw_root), modality="gsr", window_start=start, window_end=end)
            if ppg_series.values.shape[0] > 0:
                ppg[index] = _resample_series(ppg_series, duration_seconds=duration, target_rate_hz=PPG_RATE_HZ)[:, 0]
                ppg_mask[index] = True
            if acc_series.values.shape[0] > 0:
                acc[index] = _resample_series(acc_series, duration_seconds=duration, target_rate_hz=ACC_RATE_HZ).T
                acc_mask[index] = True
            if gsr_series.values.shape[0] > 0:
                raw_gsr = _resample_series(gsr_series, duration_seconds=duration, target_rate_hz=GSR_RATE_HZ)[:, 0]
                gsr[index, 0] = np.log1p(np.clip(raw_gsr, a_min=0.0, a_max=None)).astype(np.float32)
                gsr_mask[index] = True
        except Exception as exc:
            failures.append({"index": int(index), "sample_id": str(sample_id[index]), "error": repr(exc)})
        if args.progress_interval and (index + 1) % args.progress_interval == 0:
            print(f"staged {index + 1}/{len(rows)} windows in {time.time() - started:.1f}s", flush=True)

    np.savez_compressed(staged / "ppg_10s.npz", sample_id=sample_id, ppg=ppg, valid_mask=ppg_mask, sample_rate_hz=PPG_RATE_HZ)
    np.savez_compressed(staged / "acc_10s.npz", sample_id=sample_id, acc=acc, valid_mask=acc_mask, sample_rate_hz=ACC_RATE_HZ)
    np.savez_compressed(staged / "gsr_10s.npz", sample_id=sample_id, gsr=gsr, valid_mask=gsr_mask, sample_rate_hz=GSR_RATE_HZ)
    summary = {
        "stage": "stage_inputs",
        "row_count": int(len(rows)),
        "complete_mask_count": int(complete_mask.sum()),
        "ppg_valid_count": int(ppg_mask.sum()),
        "acc_valid_count": int(acc_mask.sum()),
        "gsr_valid_count": int(gsr_mask.sum()),
        "all_valid_count": int((ppg_mask & acc_mask & gsr_mask).sum()),
        "failures": failures[:50],
        "failure_count": int(len(failures)),
        "seconds": float(time.time() - started),
        "transforms": {
            "ppg": "resample_to_125hz_then_per_segment_zscore_inside_papagei_extractor",
            "acc": "resample_to_30hz_no_axis_correction",
            "gsr": "resample_to_40hz_log1p_nonnegative_no_split_scaling",
        },
    }
    _write_json(base / "logs/staged_input_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_staged(base: Path) -> dict[str, Any]:
    staged = base / "staged_inputs"
    with np.load(staged / "ppg_10s.npz", allow_pickle=True) as ppg, np.load(staged / "acc_10s.npz", allow_pickle=True) as acc, np.load(staged / "gsr_10s.npz", allow_pickle=True) as gsr:
        sample_id = ppg["sample_id"].astype(str)
        if not np.array_equal(sample_id, acc["sample_id"].astype(str)) or not np.array_equal(sample_id, gsr["sample_id"].astype(str)):
            raise ValueError("staged input sample_id arrays do not match")
        return {
            "sample_id": sample_id,
            "ppg": ppg["ppg"].astype(np.float32),
            "ppg_mask": ppg["valid_mask"].astype(bool),
            "acc": acc["acc"].astype(np.float32),
            "acc_mask": acc["valid_mask"].astype(bool),
            "gsr": gsr["gsr"].astype(np.float32),
            "gsr_mask": gsr["valid_mask"].astype(bool),
        }


def _iter_batches(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, int(indices.shape[0]), batch_size):
        yield indices[start : start + batch_size]


def _torch_device(name: str):
    import torch

    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _papagei_model(base: Path, device_name: str):
    import torch

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
    device = _torch_device(device_name)
    return model.to(device).eval(), device


def _harnet_model(base: Path, device_name: str):
    repo = base / "third_party/ssl-wearables"
    sys.path.insert(0, str(repo))
    import hubconf

    device = _torch_device(device_name)
    model = hubconf.harnet10(pretrained=True, my_device=str(device), class_num=5)
    return model.to(device).eval(), device


def _normwear_model(base: Path, device_name: str):
    import types

    parent = base / "third_party"
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

    device = _torch_device(device_name)
    model = NormWearModel(weight_path=str(base / "weights/normwear_pretrain_ckpt.pth"), optimized_cwt=True).to(device)
    return model.eval(), device


def extract_one_model(args: argparse.Namespace, model_name: str, *, smoke: bool) -> dict[str, Any]:
    import torch

    base = Path(args.base)
    embeddings_dir = base / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    data = _load_staged(base)
    if model_name == "ppg":
        out_path = embeddings_dir / "ppg_papagei_s_512d.npy"
        mask_path = embeddings_dir / "ppg_papagei_s_valid_mask.npy"
        source = data["ppg"]
        valid = data["ppg_mask"]
        model, device = _papagei_model(base, args.device)
        batch_size = args.ppg_batch_size

        def run(batch_indices: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(_zscore_rows(source[batch_indices])).unsqueeze(1).to(device)
            with torch.inference_mode():
                y = model(x)
                emb = y[0] if isinstance(y, tuple) else y
            return emb.detach().cpu().numpy().astype(np.float32)

    elif model_name == "acc":
        out_path = embeddings_dir / "acc_harnet10.npy"
        mask_path = embeddings_dir / "acc_harnet10_valid_mask.npy"
        source = data["acc"]
        valid = data["acc_mask"]
        model, device = _harnet_model(base, args.device)
        batch_size = args.acc_batch_size

        def run(batch_indices: np.ndarray) -> np.ndarray:
            x = torch.from_numpy(source[batch_indices]).to(device)
            with torch.inference_mode():
                feats = model.feature_extractor(x)
                emb = feats.reshape(feats.shape[0], -1)
            return emb.detach().cpu().numpy().astype(np.float32)

    elif model_name == "gsr":
        out_path = embeddings_dir / "gsr_normwear_768d.npy"
        mask_path = embeddings_dir / "gsr_normwear_valid_mask.npy"
        source = data["gsr"]
        valid = data["gsr_mask"]
        model, device = _normwear_model(base, args.device)
        batch_size = args.gsr_batch_size

        def run(batch_indices: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                out = model.get_embedding(source[batch_indices], sampling_rate=GSR_RATE_HZ, device=device)
                emb = out.mean(dim=2).squeeze(1)
            return emb.detach().cpu().numpy().astype(np.float32)

    else:
        raise ValueError(f"unknown model: {model_name}")

    indices = np.flatnonzero(valid)
    if smoke:
        indices = indices[: args.smoke_count]
    if indices.size == 0:
        raise ValueError(f"no valid staged inputs for {model_name}")
    first = run(indices[: min(batch_size, indices.size)])
    if first.ndim != 2 or first.shape[0] == 0:
        raise ValueError(f"{model_name} produced invalid shape {first.shape}")
    if not np.isfinite(first).all():
        raise ValueError(f"{model_name} produced NaN/Inf in smoke")
    if float(np.std(first)) <= 1e-8:
        raise ValueError(f"{model_name} produced near-constant smoke embeddings")
    if smoke:
        out = np.zeros((data["sample_id"].shape[0], first.shape[1]), dtype=np.float32)
        out[indices[: first.shape[0]]] = first
        valid_out = np.zeros((data["sample_id"].shape[0],), dtype=bool)
        valid_out[indices[: first.shape[0]]] = True
    else:
        out = np.zeros((data["sample_id"].shape[0], first.shape[1]), dtype=np.float32)
        valid_out = np.zeros((data["sample_id"].shape[0],), dtype=bool)
        cursor = 0
        for batch_indices in _iter_batches(indices, batch_size):
            emb = first if cursor == 0 and batch_indices.shape[0] == first.shape[0] else run(batch_indices)
            out[batch_indices] = emb
            valid_out[batch_indices] = True
            cursor += int(batch_indices.shape[0])
            if args.progress_interval and cursor % args.progress_interval < batch_size:
                print(f"{model_name}: embedded {cursor}/{indices.size}", flush=True)
    if not smoke:
        np.save(out_path, out)
        np.save(mask_path, valid_out)
    summary = {
        "model": model_name,
        "smoke": bool(smoke),
        "output_shape": list(out.shape),
        "valid_count": int(valid_out.sum()),
        "embedding_dim": int(out.shape[1]),
        "finite": bool(np.isfinite(out[valid_out]).all()) if valid_out.any() else False,
        "std": float(np.std(out[valid_out])) if valid_out.any() else 0.0,
        "out_path": str(out_path) if not smoke else None,
        "mask_path": str(mask_path) if not smoke else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_smoke_or_batch(args: argparse.Namespace, *, smoke: bool) -> dict[str, Any]:
    base = Path(args.base)
    _ensure_unpacked(base)
    selected = [item.strip() for item in args.models.split(",") if item.strip()]
    required_weights = {
        "ppg": ("papagei_s.pt", base / "weights/papagei_s.pt", EXPECTED["papagei_s.pt"]),
        "acc": ("mtl_best.mdl", base / "third_party/ssl-wearables/model_check_point/mtl_best.mdl", EXPECTED["mtl_best.mdl"]),
        "gsr": ("normwear_pretrain_ckpt.pth", base / "weights/normwear_pretrain_ckpt.pth", EXPECTED["normwear_pretrain_ckpt.pth"]),
    }
    weight_status = {
        weight_name: _check_weight(path, expected)
        for model_name in selected
        for weight_name, path, expected in [required_weights[model_name]]
    }
    bad = [name for name, status in weight_status.items() if not status["ok"]]
    if bad:
        raise RuntimeError(f"incomplete or missing weights: {bad}; status={weight_status}")
    results = []
    for name in selected:
        results.append(extract_one_model(args, name, smoke=smoke))
    manifest = {
        "stage": "smoke" if smoke else "batch_cache",
        "models": results,
        "weights": weight_status,
        "sources": {
            "papagei": "https://github.com/nokia-bell-labs/papagei-foundation-model + https://zenodo.org/records/13983110",
            "harnet10": "https://github.com/OxWearables/ssl-wearables + https://wearables-files.ndph.ox.ac.uk/files/ssl/mtl_best.mdl",
            "normwear": "https://github.com/Mobile-Sensing-and-UbiComp-Laboratory/NormWear/releases/tag/v1.0.0-alpha",
        },
    }
    _write_json(base / ("logs/embedding_smoke_manifest.json" if smoke else "embeddings/embedding_manifest.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="outputs/wear_fm")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--wear-meta-npz", default=DEFAULT_WEAR_META_NPZ)
    parser.add_argument("--wear-raw-root", default=DEFAULT_WEAR_RAW_ROOT)
    parser.add_argument("--mode", choices=("stage", "smoke", "batch", "all"), default="all")
    parser.add_argument("--models", default="ppg,acc,gsr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-count", type=int, default=8)
    parser.add_argument("--ppg-batch-size", type=int, default=256)
    parser.add_argument("--acc-batch-size", type=int, default=512)
    parser.add_argument("--gsr-batch-size", type=int, default=16)
    parser.add_argument("--progress-interval", type=int, default=2000)
    args = parser.parse_args()

    if args.mode in {"stage", "all"}:
        stage_inputs(args)
    if args.mode in {"smoke", "all"}:
        run_smoke_or_batch(args, smoke=True)
    if args.mode in {"batch", "all"}:
        run_smoke_or_batch(args, smoke=False)


if __name__ == "__main__":
    main()
