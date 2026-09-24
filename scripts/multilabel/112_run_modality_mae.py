#!/usr/bin/env python3
"""Train one label-free EEG or Wear MAE and export frozen 256D window tokens.

The script uses ``pretrain + finetune`` only for MAE optimization, selects the
checkpoint on ``val``, and exports tokens without inspecting the 11 labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.training.eeg_encoder_matrix import load_split_protocols
from daily_multimodal.training.modality_mae import (
    MAERuntime, eeg_to_patches, export_eeg_embeddings, export_wear_embeddings,
    train_eeg_mae, train_wear_mae, wear_to_patches,
)

ALIGNED_ROOT = Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned")
DATA_ROOT = Path("/vePFS-0x0d/DailyEEG/processed_cadt_addtime_new")
SPLITS_ROOT = Path("/vePFS-0x0d/DailyEEG/splits_new")


def _json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_sample_ids(index_path: Path) -> np.ndarray:
    with index_path.open(encoding="utf-8") as handle:
        values = [json.loads(line)["sample_id"] for line in handle]
    return np.asarray(values, dtype=str)


def _load_wear(staged_root: Path, expected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(staged_root / "ppg_10s.npz", allow_pickle=True) as ppg, np.load(staged_root / "gsr_10s.npz", allow_pickle=True) as eda, np.load(staged_root / "acc_10s.npz", allow_pickle=True) as acc:
        if not (np.array_equal(ppg["sample_id"].astype(str), expected_ids) and np.array_equal(eda["sample_id"].astype(str), expected_ids) and np.array_equal(acc["sample_id"].astype(str), expected_ids)):
            raise ValueError("staged wear sample_id order differs from canonical index")
        valid = ppg["valid_mask"].astype(bool) & eda["valid_mask"].astype(bool) & acc["valid_mask"].astype(bool)
        return ppg["ppg"].astype(np.float32), eda["gsr"].astype(np.float32), acc["acc"].astype(np.float32), valid


def _cap(values: np.ndarray, count: int) -> np.ndarray:
    return values if count <= 0 else values[: min(count, len(values))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", choices=("eeg", "wear"), required=True)
    parser.add_argument("--protocol", choices=("cross_day", "within_subject_day"), required=True)
    parser.add_argument("--aligned-root", type=Path, default=ALIGNED_ROOT)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--splits-root", type=Path, default=SPLITS_ROOT)
    parser.add_argument("--wear-staged-root", type=Path, default=ALIGNED_ROOT / "outputs/wear_fm/staged_inputs")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--mask-ratio", type=float)
    parser.add_argument("--seed", type=int, default=240800)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-per-split", type=int, default=0, help="Cap train/val rows only; 0 uses each full split.")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0.0 < (args.mask_ratio if args.mask_ratio is not None else (0.7 if args.modality == "eeg" else 0.6)) < 1.0:
        raise ValueError("mask ratio must be strictly between zero and one")

    sample_id = _load_sample_ids(args.aligned_root / "index/eeg_aligned_window_index.jsonl")
    splits = load_split_protocols(args.splits_root, (args.protocol,), row_count=len(sample_id))[args.protocol]
    train_idx, val_idx = _cap(splits.train, args.smoke_per_split), _cap(splits.val, args.smoke_per_split)
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else (0.7 if args.modality == "eeg" else 0.6)
    runtime = MAERuntime(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, patience=args.patience, mask_ratio=mask_ratio, seed=args.seed, device=args.device)
    out_dir = args.out_root / args.protocol / f"{args.modality}_seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"route": f"{args.modality}_mae_label_free_v1", "modality": args.modality, "protocol": args.protocol, "supervision_boundary": "unlabeled_reconstruction_pretrain_plus_finetune__validation_reconstruction_selection", "test_labels_read": False, "row_count": len(sample_id), "split_counts": {"pretrain": len(splits.pretrain), "finetune": len(splits.finetune), "train": len(splits.train), "val": len(splits.val), "test": len(splits.test)}, "effective_train_count": len(train_idx), "effective_val_count": len(val_idx), "embedding_seed": args.seed, "runtime": runtime.__dict__}
    if args.modality == "eeg":
        source = np.load(args.data_root / "X.npy", mmap_mode="r")
        if source.shape != (len(sample_id), 2000, 59): raise ValueError(f"unexpected EEG source shape: {source.shape}")
        model, audit = train_eeg_mae(source, train_idx, val_idx, embedding_dim=256, encoder_layers=6, decoder_layers=2, heads=8, runtime=runtime)
        embeddings = export_eeg_embeddings(model, source, batch_size=args.batch_size, device=args.device)
        valid = np.ones(len(sample_id), dtype=bool)
        model_spec = {"patch_seconds": 1, "patch_shape": [59, 200], "embedding_dim": 256, "encoder_layers": 6, "heads": 8, "decoder_layers": 2}
    else:
        ppg, eda, acc, valid = _load_wear(args.wear_staged_root, sample_id)
        train_idx = _cap(splits.train[valid[splits.train]], args.smoke_per_split)
        val_idx = _cap(splits.val[valid[splits.val]], args.smoke_per_split)
        if not len(train_idx) or not len(val_idx): raise ValueError("Wear MAE has no valid train or validation windows")
        manifest["effective_train_count"], manifest["effective_val_count"], manifest["valid_window_count"] = len(train_idx), len(val_idx), int(valid.sum())
        model, audit = train_wear_mae(ppg, eda, acc, train_idx, val_idx, embedding_dim=256, encoder_layers=4, decoder_layers=2, heads=4, runtime=runtime)
        embeddings = export_wear_embeddings(model, ppg, eda, acc, valid, batch_size=args.batch_size, device=args.device)
        model_spec = {"patch_seconds": 1, "ppg_patch_samples": 125, "eda_patch_samples": 40, "acc_patch_samples": [3, 30], "embedding_dim": 256, "encoder_layers": 4, "heads": 4, "decoder_layers": 2}
    manifest["model"] = model_spec; manifest["reconstruction_audit"] = audit
    checkpoint = out_dir / "checkpoint.pt"; torch.save({"state_dict": model.state_dict(), "manifest": manifest}, checkpoint)
    token = out_dir / "window_embeddings.npz"; np.savez_compressed(token, sample_id=sample_id, embedding=embeddings, valid_mask=valid)
    manifest["checkpoint"] = str(checkpoint); manifest["token_path"] = str(token)
    _json(out_dir / "config.json", manifest)
    print(json.dumps({"gate": "pass", "checkpoint": str(checkpoint), "token_path": str(token), "best_val_masked_nmse": audit["best_val_masked_nmse"], "best_epoch": audit["best_epoch"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
