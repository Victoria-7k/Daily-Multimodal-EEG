#!/usr/bin/env python3
"""Train one label-free EEG or Wear MAE and export frozen 256D window tokens.

The script uses ``pretrain + finetune`` only for MAE optimization, selects the
checkpoint on ``val``, and exports tokens without inspecting the 11 labels.
"""

from __future__ import annotations

import argparse
import hashlib
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
    MAERuntime, MAE_TRAINING_VERSION, MAE_PREPROCESSING_VERSION, MAE_FIXED_PREPROCESSING_VERSION,
    export_eeg_embeddings, export_wear_embeddings, representative_indices,
    train_eeg_mae, train_wear_mae, fit_signal_normalization,
    MAE_ROBUST_PREPROCESSING_VERSION, MAE_BALANCED_TRAINING_VERSION,
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
    return representative_indices(values, count)


class JoinedEEGSource:
    """Resolve training-table indices in their own canonical/raw source space."""
    def __init__(self, canonical, raw, rows):
        self.sources = (canonical, raw)
        self.kinds = np.asarray([0 if row['source_kind'] == 'canonical' else 1 for row in rows])
        self.rows = np.asarray([row['source_row'] for row in rows], dtype=np.int64)

    def __getitem__(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        output = np.empty((len(indices), 2000, 59), dtype=self.sources[0].dtype)
        for kind, source in enumerate(self.sources):
            selected = self.kinds[indices] == kind
            output[selected] = source[self.rows[indices[selected]]]
        return output


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
    parser.add_argument("--skip-full-export", action="store_true")
    parser.add_argument("--health-probe-count", type=int, default=256)
    parser.add_argument("--min-relative-variation", type=float, default=1e-3)
    parser.add_argument("--raw-normalization", choices=("per_second", "train_channel", "train_channel_robust"), default="per_second")
    parser.add_argument("--ssl-training-manifest", type=Path)
    parser.add_argument("--extra-eeg-root", type=Path)
    parser.add_argument("--normalization-config", type=Path)
    parser.add_argument("--recovery-checkpoint", action="store_true", help="Save full epoch-boundary state; replay exactly after interruption.")
    parser.add_argument("--retry-attention-math", action="store_true", help="Retry CUDA nonfinite gradients with math SDPA before any optimizer update.")
    args = parser.parse_args()
    if args.retry_attention_math and (args.modality!='eeg' or not args.recovery_checkpoint):
        raise ValueError('attention recovery requires EEG epoch-state diagnostics')
    if args.ssl_training_manifest and (args.modality != 'eeg' or not args.extra_eeg_root or not args.normalization_config or args.raw_normalization != 'train_channel_robust'):
        raise ValueError('joint EEG training requires raw source, frozen robust normalization and EEG modality')
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0.0 < (args.mask_ratio if args.mask_ratio is not None else (0.7 if args.modality == "eeg" else 0.6)) < 1.0:
        raise ValueError("mask ratio must be strictly between zero and one")

    sample_id = _load_sample_ids(args.aligned_root / "index/eeg_aligned_window_index.jsonl")
    splits = load_split_protocols(args.splits_root, (args.protocol,), row_count=len(sample_id))[args.protocol]
    train_idx, val_idx = _cap(splits.train, args.smoke_per_split), _cap(splits.val, args.smoke_per_split)
    mask_ratio = args.mask_ratio if args.mask_ratio is not None else (0.7 if args.modality == "eeg" else 0.6)
    robust = args.raw_normalization == "train_channel_robust"
    runtime = MAERuntime(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, patience=args.patience, mask_ratio=mask_ratio, seed=args.seed, device=args.device, health_probe_count=args.health_probe_count, min_relative_variation=args.min_relative_variation,
                         reconstruction_loss="window_energy_balanced_mse_v1" if robust else "masked_mse", robust_health=robust,
                         eeg_masking="channel" if robust and args.modality == "eeg" else "temporal")
    out_dir = args.out_root / args.protocol / f"{args.modality}_seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "checkpoint.pt").exists():
        raise FileExistsError(f"preserve existing checkpoint; choose a new output root: {out_dir}")
    manifest: dict[str, Any] = {"route": f"{args.modality}_mae_label_free_v1", "modality": args.modality, "protocol": args.protocol, "supervision_boundary": "unlabeled_reconstruction_pretrain_plus_finetune__validation_reconstruction_selection", "test_labels_read": False, "row_count": len(sample_id), "split_counts": {"pretrain": len(splits.pretrain), "finetune": len(splits.finetune), "train": len(splits.train), "val": len(splits.val), "test": len(splits.test)}, "effective_train_count": len(train_idx), "effective_val_count": len(val_idx), "embedding_seed": args.seed, "runtime": runtime.__dict__}
    normalization = None
    def fit_normalization(sources: tuple[np.ndarray, ...]) -> dict[str, Any] | None:
        if args.raw_normalization == "per_second":
            return None
        if np.intersect1d(train_idx, splits.val).size or np.intersect1d(train_idx, splits.test).size:
            raise ValueError("normalization train rows overlap val/test")
        value = fit_signal_normalization(args.modality, sources, train_idx, batch_size=args.batch_size, robust=robust)
        print(json.dumps({"normalization_fit": "train_only", "modality": args.modality, "protocol": args.protocol,
                          "fit_row_count": value["fit_row_count"], "fit_indices_sha256": value["fit_indices_sha256"]}), flush=True)
        return value
    if args.modality == "eeg":
        source = np.load(args.data_root / "X.npy", mmap_mode="r")
        if source.shape != (len(sample_id), 2000, 59): raise ValueError(f"unexpected EEG source shape: {source.shape}")
        if args.normalization_config:
            old = json.loads(args.normalization_config.read_text(encoding='utf-8'))
            normalization = old['raw_normalization']
            expected = hashlib.sha256(np.sort(splits.train).astype('<i8').tobytes()).hexdigest()
            if old['protocol'] != args.protocol or normalization['version'] != MAE_ROBUST_PREPROCESSING_VERSION or normalization['fit_indices_sha256'] != expected:
                raise ValueError('normalization source differs from canonical protocol training rows')
            manifest['normalization_fit_manifest'] = {'source_config': str(args.normalization_config), 'config_sha256': hashlib.sha256(args.normalization_config.read_bytes()).hexdigest(), 'fit_indices_sha256': expected}
        else:
            normalization = fit_normalization((source,))
        training_source = None
        if args.ssl_training_manifest:
            report = json.loads(args.ssl_training_manifest.with_name('filter_report.json').read_text())
            if report['status'] != 'pass' or report['protocol'] != args.protocol or report['raw_holdout_overlap_count_after_filter'] != 0:
                raise ValueError('joint training input gate failed')
            rows = [json.loads(line) for line in args.ssl_training_manifest.open(encoding='utf-8') if line.strip()]
            if any(r['source_kind'] not in ('canonical', 'raw') for r in rows):raise ValueError('unsupported training source')
            canonical_rows = np.asarray([r['source_row'] for r in rows if r['source_kind']=='canonical'])
            if not np.array_equal(np.sort(canonical_rows), np.sort(splits.train)):raise ValueError('canonical training membership changed')
            raw = np.load(args.extra_eeg_root/'X.npy', mmap_mode='r')
            if raw.shape[1:] != (2000,59) or raw.dtype != source.dtype:raise ValueError('raw signal contract differs')
            if any(r['source_row'] < 0 or r['source_row'] >= (len(source) if r['source_kind']=='canonical' else len(raw)) for r in rows):raise ValueError('joint row out of source bounds')
            if len({(r['source_kind'],r['source_row']) for r in rows})!=len(rows):raise ValueError('duplicate training identity')
            training_source = JoinedEEGSource(source, raw, rows)
            train_idx = np.arange(len(rows),dtype=np.int64)
            if args.smoke_per_split:
                groups = [np.flatnonzero(training_source.kinds==kind) for kind in (0,1)]
                counts = [args.smoke_per_split//2,args.smoke_per_split-args.smoke_per_split//2]
                train_idx = np.concatenate([_cap(group,cap) for group,cap in zip(groups,counts)])
            manifest.update(ssl_training_manifest=str(args.ssl_training_manifest), ssl_training_manifest_sha256=hashlib.sha256(args.ssl_training_manifest.read_bytes()).hexdigest(),
                            effective_train_count=len(train_idx), training_source_counts={kind: int((training_source.kinds[train_idx]==i).sum()) for i,kind in enumerate(('canonical','raw'))},
                            input_variant='E_POOL', source_index_space='source_kind_plus_source_row; independent canonical val and export')
        model, audit = train_eeg_mae(source, train_idx, val_idx, embedding_dim=256, encoder_layers=6, decoder_layers=2, heads=8, runtime=runtime, normalization=normalization, train_source=training_source,
                                     recovery_path=out_dir/'recovery.pt' if args.recovery_checkpoint else None,retry_attention_math=args.retry_attention_math)
        valid = np.ones(len(sample_id), dtype=bool)
        model_spec = {"patch_seconds": 1, "patch_shape": [59, 200], "embedding_dim": 256, "encoder_layers": 6, "heads": 8, "decoder_layers": 2}
    else:
        ppg, eda, acc, valid = _load_wear(args.wear_staged_root, sample_id)
        train_idx = _cap(splits.train[valid[splits.train]], args.smoke_per_split)
        val_idx = _cap(splits.val[valid[splits.val]], args.smoke_per_split)
        if not len(train_idx) or not len(val_idx): raise ValueError("Wear MAE has no valid train or validation windows")
        manifest["effective_train_count"], manifest["effective_val_count"], manifest["valid_window_count"] = len(train_idx), len(val_idx), int(valid.sum())
        normalization = fit_normalization((ppg, eda, acc))
        model, audit = train_wear_mae(ppg, eda, acc, train_idx, val_idx, embedding_dim=256, encoder_layers=4, decoder_layers=2, heads=4, runtime=runtime, normalization=normalization)
        model_spec = {"patch_seconds": 1, "ppg_patch_samples": 125, "eda_patch_samples": 40, "acc_patch_samples": [3, 30], "embedding_dim": 256, "encoder_layers": 4, "heads": 4, "decoder_layers": 2}
    manifest["model"] = model_spec; manifest["reconstruction_audit"] = audit
    if args.modality=='eeg':manifest['numerical_gradient_recovery']={'enabled':args.retry_attention_math,'strategy':'same_batch_same_mask_math_SDPA_retry_before_update','rng_after_retry':'discarded_default_forward_stream','retry_count':len(audit.get('attention_retry_records',[]))}
    manifest["training_version"] = MAE_BALANCED_TRAINING_VERSION if robust else MAE_TRAINING_VERSION
    manifest["preprocessing_version"] = normalization["version"] if normalization is not None else MAE_PREPROCESSING_VERSION
    manifest["raw_normalization"] = normalization
    manifest["reconstruction_masking"] = "random_channels_shared_across_10_seconds" if runtime.eeg_masking == "channel" else "random_whole_second_tokens"
    manifest["split_root"] = str(args.splits_root / args.protocol)
    if normalization is not None:
        manifest["route"] = f"{args.modality}_mae_label_free_{args.raw_normalization}_v1"
    manifest["validation_mask"] = "fixed_seed_plus_100003_fixed_count_independent_of_training_rng"
    manifest["smoke_per_split"] = args.smoke_per_split
    if args.smoke_per_split:
        manifest["smoke_indices"] = {"train": train_idx.tolist(), "val": val_idx.tolist()}
    checkpoint = out_dir / "checkpoint.pt"; torch.save({"state_dict": model.state_dict(), "manifest": manifest}, checkpoint)
    manifest["checkpoint"] = str(checkpoint)
    if args.skip_full_export:
        manifest["token_export"] = "skipped_for_smoke"
    else:
        embeddings = export_eeg_embeddings(model, source, batch_size=args.batch_size, device=args.device, normalization=normalization) if args.modality == "eeg" else export_wear_embeddings(model, ppg, eda, acc, valid, batch_size=args.batch_size, device=args.device, normalization=normalization)
        token = out_dir / "window_embeddings.npz"
        np.savez_compressed(token, sample_id=sample_id, embedding=embeddings, valid_mask=valid)
        manifest["token_path"] = str(token)
    _json(out_dir / "config.json", manifest)
    print(json.dumps({"gate": "pass", "checkpoint": str(checkpoint), "token_path": manifest.get("token_path", "skipped_for_smoke"), "best_val_masked_nmse": audit["best_val_masked_nmse"], "best_epoch": audit["best_epoch"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
