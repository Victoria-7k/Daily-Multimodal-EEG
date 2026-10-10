#!/usr/bin/env python3
"""Cache deterministic frozen encoder prefixes; no labels are read here.

The last two transformer blocks remain outside the cache and are subsequently
optimized with event-level supervision. Video cache processing stays on ncc.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_tail(model, blocks: int = 2):
    first = len(model.encoder.layers) - blocks
    tail = torch.nn.TransformerEncoder(copy.deepcopy(model.encoder.layers[first]), blocks,
                                       enable_nested_tensor=False)
    for index in range(blocks):
        tail.layers[index].load_state_dict(model.encoder.layers[first + index].state_dict())
    return tail


def export_batch_size(requested: int, *, match_stage_a: bool, modality: str,
                      initialization: str, checkpoint_payload: dict | None) -> int:
    if requested <= 0:
        raise ValueError("prefix batch size must be positive")
    if not match_stage_a:
        return requested
    if initialization != "pretrained" or modality not in ("eeg", "wear"):
        raise ValueError("Stage-A batch matching requires a pretrained EEG/Wear checkpoint")
    size = (checkpoint_payload or {}).get("manifest", {}).get("runtime", {}).get("batch_size")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Stage-A checkpoint is missing a positive runtime batch_size")
    return size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modality", required=True, choices=("eeg", "wear", "video"))
    parser.add_argument("--initialization", required=True, choices=("pretrained", "random"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--eeg-source", type=Path)
    parser.add_argument("--wear-root", type=Path)
    parser.add_argument("--video-cache", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--match-stage-a-batch-size", action="store_true",
                        help="Use the pretrained EEG/Wear token export batch size to avoid batch-dependent fp32 drift.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--normalization-config", type=Path, help="Training-only raw normalization for a matched random temporal encoder.")
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    if args.initialization == "pretrained" and args.checkpoint is None:
        raise ValueError("pretrained prefix requires the Stage-A checkpoint")
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False) if args.initialization == "pretrained" else None
    requested_batch_size = args.batch_size
    args.batch_size = export_batch_size(requested_batch_size, match_stage_a=args.match_stage_a_batch_size,
                                       modality=args.modality, initialization=args.initialization,
                                       checkpoint_payload=checkpoint_payload)
    if args.match_stage_a_batch_size:
        print(f"prefix_export_batch_size={args.batch_size} requested={requested_batch_size} matched_to_stage_a=True", flush=True)
    normalization = checkpoint_payload.get("manifest", {}).get("raw_normalization") if checkpoint_payload is not None else None
    if args.normalization_config is not None:
        if args.initialization != "random" or args.modality == "video":
            raise ValueError("normalization config is only for random EEG/Wear prefixes")
        config = json.loads(args.normalization_config.read_text())
        normalization = config["raw_normalization"]
        if normalization is None:
            raise ValueError("random normalization config must contain fitted channel statistics")
    if normalization is not None and normalization.get("modality") != args.modality:
        raise ValueError("prefix normalization modality mismatch")
    if checkpoint_payload is not None and checkpoint_payload.get("manifest", {}).get("preprocessing_version") in ("train_channel_zscore_v1", "train_channel_robust_zscore_v1") and normalization is None:
        raise ValueError("train-channel checkpoint is missing normalization")
    here = Path(__file__).resolve().parent
    source_file = here / ("114_run_video_mae.py" if args.modality == "video" else "modality_mae.py")
    provenance = {"initialization": args.initialization, "modality": args.modality,
                  "checkpoint_sha256": file_hash(args.checkpoint) if args.initialization == "pretrained" else None,
                  "source_token_sha256": file_hash(args.token_file),
                  "encoder_source_sha256": file_hash(source_file),
                  "raw_normalization_sha256": hashlib.sha256(json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()).hexdigest() if normalization is not None else None}
    if (args.out_dir / "PREFIX_COMPLETE").exists():
        manifest = json.loads((args.out_dir / "manifest.json").read_text())
        if any(manifest.get(key) != value for key, value in provenance.items()):
            raise ValueError("existing prefix provenance mismatch")
        if args.match_stage_a_batch_size and manifest.get("prefix_export_batch_size") != args.batch_size:
            raise ValueError("existing prefix export batch size does not match Stage-A")
        for filename in ("prefix.npy", "tail.pt", "window_embeddings.npz"):
            if not (args.out_dir / filename).is_file():
                raise FileNotFoundError(args.out_dir / filename)
        print(f"prefix_already_complete={args.out_dir}", flush=True)
        return 0
    seed = 240800 + {"eeg": 0, "wear": 1, "video": 2}[args.modality]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with np.load(args.token_file, allow_pickle=False) as token:
        sample_id = token["sample_id"].astype(str)
        valid = token["valid_mask"].astype(bool)
        expected = token["embedding"].astype(np.float32)
    if len(set(sample_id.tolist())) != len(sample_id) or expected.shape != (len(valid), 256):
        raise ValueError("invalid canonical token file")
    temporal = None
    if args.modality == "video":
        source = load_module("video_mae_prefix_source", here / "114_run_video_mae.py")
        model = source.VideoMaskedAutoencoder(frames=8, size=112, embedding_dim=256,
                                              encoder_layers=6, decoder_layers=2, heads=8)
        video_meta = json.loads(args.video_cache.with_suffix(args.video_cache.suffix + ".json").read_text())
        if not video_meta["complete"] or video_meta["shape"] != [len(valid), 3, 8, 112, 112]:
            raise ValueError("video decode cache is incomplete or has wrong shape")
        completed = set(video_meta["completed_indices"])
        if any(int(i) not in completed for i in np.flatnonzero(valid)):
            raise ValueError("video valid mask contains an undecoded row")
        raw = np.memmap(args.video_cache, mode="r", dtype=np.uint8, shape=tuple(video_meta["shape"]))
        token_count, heads, total_layers = 196, 8, 6
    else:
        temporal = load_module("temporal_mae_prefix_source", here / "modality_mae.py")
        if args.modality == "eeg":
            model = temporal.TemporalMaskedAutoencoder(patch_dim=11800, embedding_dim=256,
                                                       encoder_layers=6, decoder_layers=2, heads=8)
            raw = np.load(args.eeg_source, mmap_mode="r")
            if raw.shape != (len(valid), 2000, 59):
                raise ValueError("EEG shape/order contract changed")
            heads, total_layers = 8, 6
        else:
            model = temporal.WearMaskedAutoencoder(embedding_dim=256, encoder_layers=4,
                                                   decoder_layers=2, heads=4)
            raw = []
            raw_valid = np.ones(len(valid), dtype=bool)
            for filename, key in (("ppg_10s.npz", "ppg"), ("gsr_10s.npz", "gsr"), ("acc_10s.npz", "acc")):
                with np.load(args.wear_root / filename, allow_pickle=True) as signals:
                    if not np.array_equal(signals["sample_id"].astype(str), sample_id):
                        raise ValueError("Wear canonical sample order mismatch")
                    raw.append(signals[key].astype(np.float32))
                    raw_valid &= signals["valid_mask"].astype(bool)
            if not np.array_equal(raw_valid, valid):
                raise ValueError("Wear complete mask mismatch")
            heads, total_layers = 4, 4
        token_count = 10
    if args.initialization == "pretrained":
        if args.checkpoint is None:
            raise ValueError("pretrained prefix requires the Stage-A checkpoint")
        model.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    dev = torch.device(args.device)
    model.to(dev).eval()
    tail = make_tail(model).to(dev).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    valid_indices = np.flatnonzero(valid)
    prefix_tmp = args.out_dir / "prefix.partial.npy"
    prefix = np.lib.format.open_memmap(prefix_tmp, mode="w+", dtype=np.float32,
                                      shape=(len(valid_indices), token_count, 256))
    embedding = np.zeros((len(valid), 256), dtype=np.float32)
    largest_error = 0.0
    composition_error = 0.0
    with torch.no_grad():
        for start in range(0, len(valid_indices), args.batch_size):
            idx = valid_indices[start:start + args.batch_size]
            if args.modality == "eeg":
                patches = torch.as_tensor(temporal.eeg_to_patches(raw[idx], normalization), device=dev)
                hidden = model.patch_embed(patches) + model.position
            elif args.modality == "wear":
                patches = [torch.as_tensor(x, device=dev) for x in temporal.wear_to_patches(*(x[idx] for x in raw), normalization=normalization)]
                hidden = model._embed(*patches)
            else:
                clip = torch.as_tensor(raw[idx].astype(np.float32) / 255.0, device=dev)
                hidden = model._tokens(clip) + model.position
            embedded = hidden
            for block in model.encoder.layers[:-2]:
                hidden = block(hidden)
            prefix[start:start + len(idx)] = hidden.cpu().numpy()
            full = tail(hidden).mean(dim=1).cpu().numpy()
            if start == 0:
                direct = model.encoder(embedded).mean(dim=1).cpu().numpy()
                composition_error = float(np.abs(full - direct).max())
                if composition_error > 1e-5:
                    raise ValueError(f"frozen prefix composition differs from full encoder: {composition_error}")
            if not np.isfinite(full).all() or not torch.isfinite(hidden).all():
                raise RuntimeError("non-finite frozen prefix")
            embedding[idx] = full
            if args.initialization == "pretrained":
                largest_error = max(largest_error, float(np.abs(full - expected[idx]).max()))
            if start % (args.batch_size * 20) == 0:
                print(f"prefix_progress modality={args.modality} init={args.initialization} rows={start + len(idx)}/{len(valid_indices)}", flush=True)
    # Archived exports used different batch sizes and fused CUDA kernels. Allow
    # their small fp32 roundoff; the same-batch full-encoder check above is strict.
    if args.initialization == "pretrained" and largest_error > 2e-4:
        raise ValueError(f"prefix + tail fails Stage-A token equivalence: max_error={largest_error}")
    prefix.flush()
    del prefix
    prefix_tmp.replace(args.out_dir / "prefix.npy")
    torch.save({"state_dict": tail.cpu().state_dict(), "heads": heads, "blocks": 2,
                "embedding_dim": 256, "initialization": args.initialization,
                "initialization_seed": seed, "modality": args.modality}, args.out_dir / "tail.pt")
    # Preserve the full initialization for replay, including the frozen prefix.
    torch.save({"state_dict": model.cpu().state_dict(), "modality": args.modality,
                "initialization": args.initialization, "initialization_seed": seed},
               args.out_dir / "encoder_initialization.pt")
    np.savez_compressed(args.out_dir / "window_embeddings.npz", sample_id=sample_id,
                        valid_mask=valid, embedding=embedding, valid_indices=valid_indices)
    manifest = {"modality": args.modality, "initialization": args.initialization,
                **provenance,
                "initialization_seed": seed,
                "checkpoint": str(args.checkpoint) if args.initialization == "pretrained" else None,
                "ssl_checkpoint_loaded": args.initialization == "pretrained",
                "source_token_file": str(args.token_file), "labels_read": False,
                "raw_normalization": normalization,
                "prefix_export_batch_size": args.batch_size,
                "requested_batch_size": requested_batch_size,
                "batch_size_matched_to_stage_a": args.match_stage_a_batch_size,
                "row_count": len(valid), "valid_count": len(valid_indices),
                "prefix_shape": [len(valid_indices), token_count, 256], "dtype": "float32",
                "frozen_blocks": total_layers - 2, "trainable_tail_blocks": 2,
                "prefix_dropout": "eval", "stage_a_embedding_max_abs_error": largest_error,
                "same_batch_full_encoder_max_abs_error": composition_error,
                "prefix_sha256": file_hash(args.out_dir / "prefix.npy")}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.out_dir / "PREFIX_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
