#!/usr/bin/env python3
"""Convert EEG encoder matrix predictions into fusion-compatible EEG tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


DEFAULT_PROFILES = (
    "eegpt_partial_ft_v1",
    "cbramod_frozen_v1",
    "cbramod_partial_ft_v1",
    "eeg_de_5band_1s_avg_v1",
)
DEFAULT_PROTOCOLS = ("cross_subject", "cross_day", "within_subject_day")
DEFAULT_SEEDS = (240800, 240801, 240802, 240803, 240804)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=Path("outputs/predictions/eeg_encoder_matrix"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/embeddings/eeg_encoder_tokens"))
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--protocols", default=",".join(DEFAULT_PROTOCOLS))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    profiles = _split_csv(args.profiles)
    protocols = _split_csv(args.protocols)
    seeds = tuple(int(value) for value in _split_csv(args.seeds))
    rows = []
    failures = []
    for protocol in protocols:
        for profile in profiles:
            for seed in seeds:
                source = args.predictions_root / protocol / profile / f"seed_{seed}.npz"
                target = args.out_root / protocol / profile / f"seed_{seed}.npz"
                try:
                    report = convert_prediction_npz(source, target, embedding_dim=args.embedding_dim)
                    rows.append({"protocol": protocol, "profile": profile, "seed": int(seed), **report})
                except Exception as exc:
                    failures.append(
                        {
                            "protocol": protocol,
                            "profile": profile,
                            "seed": int(seed),
                            "source": str(source),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
    summary = {
        "task": "eeg_encoder_prediction_tokens",
        "predictions_root": str(args.predictions_root),
        "out_root": str(args.out_root),
        "profiles": list(profiles),
        "protocols": list(protocols),
        "seeds": [int(seed) for seed in seeds],
        "embedding_dim": int(args.embedding_dim),
        "token_count": len(rows),
        "failure_count": len(failures),
        "tokens": rows,
        "failures": failures,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"token_count={len(rows)}")
    print(f"failure_count={len(failures)}")
    if args.out_json:
        print(f"out_json={args.out_json}")
    return 0 if not failures else 1


def convert_prediction_npz(source: Path, target: Path, *, embedding_dim: int) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=True) as loaded:
        sample_id = loaded["sample_id"].astype(str)
        subject_id = loaded["subject_id"].astype(str) if "subject_id" in loaded.files else np.asarray([""] * len(sample_id), dtype=str)
        day_id = loaded["day_id"].astype(str) if "day_id" in loaded.files else np.asarray([""] * len(sample_id), dtype=str)
        train_index = loaded["train_index"].astype(np.int64)
        val_index = loaded["val_index"].astype(np.int64)
        test_index = loaded["test_index"].astype(np.int64)
        train_prediction = loaded["train_prediction"].astype(np.float32)
        val_prediction = loaded["val_prediction"].astype(np.float32)
        test_prediction = loaded["test_prediction"].astype(np.float32)
    row_count = len(sample_id)
    pred = np.full(row_count, np.nan, dtype=np.float32)
    _fill_split_prediction(pred, train_index, train_prediction, "train")
    _fill_split_prediction(pred, val_index, val_prediction, "val")
    _fill_split_prediction(pred, test_index, test_prediction, "test")
    finite = np.isfinite(pred)
    if not finite.any():
        raise ValueError(f"{source} contains no finite predictions")
    emb = np.zeros((row_count, int(embedding_dim)), dtype=np.float32)
    emb[:, 0] = np.where(finite, pred, 0.0).astype(np.float32)
    mask = finite.astype(bool)
    modality_mask = np.zeros((row_count, 4), dtype=np.int8)
    modality_mask[:, 0] = mask.astype(np.int8)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        sample_id=sample_id,
        subject_id=subject_id,
        day_id=day_id,
        eeg_emb=emb,
        eeg_mask=mask.astype(np.int8),
        modality_mask=modality_mask,
        encoder_version=np.asarray(["eeg_encoder_prediction_token_v1"] * row_count, dtype=object),
        source_prediction_npz=np.asarray([str(source)] * row_count, dtype=object),
    )
    return {
        "source": str(source),
        "path": str(target),
        "row_count": int(row_count),
        "mask_sum": int(mask.sum()),
        "finite_prediction_count": int(finite.sum()),
    }


def _fill_split_prediction(target: np.ndarray, indices: np.ndarray, prediction: np.ndarray, split_name: str) -> None:
    if len(indices) != len(prediction):
        raise ValueError(f"{split_name} index/prediction length mismatch: {len(indices)} != {len(prediction)}")
    if len(indices):
        target[indices] = prediction


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    raise SystemExit(main())
