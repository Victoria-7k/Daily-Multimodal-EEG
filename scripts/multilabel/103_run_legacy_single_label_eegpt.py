#!/usr/bin/env python3
"""Build label-specific EEGPT tokens with the original 0814 window-level FT contract.

This deliberately calls eeg_encoder_matrix.run_torch_eeg_profile rather than the
event-balanced/strict-last-block trainer used by the 20260913 token bank.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.training.eeg_encoder_matrix import (
    DEFAULT_DATA_ROOT,
    DEFAULT_INDEX_PATH,
    DEFAULT_SPLITS_ROOT,
    MatrixRuntime,
    load_eeg_aligned_dataset,
    load_split_protocols,
    run_torch_eeg_profile,
    target_array,
    write_prediction_npz,
)

PROTOCOLS = ("cross_day", "within_subject_day")
SEED = 240800
GOLDEN_FATIGUE = Path(
    "/vePFS-0x0d/DailyEEG_multimodal/embeddings/eeg_encoder_256d_tokens/"
    "cross_day/eegpt_partial_ft_v1/seed_240800.npz"
)
TOKEN_ROOT = Path(
    "/vePFS-0x0d/DailyEEG_multimodal/embeddings/"
    "eeg_encoder_256d_tokens_legacy_20260917"
)


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def audit_token(path: Path, dataset, split, protocol: str, label: str, seed: int) -> None:
    install_numpy_core_pickle_aliases()
    with np.load(path, allow_pickle=True) as token:
        if not np.array_equal(token["sample_id"].astype(str), dataset.sample_id.astype(str)):
            raise ValueError(f"sample order differs: {path}")
        for key, expected in (("train_index", split.train), ("val_index", split.val), ("test_index", split.test)):
            if not np.array_equal(token[key].astype(np.int64), expected):
                raise ValueError(f"{key} differs: {path}")
        if token["eeg_emb"].shape != (dataset.row_count, 256) or not np.isfinite(token["eeg_emb"]).all():
            raise ValueError(f"invalid EEG token: {path}")
        if str(token["protocol"][0]) != protocol or int(token["seed"][0]) != seed:
            raise ValueError(f"protocol/seed mismatch: {path}")
        if "target_label" in token and str(token["target_label"][0]) != label:
            raise ValueError(f"target label mismatch: {path}")


def write_token(path: Path, embeddings: np.ndarray, dataset, split, protocol: str,
                label: str, seed: int, prediction_path: Path) -> None:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.shape != (dataset.row_count, 256) or not np.isfinite(values).all():
        raise ValueError(f"invalid generated EEG tokens: {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    modality_mask = np.zeros((dataset.row_count, 4), dtype=np.int8)
    modality_mask[:, 0] = 1
    np.savez_compressed(
        path,
        sample_id=dataset.sample_id,
        subject_id=dataset.subject_id,
        day_id=dataset.day_id,
        eeg_emb=values,
        eeg_mask=np.ones(dataset.row_count, dtype=np.int8),
        modality_mask=modality_mask,
        encoder_profile=np.asarray([f"eegpt_partial_ft_single_{label}_legacy_v1"] * dataset.row_count, dtype=object),
        encoder_version=np.asarray(["eeg_encoder_256d_supervised_v1"] * dataset.row_count, dtype=object),
        protocol=np.asarray([protocol], dtype=object),
        seed=np.asarray([seed], dtype=np.int64),
        target_label=np.asarray([label], dtype=object),
        train_index=split.train,
        val_index=split.val,
        test_index=split.test,
        train_supervision=np.asarray([f"{label}_supervised_legacy_window_mse_train_val_selected"], dtype=object),
        supervision_boundary=np.asarray(["legacy_window_mse_last2_plus_norm_proj"], dtype=object),
        source_prediction_npz=np.asarray([str(prediction_path)], dtype=object),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--splits-root", type=Path, default=DEFAULT_SPLITS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/checkpoints/eegpt-pretrained")
    parser.add_argument("--golden-fatigue-token", type=Path, default=GOLDEN_FATIGUE)
    parser.add_argument("--token-root", type=Path, default=TOKEN_ROOT)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/multiemotion_legacy_replay_20260917/legacy_eeg"))
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--labels", default=",".join(LABEL_NAMES))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    protocols, labels = split_csv(args.protocols), split_csv(args.labels)
    if not protocols or set(protocols) - set(PROTOCOLS) or not labels or set(labels) - set(LABEL_NAMES):
        raise ValueError("unsupported or empty protocol/label selection")
    if args.seed != SEED:
        raise ValueError("this matched experiment locks upstream EEG seed to 240800")
    if not args.checkpoint.is_dir() or not args.golden_fatigue_token.is_file():
        raise FileNotFoundError("old EEGPT checkpoint or golden cross-day fatigue token is missing")
    dataset = load_eeg_aligned_dataset(data_root=args.data_root, index_path=args.index_path)
    splits = load_split_protocols(args.splits_root, protocols, row_count=dataset.row_count)
    runtime = MatrixRuntime()
    if runtime.batch_size != 256 or runtime.partial_last_n_blocks != 2 or runtime.epochs != 80:
        raise ValueError("legacy EEGPT runtime defaults drifted")
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "contract.json").write_text(json.dumps({
        "profile": "eegpt_partial_ft_v1", "target_labels": list(labels), "protocols": list(protocols),
        "upstream_seed": args.seed, "runtime": asdict(runtime), "checkpoint": str(args.checkpoint),
        "golden_fatigue_token": str(args.golden_fatigue_token),
        "selection_metric": "validation window-level RMSE",
        "train_objective": "window-level standardized MSE",
        "encoder_trainability": "legacy partial: last two blocks plus name-matched norm/proj/head",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.preflight_only:
        for protocol in protocols:
            split = splits[protocol]
            print(f"preflight protocol={protocol} train={len(split.train)} val={len(split.val)} test={len(split.test)}")
        for label in labels:
            values = target_array(dataset.y, label)
            if values.shape != (dataset.row_count,) or not np.isfinite(values).all():
                raise ValueError(f"invalid target for {label}")
        return 0
    attempted = 0
    for protocol in protocols:
        split = splits[protocol]
        for label in labels:
            path = args.token_root / "single_task" / protocol / label / f"seed_{args.seed}.npz"
            run_dir = args.out_root / protocol / label / f"seed_{args.seed}"
            metrics_path = run_dir / "metrics.json"
            if path.is_file() and metrics_path.is_file():
                previous = json.loads(metrics_path.read_text(encoding="utf-8"))
                if previous.get("status") == "ok":
                    audit_token(path, dataset, split, protocol, label, args.seed)
                    print(f"skipped completed protocol={protocol} label={label}", flush=True)
                    continue
            if args.limit_runs and attempted >= args.limit_runs:
                return 0
            attempted += 1
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            print(f"starting legacy EEGPT protocol={protocol} label={label} seed={args.seed}", flush=True)
            if protocol == "cross_day" and label == "fatigue":
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.golden_fatigue_token, path)
                audit_token(path, dataset, split, protocol, label, args.seed)
                result = {"status": "ok", "reused_exact_historical_token": True,
                          "source_token": str(args.golden_fatigue_token)}
            else:
                target = target_array(dataset.y, label)
                result = run_torch_eeg_profile(
                    x=dataset.x, target=target, subjects=dataset.subject_id, split=split,
                    protocol=protocol, profile="eegpt_partial_ft_v1", seed=args.seed,
                    runtime=runtime, cbramod_checkpoint=None, eegpt_checkpoint=args.checkpoint,
                    allow_cbramod_download=False,
                )
                trainability = result["trainability"]
                if (trainability["strategy"], trainability["parameter_count"],
                        trainability["trainable_count"]) != ("partial", 104, 66):
                    raise ValueError(f"legacy trainability drift: {trainability}")
                predictions = result.pop("predictions")
                embeddings = result.pop("embeddings")
                prediction_path = run_dir / "window_predictions.npz"
                write_prediction_npz(predictions, prediction_path, dataset=dataset, split=split, target=target)
                write_token(path, embeddings, dataset, split, protocol, label, args.seed, prediction_path)
                audit_token(path, dataset, split, protocol, label, args.seed)
                result["prediction_path"] = str(prediction_path)
            result.update({"status": "ok", "protocol": protocol, "target_label": label,
                           "seed": args.seed, "token_path": str(path),
                           "duration_seconds": time.time() - started,
                           "legacy_runtime": asdict(runtime)})
            metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"completed legacy EEGPT protocol={protocol} label={label} "
                  f"seconds={result['duration_seconds']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
