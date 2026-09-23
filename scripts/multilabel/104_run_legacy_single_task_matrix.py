#!/usr/bin/env python3
"""Run the independent 19-structure x 11-label matrix with legacy EEGPT tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES, build_daily_affect_bags, load_jsonl, load_window_split
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.daily_affect.regression_training import run_daily_affect_regression_run
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.structure_emotion import conditions

PROTOCOLS = ("cross_day", "date_in_order")
SEEDS = (240729, 240730, 240731)
EMBEDDING_SEED = 240800
TOKEN_ROOT_NAME = "eeg_encoder_256d_tokens_legacy_20260917"


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def audit_token(path: Path, index_rows: list[dict], split_root: Path,
                protocol: str, label: str) -> None:
    install_numpy_core_pickle_aliases()
    expected_ids = np.asarray([str(row["sample_id"]) for row in index_rows])
    leaves = load_window_split(split_root / protocol, len(index_rows))
    expected = {
        "train_index": np.flatnonzero(np.isin(leaves, ("pretrain", "finetune"))),
        "val_index": np.flatnonzero(leaves == "val"),
        "test_index": np.flatnonzero(leaves == "test"),
    }
    with np.load(path, allow_pickle=True) as token:
        if not np.array_equal(token["sample_id"].astype(str), expected_ids):
            raise ValueError(f"EEG sample order mismatch: {path}")
        for key, indices in expected.items():
            if not np.array_equal(np.sort(token[key].astype(np.int64)), indices):
                raise ValueError(f"EEG {key} mismatch: {path}")
        if str(token["protocol"][0]) != protocol or int(token["seed"][0]) != EMBEDDING_SEED:
            raise ValueError(f"EEG protocol/seed mismatch: {path}")
        if protocol == "cross_day" and label == "fatigue":
            profile, supervision = "eegpt_partial_ft_v1", "fatigue_supervised_train_val_selected"
        else:
            profile = f"eegpt_partial_ft_single_{label}_legacy_v1"
            supervision = f"{label}_supervised_legacy_window_mse_train_val_selected"
            if str(token["target_label"][0]) != label:
                raise ValueError(f"EEG target label mismatch: {path}")
        if (np.unique(token["encoder_profile"].astype(str)).tolist() != [profile] or
                str(token["train_supervision"][0]) != supervision):
            raise ValueError(f"EEG legacy provenance mismatch: {path}")
        if token["eeg_emb"].shape != (len(index_rows), 256) or not np.isfinite(token["eeg_emb"]).all():
            raise ValueError(f"invalid EEG embeddings: {path}")


def audit_bag(bag, token_path: Path, protocol: str, label: str) -> dict:
    expected_route = f"A1_Wphysio_no_audio__eeg_eegpt_partial_ft_single_{label}_v1"
    sources = json.loads(bag.source_npz_json)
    source_key = f"eeg_eegpt_partial_ft_single_{label}_v1"
    if bag.route_id != expected_route or bag.target_label != label:
        raise ValueError(f"bag route/target mismatch: {bag.bag_path}")
    if set(sources) != {source_key, "wear_physio", "video_A1"}:
        raise ValueError(f"unexpected bag sources: {sources}")
    if Path(sources[source_key]) != token_path:
        raise ValueError(f"bag EEG source mismatch: {sources[source_key]}")
    if bag.tokens.shape != (1253, 23, 4, 256) or bag.modality_mask[:, :, 3].any():
        raise ValueError(f"bag shape/audio mismatch: {bag.bag_path}")
    split = bag.split_indices()
    groups = {leaf: set(split[leaf].tolist()) for leaf in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"bag split overlap: {bag.bag_path}")
    if protocol == "date_in_order":
        days = {leaf: {(str(bag.subject_id[i]), str(bag.day_id[i])) for i in indices}
                for leaf, indices in groups.items()}
        if any(days[a] & days[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"subject-day overlap: {bag.bag_path}")
    return {"protocol": protocol, "label": label, "route_id": expected_route,
            "token_path": str(token_path), "bag_path": str(bag.bag_path),
            "split_counts": {leaf: len(groups[leaf]) for leaf in groups},
            "supervision_boundary": bag.supervision_boundary}


def audit_historical_fatigue_bag(bag, historical_path: Path) -> None:
    old = load_bag_dataset(historical_path)
    for key in ("event_id", "label", "tokens", "modality_mask", "train_index", "val_index", "test_index"):
        if not np.array_equal(getattr(bag, key), getattr(old, key)):
            raise ValueError(f"rebuilt cross-day fatigue bag differs in {key}: {bag.bag_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"))
    parser.add_argument("--embeddings-root", type=Path, default=Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings"))
    parser.add_argument("--token-root-name", default=TOKEN_ROOT_NAME)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/multiemotion_legacy_replay_20260917/single_task_matrix_A1"))
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--labels", default=",".join(LABEL_NAMES))
    parser.add_argument("--conditions", default=",".join(conditions()))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--stage", choices=("preflight", "matrix"), default="matrix")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    args = parser.parse_args()
    protocols, labels, selected = split_csv(args.protocols), split_csv(args.labels), split_csv(args.conditions)
    seeds = tuple(map(int, split_csv(args.seeds)))
    if (not protocols or set(protocols) - set(PROTOCOLS) or not labels or set(labels) - set(LABEL_NAMES)
            or not selected or set(selected) - set(conditions()) or not seeds):
        raise ValueError("unsupported or empty protocol/label/condition/seed selection")
    if args.batch_size != 128 or args.epochs != 80 or args.patience != 15:
        raise ValueError("legacy downstream batch/epoch/patience contract drifted")
    torch.set_num_threads(max(1, args.torch_threads))
    if args.stage == "matrix" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    index_path = args.root / "index/eeg_aligned_window_index.jsonl"
    index_rows = load_jsonl(index_path)
    split_root = args.root / "outputs/splits"
    attempted = 0
    for protocol in protocols:
        for label in labels:
            branch = f"eeg_eegpt_partial_ft_single_{label}_v1"
            token_path = (args.embeddings_root / args.token_root_name / "single_task" /
                          protocol / label / f"seed_{EMBEDDING_SEED}.npz")
            audit_token(token_path, index_rows, split_root, protocol, label)
            route = f"A1_Wphysio_no_audio__{branch}"
            bag_dir = args.out_root / "bags" / protocol / label / route / f"embedding_seed_{EMBEDDING_SEED}"
            bag_path = bag_dir / "ema_bags.npz"
            if not bag_path.is_file():
                build_daily_affect_bags(
                    index_path=index_path, splits_root=split_root, embeddings_root=args.embeddings_root,
                    protocol=protocol, route_id="A1_Wphysio_no_audio", out_dir=bag_dir,
                    target_label=label, eeg_branch=branch, eeg_seed=EMBEDDING_SEED,
                    eeg_token_root=args.token_root_name,
                )
            bag = load_bag_dataset(bag_path)
            audit = audit_bag(bag, token_path, protocol, label)
            if protocol == "cross_day" and label == "fatigue":
                historical_path = (args.root / "outputs/daily_affect_scalar_regression_20260908/"
                                   "v2_partialft_noaudio_bags/cross_day/"
                                   "A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1/seed_240729/ema_bags.npz")
                audit_historical_fatigue_bag(bag, historical_path)
                audit["historical_bag_array_equal"] = True
            audit_path = args.out_root / "preflight" / protocol / f"{label}.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"preflight passed protocol={protocol} label={label}", flush=True)
            if args.stage == "preflight":
                continue
            for condition_id in selected:
                model_id, temporal_policy = conditions()[condition_id]
                for seed in seeds:
                    run_dir = args.out_root / "runs" / protocol / label / condition_id / f"seed_{seed}"
                    metrics_path = run_dir / "metrics.json"
                    if metrics_path.is_file() and (run_dir / "predictions.npz").is_file():
                        old = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if (old.get("status") == "ok" and old.get("embedding_seed") == EMBEDDING_SEED
                                and old.get("downstream_seed") == seed and old.get("target_label") == label
                                and old.get("eeg_token_path") == str(token_path)):
                            print(f"skipped completed protocol={protocol} label={label} "
                                  f"condition={condition_id} seed={seed}", flush=True)
                            continue
                        raise ValueError(f"existing run has incompatible provenance: {metrics_path}")
                    print(f"starting protocol={protocol} label={label} condition={condition_id} seed={seed}", flush=True)
                    result = run_daily_affect_regression_run(
                        dataset=bag, protocol=protocol, condition_id=condition_id,
                        model_id=model_id, normalization="per_modality", adapter_mode="per_modality",
                        temporal_policy=temporal_policy, seed=seed, run_dir=run_dir,
                        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, device=args.device,
                    )
                    result.update({"status": "ok", "embedding_seed": EMBEDDING_SEED,
                                   "downstream_seed": seed, "eeg_token_path": str(token_path),
                                   "pipeline": "single_task_legacy_eegpt_scalar"})
                    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    attempted += 1
                    print(f"completed protocol={protocol} label={label} condition={condition_id} "
                          f"seed={seed} raw_r={result['test']['raw_r']}", flush=True)
                    if args.limit_runs and attempted >= args.limit_runs:
                        return 0
    print(f"run_count={attempted} stage={args.stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
