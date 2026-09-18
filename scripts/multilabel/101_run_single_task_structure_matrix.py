#!/usr/bin/env python3
"""Independent scalar 0814/0906 regressions using each label's own EEGPT token."""

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

EMBEDDING_SEED = 240800
PROTOCOLS = ("cross_day", "within_subject_day")
SEEDS = (240800, 240801, 240802)


def split_csv(text: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in text.split(",") if value.strip())


def audit_token(path: Path, index_rows: list[dict], split_root: Path, protocol: str, label: str) -> None:
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
            raise ValueError(f"sample order mismatch: {path}")
        for key, indices in expected.items():
            if not np.array_equal(np.sort(token[key].astype(np.int64)), indices):
                raise ValueError(f"{key} mismatch: {path}")
        for key, value in {
            "protocol": protocol,
            "target_label": label,
            "encoder_profile": f"eegpt_partial_ft_single_{label}_v1",
            "train_supervision": "single_event_supervised_eegpt_partial_ft",
        }.items():
            if np.unique(token[key].astype(str)).tolist() != [value]:
                raise ValueError(f"{key} mismatch: {path}")
        if int(token["seed"][0]) != EMBEDDING_SEED:
            raise ValueError(f"embedding seed mismatch: {path}")


def audit_bag(dataset, protocol: str, label: str, token_path: Path) -> dict:
    expected_route = f"A1_Wphysio_no_audio__eeg_eegpt_partial_ft_single_{label}_v1"
    sources = json.loads(dataset.source_npz_json)
    if dataset.route_id != expected_route or dataset.target_label != label:
        raise ValueError(f"bag target/route mismatch: {dataset.bag_path}")
    if set(sources) != {f"eeg_eegpt_partial_ft_single_{label}_v1", "wear_physio", "video_A1"}:
        raise ValueError(f"bag sources mismatch: {sources}")
    if Path(sources[f"eeg_eegpt_partial_ft_single_{label}_v1"]) != token_path:
        raise ValueError(f"bag EEG source mismatch: {sources}")
    if dataset.tokens.shape != (1253, 23, 4, 256) or dataset.modality_mask[:, :, 3].any():
        raise ValueError(f"bag shape/no-audio mismatch: {dataset.bag_path}")
    if not dataset.modality_mask[:, :, 0].all() or not np.isfinite(dataset.tokens).all():
        raise ValueError(f"bag EEG/finite mismatch: {dataset.bag_path}")
    split = dataset.split_indices()
    groups = {name: set(split[name].tolist()) for name in ("train", "val", "test")}
    if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError(f"event split overlap: {dataset.bag_path}")
    if protocol == "within_subject_day":
        days = {name: {(dataset.subject_id[i], dataset.day_id[i]) for i in groups[name]} for name in groups}
        if any(days[a] & days[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError(f"subject-day overlap: {dataset.bag_path}")
    return {"protocol": protocol, "label": label, "route_id": expected_route,
            "embedding_seed": EMBEDDING_SEED, "token_path": str(token_path),
            "bag_path": str(dataset.bag_path),
            "split_counts": {name: len(groups[name]) for name in groups},
            "supervision_boundary": dataset.supervision_boundary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"))
    parser.add_argument("--embeddings-root", type=Path, default=Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/multiemotion_20260913/single_task_structure_matrix_A1"))
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--labels", default=",".join(LABEL_NAMES))
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="Downstream only; EEG token seed is fixed at 240800.")
    parser.add_argument("--conditions", default=",".join(conditions()))
    parser.add_argument("--stage", choices=("preflight", "matrix"), default="matrix")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    protocols, labels = split_csv(args.protocols), split_csv(args.labels)
    seeds = tuple(map(int, split_csv(args.seeds)))
    selected = split_csv(args.conditions)
    if not protocols or set(protocols) - set(PROTOCOLS) or not labels or set(labels) - set(LABEL_NAMES):
        raise ValueError("unsupported/empty protocol or label selection")
    if not selected or set(selected) - set(conditions()) or not seeds:
        raise ValueError("unsupported/empty condition or downstream seed selection")
    torch.set_num_threads(max(1, args.torch_threads))
    if args.stage == "matrix" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    index_path = args.root / "index/eeg_aligned_window_index.jsonl"
    index_rows = load_jsonl(index_path)
    split_root = args.root / "outputs/splits"
    audits = []
    attempted = 0
    for protocol in protocols:
        for label in labels:
            branch = f"eeg_eegpt_partial_ft_single_{label}_v1"
            token_path = args.embeddings_root / "eeg_encoder_256d_tokens/single_task" / protocol / label / f"seed_{EMBEDDING_SEED}.npz"
            audit_token(token_path, index_rows, split_root, protocol, label)
            route = f"A1_Wphysio_no_audio__{branch}"
            bag_dir = args.out_root / "bags" / protocol / label / route / f"embedding_seed_{EMBEDDING_SEED}"
            bag_path = bag_dir / "ema_bags.npz"
            if not bag_path.is_file():
                build_daily_affect_bags(index_path=index_path, splits_root=split_root,
                    embeddings_root=args.embeddings_root, protocol=protocol,
                    route_id="A1_Wphysio_no_audio", out_dir=bag_dir,
                    target_label=label, eeg_branch=branch, eeg_seed=EMBEDDING_SEED)
            dataset = load_bag_dataset(bag_path)
            audit = audit_bag(dataset, protocol, label, token_path)
            audits.append(audit)
            args.out_root.mkdir(parents=True, exist_ok=True)
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
                    if args.skip_existing and metrics_path.is_file() and (run_dir / "predictions.npz").is_file():
                        old = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if old.get("embedding_seed") == EMBEDDING_SEED and old.get("target_label") == label:
                            continue
                    print(f"starting protocol={protocol} label={label} condition={condition_id} downstream_seed={seed}", flush=True)
                    result = run_daily_affect_regression_run(
                        dataset=dataset, protocol=protocol, condition_id=condition_id,
                        model_id=model_id, normalization="per_modality", adapter_mode="per_modality",
                        temporal_policy=temporal_policy, seed=seed, run_dir=run_dir,
                        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience, device=args.device)
                    result.update({"status": "ok", "embedding_seed": EMBEDDING_SEED,
                                   "downstream_seed": seed, "eeg_token_path": str(token_path),
                                   "pipeline": "single_task_label_specific_eegpt_scalar"})
                    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    attempted += 1
                    print(f"completed protocol={protocol} label={label} condition={condition_id} seed={seed} raw_r={result['test']['raw_r']}", flush=True)
                    if args.limit_runs and attempted >= args.limit_runs:
                        print(f"run_count={attempted} limit_reached=true", flush=True)
                        return 0
    if args.stage == "preflight" and set(protocols) == set(PROTOCOLS) and set(labels) == set(LABEL_NAMES):
        (args.out_root / "preflight.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"run_count={attempted} stage={args.stage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
