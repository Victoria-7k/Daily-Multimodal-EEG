#!/usr/bin/env python3
"""Audit the reusable one-seed, single-label EEGPT embedding bank."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import load_jsonl, load_window_split
from daily_multimodal.daily_affect.npz_compat import install_numpy_core_pickle_aliases
from daily_multimodal.training.eeg_multitask import LABEL_NAMES


def audit(args: argparse.Namespace) -> dict:
    install_numpy_core_pickle_aliases()
    rows = load_jsonl(args.root / "index/eeg_aligned_window_index.jsonl")
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("canonical index has duplicate sample_id")
    inventory = []
    protocols = tuple(value.strip() for value in args.protocols.split(",") if value.strip())
    if not protocols or len(protocols) != len(set(protocols)) or set(protocols) - {"cross_day", "within_subject_day", "date_in_order"}:
        raise ValueError("unsupported or duplicated protocols")
    split_root = args.splits_root or args.root / "outputs/splits"
    for protocol in protocols:
        leaf = load_window_split(split_root / protocol, len(rows))
        expected_split = {
            "train_index": np.flatnonzero(np.isin(leaf, ("pretrain", "finetune"))),
            "val_index": np.flatnonzero(leaf == "val"),
            "test_index": np.flatnonzero(leaf == "test"),
        }
        for label in LABEL_NAMES:
            path = args.embeddings_root / "eeg_encoder_256d_tokens/single_task" / protocol / label / "seed_240800.npz"
            metrics_path = args.root / "outputs/multiemotion_20260913/phase3_single_task_eeg" / protocol / label / "seed_240800/metrics.json"
            item = {"protocol": protocol, "label": label, "embedding_seed": 240800,
                    "split_root": str(split_root / protocol),
                    "token_path": str(path), "metrics_path": str(metrics_path), "checks": {}, "errors": []}
            checks = item["checks"]
            if not path.is_file():
                item["errors"].append("token_missing")
            else:
                try:
                    with np.load(path, allow_pickle=True) as token:
                        values = token["eeg_emb"]
                        checks["shape"] = values.shape == (len(rows), 256)
                        checks["finite"] = bool(np.isfinite(values).all())
                        checks["sample_order"] = bool(np.array_equal(token["sample_id"].astype(str), sample_ids))
                        checks["protocol"] = str(token["protocol"][0]) == protocol
                        checks["target_label"] = str(token["target_label"][0]) == label
                        checks["seed"] = int(token["seed"][0]) == 240800
                        checks["supervision"] = (
                            str(token["train_supervision"][0]) == "single_event_supervised_eegpt_partial_ft"
                            and str(token["supervision_boundary"][0]) == "raw_eeg_encoder_and_eeg_only_heads"
                        )
                        for key, indices in expected_split.items():
                            checks[key] = bool(np.array_equal(np.sort(token[key].astype(np.int64)), indices))
                except (OSError, KeyError, ValueError) as exc:
                    item["errors"].append(f"token_read:{type(exc).__name__}:{exc}")
            if not metrics_path.is_file():
                item["errors"].append("metrics_missing")
            else:
                try:
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    checks["metrics_status"] = metrics.get("status") == "ok"
                    checks["metrics_identity"] = (
                        metrics.get("protocol") == protocol and metrics.get("task_name") == label
                        and metrics.get("seed") == 240800
                    )
                except (OSError, ValueError) as exc:
                    item["errors"].append(f"metrics_read:{type(exc).__name__}:{exc}")
            item["gate_passed"] = bool(checks and all(checks.values()) and not item["errors"])
            inventory.append(item)
    return {"expected_count": len(protocols) * len(LABEL_NAMES), "passed_count": sum(item["gate_passed"] for item in inventory),
            "embedding_seed": 240800, "sample_count": len(rows), "embedding_dim": 256,
            "gate_passed": all(item["gate_passed"] for item in inventory), "inventory": inventory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/vePFS-0x0d/home/wangzw/DailyEEG_multimodal_eeg_aligned"))
    parser.add_argument("--splits-root", type=Path)
    parser.add_argument("--protocols", default="cross_day,date_in_order")
    parser.add_argument("--embeddings-root", type=Path, default=Path("/vePFS-0x0d/DailyEEG_multimodal/embeddings"))
    parser.add_argument("--out", type=Path, default=Path("outputs/multiemotion_20260913/phase3_single_task_eeg/bank_audit.json"))
    args = parser.parse_args()
    result = audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"bank_audit={result['passed_count']}/{result['expected_count']} gate_passed={result['gate_passed']} out={args.out}")
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
