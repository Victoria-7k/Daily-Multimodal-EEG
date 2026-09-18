#!/usr/bin/env python3
"""Summarize independent label-specific EEGPT scalar runs as structure x emotion tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from daily_multimodal.daily_affect.ema_bags import LABEL_NAMES
from daily_multimodal.daily_affect.training import load_bag_dataset
from daily_multimodal.training.structure_emotion import conditions

PROTOCOLS = ("cross_day", "within_subject_day")
SEEDS = (240800, 240801, 240802)
METRICS = ("raw_r", "standardized_rmse", "within_subject_centered_r")


def format_cell(values: list[float]) -> str:
    if not values:
        return "NA"
    return f"{mean(values):.4f} ± {stdev(values):.4f}" if len(values) > 1 else f"{values[0]:.4f} (n=1)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/multiemotion_20260913/single_task_structure_matrix_A1"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)),
                        help="Comma-separated downstream seeds; the upstream EEG seed stays 240800.")
    parser.add_argument("--pipeline", default="single_task_label_specific_eegpt_scalar")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("downstream seeds must be nonempty and unique")
    out_dir = args.out_dir or args.root / "summary"
    rows = {}
    missing = []
    label_scale = {}
    for protocol in PROTOCOLS:
        reference = None
        for label in LABEL_NAMES:
            route = f"A1_Wphysio_no_audio__eeg_eegpt_partial_ft_single_{label}_v1"
            bag_path = args.root / "bags" / protocol / label / route / "embedding_seed_240800/ema_bags.npz"
            if not bag_path.is_file():
                if not args.allow_partial:
                    raise ValueError(f"missing bag: {bag_path}")
                continue
            bag = load_bag_dataset(bag_path)
            split = bag.split_indices()
            current = tuple(bag.event_id[split[leaf]].astype(str) for leaf in ("val", "test"))
            if reference is None:
                reference = current
            elif not all(np.array_equal(a, b) for a, b in zip(reference, current)):
                raise ValueError(f"unmatched event splits: {bag_path}")
            label_scale[(protocol, label)] = float(np.std(bag.label[split["train"]]))
            if label_scale[(protocol, label)] <= 0:
                raise ValueError(f"zero train target scale: {bag_path}")
            for condition in conditions():
                for seed in seeds:
                    path = args.root / "runs" / protocol / label / condition / f"seed_{seed}" / "metrics.json"
                    if not path.is_file() or not path.with_name("predictions.npz").is_file():
                        missing.append(str(path))
                        continue
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if (row.get("status") != "ok" or row.get("protocol") != protocol or
                            row.get("target_label") != label or row.get("condition_id") != condition or
                            row.get("downstream_seed") != seed or row.get("embedding_seed") != 240800 or
                            row.get("route_id") != route):
                        raise ValueError(f"invalid run provenance: {path}")
                    with np.load(path.with_name("predictions.npz"), allow_pickle=True) as pred:
                        if not np.array_equal(pred["event_id"][pred["test_index"]].astype(str), current[1]):
                            raise ValueError(f"test events differ: {path}")
                    rows[(protocol, label, condition, seed)] = row
    if missing and not args.allow_partial:
        raise ValueError(f"matrix incomplete: {len(missing)} missing; first={missing[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = {}
    for protocol in PROTOCOLS:
        selection[protocol] = {}
        for label in LABEL_NAMES:
            scores = {}
            for condition in conditions():
                available = [rows.get((protocol, label, condition, seed)) for seed in seeds]
                if all(row is not None for row in available):
                    scores[condition] = mean(row["val"]["rmse"] / label_scale[(protocol, label)] for row in available)
            selection[protocol][label] = min(scores, key=scores.get) if scores else None
    long_rows = []
    for protocol in PROTOCOLS:
        for metric in METRICS:
            lines = [f"# {protocol}: {metric}", "", "Independent scalar models: each emotion uses its own EEGPT partial-FT token (upstream seed 240800).",
                     "Cells are test EMA-event mean ± sample SD over three downstream seeds; NA means incomplete.", "",
                     "| Structure | " + " | ".join(LABEL_NAMES) + " |",
                     "| --- | " + " | ".join("---:" for _ in LABEL_NAMES) + " |"]
            for condition in conditions():
                cells = []
                for label in LABEL_NAMES:
                    values = []
                    for seed in seeds:
                        row = rows.get((protocol, label, condition, seed))
                        if row is None:
                            continue
                        value = row["test"]["rmse"] / label_scale[(protocol, label)] if metric == "standardized_rmse" else row["test"][metric]
                        if value is not None and np.isfinite(float(value)):
                            values.append(float(value))
                    cells.append(format_cell(values))
                    long_rows.append({"protocol": protocol, "condition_id": condition, "label": label,
                                      "metric": metric, "mean": mean(values) if values else "",
                                      "sd": stdev(values) if len(values) > 1 else "", "seed_count": len(values),
                                      "embedding_seed": 240800})
                lines.append(f"| {condition} | " + " | ".join(cells) + " |")
            (out_dir / f"{protocol}_{metric}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "structure_emotion_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    summary = {"pipeline": args.pipeline, "completed_runs": len(rows),
               "expected_runs": len(PROTOCOLS) * len(LABEL_NAMES) * len(conditions()) * len(seeds),
               "missing_count": len(missing), "embedding_seed": 240800, "downstream_seeds": list(seeds),
               "fixed_non_eeg_route": "A1_Wphysio_no_audio", "missing_metrics": missing,
               "validation_per_label_srmse_winner": selection}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"completed={len(rows)}/{summary['expected_runs']} missing={len(missing)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
