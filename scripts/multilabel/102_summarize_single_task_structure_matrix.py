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

PROTOCOLS = ("cross_day", "date_in_order")
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
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    args = parser.parse_args()
    protocols = tuple(value.strip() for value in args.protocols.split(",") if value.strip())
    if not protocols or len(protocols) != len(set(protocols)) or set(protocols) - {"cross_day", "within_subject_day", "date_in_order"}:
        raise ValueError("unsupported or duplicated protocols")
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("downstream seeds must be nonempty and unique")
    out_dir = args.out_dir or args.root / "summary"
    rows = {}
    missing = []
    label_scale = {}
    for protocol in protocols:
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
    for protocol in protocols:
        selection[protocol] = {}
        for label in LABEL_NAMES:
            scores = {}
            for condition in conditions():
                available = [rows.get((protocol, label, condition, seed)) for seed in seeds]
                if all(row is not None for row in available):
                    scores[condition] = mean(row["val"]["rmse"] / label_scale[(protocol, label)] for row in available)
            selection[protocol][label] = min(scores, key=scores.get) if scores else None
    long_rows = []
    for protocol in protocols:
        for metric in METRICS:
            best_raw_condition = {}
            if metric == "raw_r":
                for label in LABEL_NAMES:
                    candidates = []
                    for condition in conditions():
                        values = [rows[(protocol, label, condition, seed)]["test"][metric]
                                  for seed in seeds if (protocol, label, condition, seed) in rows]
                        finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
                        if finite:
                            candidates.append((mean(finite), condition))
                    if candidates:
                        best_raw_condition[label] = max(candidates, key=lambda item: item[0])[1]
            lines = [f"# {protocol}: {metric}", "", "Independent scalar models: each emotion uses its own EEGPT partial-FT token (upstream seed 240800).",
                     "Cells are test EMA-event mean ± sample SD over three downstream seeds; bold marks each label's highest test raw r for descriptive reading. NA means incomplete.", "",
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
                    cell = format_cell(values)
                    cells.append(f"**{cell}**" if metric == "raw_r" and condition == best_raw_condition.get(label) else cell)
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
               "protocols": list(protocols),
               "expected_runs": len(protocols) * len(LABEL_NAMES) * len(conditions()) * len(seeds),
               "missing_count": len(missing), "embedding_seed": 240800, "downstream_seeds": list(seeds),
               "fixed_non_eeg_route": "A1_Wphysio_no_audio", "missing_metrics": missing,
               "validation_per_label_srmse_winner": selection}
    preflight = []
    for protocol in protocols:
        for label in LABEL_NAMES:
            path = args.root / "preflight" / protocol / f"{label}.json"
            if path.is_file():
                preflight.append(json.loads(path.read_text(encoding="utf-8")))
    split_audit = {}
    for protocol in protocols:
        matching = [item for item in preflight if item["protocol"] == protocol]
        if matching:
            split_audit[protocol] = {
                "split_root": matching[0].get("split_root"),
                "window_event_overlap": matching[0].get("window_event_overlap", {}),
            }
    summary["split_audit"] = split_audit
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.pipeline == "single_task_label_specific_eegpt_scalar":
        def macro(protocol: str, condition: str, leaf: str, metric: str) -> float:
            values = []
            for label in LABEL_NAMES:
                for seed in seeds:
                    row = rows.get((protocol, label, condition, seed))
                    if row is None:
                        continue
                    value = row[leaf]["rmse"] / label_scale[(protocol, label)] if metric == "standardized_rmse" else row[leaf][metric]
                    if value is not None and np.isfinite(float(value)):
                        values.append(float(value))
            return mean(values) if values else float("nan")

        baseline = "window_attention_regression_full_mean"
        readme = [
            "# 独立单标签 EEGPT：结构 × 11 情绪结果", "",
            "每个情绪用本标签监督的 EEGPT partial-FT token（上游 seed `240800`），"
            "分别训练 0814 窗口结构和 0906 的 18 个 EMA-bag 结构；固定输入为 "
            "`A1_Wphysio_no_audio`，下游使用 3 个 seed。评价单位为 EMA event。", "",
            f"完成度：`{len(rows)}/{summary['expected_runs']}`，缺失 `{len(missing)}`。", "",
            "| 协议 | 0814 val macro sRMSE ↓ | 0814 test macro sRMSE ↓ | 0814 test macro raw r ↑ | 0814 test macro centered r ↑ | 选中 0814 的标签数 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for protocol in protocols:
            readme.append(
                f"| `{protocol}` | {macro(protocol, baseline, 'val', 'standardized_rmse'):.4f} | "
                f"{macro(protocol, baseline, 'test', 'standardized_rmse'):.4f} | "
                f"{macro(protocol, baseline, 'test', 'raw_r'):.4f} | "
                f"{macro(protocol, baseline, 'test', 'within_subject_centered_r'):.4f} | "
                f"{sum(winner == baseline for winner in selection[protocol].values())}/11 |"
            )
        readme.extend(["", "## Split 来源", ""])
        for protocol in protocols:
            audit = split_audit.get(protocol, {})
            root = audit.get("split_root") or "见上游 metrics.json"
            overlaps = audit.get("window_event_overlap", {})
            readme.append(f"- `{protocol}`：`{root}`。"
                          f"窗口级 train/val、val/test 共享 EMA event 数分别为 "
                          f"{overlaps.get('train_val', '未记录')}、{overlaps.get('val_test', '未记录')}。")
        if split_audit.get("within_subject_day", {}).get("window_event_overlap", {}).get("train_val", 0):
            readme.extend(["", "`within_subject_day` 的窗口级 split 存在跨集合事件共享；其测试分数不能当作独立 held-out-day 泛化估计。"])
        shared_summary_dir = args.root.parent / "structure_matrix_A1" / "summary"
        shared_summary_path = shared_summary_dir / "summary.json"
        shared_conditions_path = shared_summary_dir / "condition_summary.csv"
        if shared_summary_path.is_file() and shared_conditions_path.is_file():
            shared_summary = json.loads(shared_summary_path.read_text(encoding="utf-8"))
            shared_roots = shared_summary.get("split_roots", {})
            if (shared_summary.get("missing_count") == 0
                    and set(shared_summary.get("protocols", [])) == set(protocols)
                    and all(shared_roots.get(protocol) == split_audit.get(protocol, {}).get("split_root")
                            for protocol in protocols)):
                with shared_conditions_path.open("r", encoding="utf-8", newline="") as handle:
                    shared_conditions = list(csv.DictReader(handle))
                readme.extend(["", "## 相同 0814 窗口结构的两路线对照", "",
                               "两路线分别使用标签专属 EEGPT＋独立标量头、11标签共同监督 EEGPT＋共享多头；"
                               "下表按同一协议和下游 seed 报告宏平均筛查指标。", "",
                               "| 协议 | 路线 | test macro sRMSE ↓ | test macro raw r ↑ | test macro centered r ↑ |",
                               "| --- | --- | ---: | ---: | ---: |"])
                for protocol in protocols:
                    current = next(item for item in shared_conditions
                                   if item["protocol"] == protocol and item["condition_id"] == baseline)
                    readme.append(
                        f"| `{protocol}` | 独立单标签 | {macro(protocol, baseline, 'test', 'standardized_rmse'):.4f} | "
                        f"{macro(protocol, baseline, 'test', 'raw_r'):.4f} | "
                        f"{macro(protocol, baseline, 'test', 'within_subject_centered_r'):.4f} |"
                    )
                    readme.append(
                        f"| `{protocol}` | 共享多头 MT11 | {float(current['test_standardized_rmse_mean']):.4f} | "
                        f"{float(current['test_raw_r_mean']):.4f} | "
                        f"{float(current['test_within_subject_centered_r_mean']):.4f} |"
                    )
        readme.extend(["", "## 结构 × 11 情绪表", "",
                       "| 协议 | raw r ↑ | standardized RMSE ↓ | within-subject centered r ↑ |",
                       "| --- | --- | --- | --- |"])
        for protocol in protocols:
            readme.append(f"| `{protocol}` | [11 情绪表]({protocol}_raw_r.md) | "
                          f"[11 情绪表]({protocol}_standardized_rmse.md) | "
                          f"[11 情绪表]({protocol}_within_subject_centered_r.md) |")
        readme.extend(["", "[逐单元格长表](structure_emotion_long.csv) · "
                       "[完整性与验证集选择](summary.json) · [输入审计](preflight.json) · "
                       "[共享多头路线](../structure_matrix_summary/README.md)", ""])
        (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"completed={len(rows)}/{summary['expected_runs']} missing={len(missing)} out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
