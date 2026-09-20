#!/usr/bin/env python3
"""Pivot matched structure runs into protocol-specific structure × 11 emotion tables."""

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


LABELS = (
    "inspired", "alert", "determined", "attentive", "active",
    "hostile", "nervous", "upset", "afraid", "ashamed", "fatigue",
)
METRICS = ("raw_r", "standardized_rmse", "within_subject_centered_r")
BASELINE = "window_attention_regression_full_mean"
PROTOCOLS = ("cross_day", "within_subject_day")
SEEDS = (240800, 240801, 240802)


def condition_order() -> tuple[str, ...]:
    from daily_multimodal.training.structure_emotion import conditions
    return tuple(conditions())


def fixed_route_id() -> str:
    from daily_multimodal.training.structure_emotion import ROUTE_ID
    return ROUTE_ID


def format_cell(values: list[float]) -> str:
    if not values:
        return "NA"
    if len(values) == 1:
        return f"{values[0]:.4f} (n=1)"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def read_runs(root: Path, *, allow_partial: bool) -> tuple[dict[tuple[str, str, int], dict], list[str]]:
    rows = {}
    missing = []
    for protocol in PROTOCOLS:
        for condition in condition_order():
            for seed in SEEDS:
                path = root / "runs" / protocol / condition / f"seed_{seed}" / "metrics.json"
                prediction_path = path.with_name("event_predictions.npz")
                if not path.is_file() or not prediction_path.is_file():
                    missing.append(str(path))
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") != "ok" or row.get("embedding_seed") != 240800 or row.get("downstream_seed") != seed:
                    raise ValueError(f"invalid run provenance: {path}")
                if row.get("route_id") != fixed_route_id():
                    raise ValueError(f"mixed embedding routes: {path}")
                if row.get("head_variant") != "H1_shared2_11xhead2":
                    raise ValueError(f"mixed head architectures: {path}")
                rows[(protocol, condition, seed)] = row
    if missing and not allow_partial:
        raise ValueError(f"matrix incomplete: {len(missing)} missing runs; first={missing[0]}")
    return rows, missing


def audit_matched_events(root: Path, rows: dict[tuple[str, str, int], dict]) -> None:
    for protocol in PROTOCOLS:
        reference = None
        for condition in condition_order():
            for seed in SEEDS:
                if (protocol, condition, seed) not in rows:
                    continue
                path = root / "runs" / protocol / condition / f"seed_{seed}" / "event_predictions.npz"
                with np.load(path) as z:
                    labels = tuple(z["label_names"].astype(str).tolist())
                    if labels != LABELS:
                        raise ValueError(f"label order mismatch: {path}")
                    current = tuple((z[f"{leaf}_event_id"].astype(str), z[f"{leaf}_target"]) for leaf in ("val", "test"))
                if reference is None:
                    reference = current
                elif not all(
                    np.array_equal(current[i][0], reference[i][0]) and np.array_equal(current[i][1], reference[i][1])
                    for i in (0, 1)
                ):
                    raise ValueError(f"unmatched EMA events or labels: {path}")


def write_tables(root: Path, out_dir: Path, rows: dict[tuple[str, str, int], dict], missing: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    long_rows = []
    condition_rows = []
    selection = {}
    for protocol in PROTOCOLS:
        scores = {}
        for condition in condition_order():
            available = [rows[(protocol, condition, seed)] for seed in SEEDS if (protocol, condition, seed) in rows]
            if len(available) == len(SEEDS):
                scores[condition] = mean(row["val"]["summary"]["standardized_rmse"] for row in available)
        selection[protocol] = min(scores, key=scores.get) if scores else None
        for condition in condition_order():
            available_seeds = [seed for seed in SEEDS if (protocol, condition, seed) in rows]
            item = {"protocol": protocol, "condition_id": condition, "seed_count": len(available_seeds),
                    "validation_selected": condition == selection[protocol]}
            for leaf in ("val", "test"):
                for metric in METRICS:
                    values = [rows[(protocol, condition, seed)][leaf]["summary"][metric]
                              for seed in available_seeds]
                    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
                    item[f"{leaf}_{metric}_mean"] = mean(finite) if finite else ""
                    item[f"{leaf}_{metric}_sd"] = stdev(finite) if len(finite) > 1 else ""
            paired_seeds = [seed for seed in available_seeds if (protocol, BASELINE, seed) in rows]
            for metric in METRICS:
                deltas = []
                for seed in paired_seeds:
                    current = rows[(protocol, condition, seed)]["test"]["summary"][metric]
                    baseline = rows[(protocol, BASELINE, seed)]["test"]["summary"][metric]
                    if current is not None and baseline is not None:
                        deltas.append(float(current) - float(baseline))
                item[f"paired_test_delta_{metric}_mean"] = mean(deltas) if deltas else ""
            condition_rows.append(item)
        for metric in METRICS:
            table = [
                f"# {protocol}: {metric}", "",
                f"Fixed input: `{fixed_route_id()}` (`embedding_seed=240800`). ",
                "Each cell is test EMA-event mean ± sample SD over downstream seeds `240800,240801,240802`. ",
                f"Validation macro-sRMSE winner: `{selection[protocol] or 'pending'}`. ", "",
                "| Structure | " + " | ".join(LABELS) + " |",
                "| --- | " + " | ".join("---:" for _ in LABELS) + " |",
            ]
            for condition in condition_order():
                cells = []
                for label in LABELS:
                    values = []
                    for seed in SEEDS:
                        row = rows.get((protocol, condition, seed))
                        value = row["test"]["per_label"][label][metric] if row else None
                        if value is not None and np.isfinite(float(value)):
                            values.append(float(value))
                    cells.append(format_cell(values))
                    long_rows.append({
                        "protocol": protocol, "condition_id": condition, "label": label, "metric": metric,
                        "mean": mean(values) if values else "", "std": stdev(values) if len(values) > 1 else "",
                        "seed_count": len(values), "embedding_seed": 240800,
                    })
                table.append(f"| {condition} | " + " | ".join(cells) + " |")
            (out_dir / f"{protocol}_{metric}.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    with (out_dir / "structure_emotion_long.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
        writer.writeheader()
        writer.writerows(long_rows)
    with (out_dir / "condition_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(condition_rows[0]))
        writer.writeheader()
        writer.writerows(condition_rows)
    payload = {
        "completed_runs": len(rows), "expected_runs": len(PROTOCOLS) * len(condition_order()) * len(SEEDS),
        "missing_count": len(missing), "missing_metrics": missing,
        "embedding_seed": 240800, "downstream_seeds": list(SEEDS),
        "fixed_route": fixed_route_id(),
        "validation_macro_srmse_winner": selection,
        "tables": [f"{protocol}_{metric}.md" for protocol in PROTOCOLS for metric in METRICS],
        "condition_summary": "condition_summary.csv",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    preflight_path = root / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError(f"missing structure-matrix preflight: {preflight_path}")
    (out_dir / "preflight.json").write_bytes(preflight_path.read_bytes())
    overview_rows = []
    for protocol in PROTOCOLS:
        protocol_rows = [row for row in condition_rows if row["protocol"] == protocol]
        window_row = next(row for row in protocol_rows if row["condition_id"] == BASELINE)
        best_0906 = min(
            (row for row in protocol_rows if row["condition_id"] != BASELINE),
            key=lambda row: float(row["val_standardized_rmse_mean"]),
        )
        for route_name, row in (("0814 窗口路线", window_row), (f"0906 `{best_0906['condition_id']}`", best_0906)):
            overview_rows.append(
                f"| `{protocol}` | {route_name} | {float(row['val_standardized_rmse_mean']):.4f} | "
                f"{float(row['test_standardized_rmse_mean']):.4f} | {float(row['test_raw_r_mean']):.4f} | "
                f"{float(row['test_within_subject_centered_r_mean']):.4f} |"
            )
    readme = [
        "# MT11 EEGPT：结构 × 11 情绪结果",
        "",
        f"固定输入：`{fixed_route_id()}`；EEGPT 上游 `embedding_seed=240800`，下游 seeds 为 "
        "`240800,240801,240802`。EEGPT encoder 由 11 个标签共同监督，所有结构行共用同一套 256D EEG token，"
        "并统一使用两层共享 MLP 与 11 个独立两层回归头。评价单位为 EMA event。",
        "",
        f"完成度：`{len(rows)}/{payload['expected_runs']}`，缺失 `{len(missing)}`。结构只按 validation macro "
        "standardized RMSE 选择。",
        "",
        "| 协议 | validation 选中结构 |",
        "| --- | --- |",
        *[f"| `{protocol}` | `{selection[protocol] or 'pending'}` |" for protocol in PROTOCOLS],
        "",
        "## 0814 与验证集最优 0906 结构",
        "",
        "以下 test 指标均为三个下游 seed 的均值；0906 行仅按 validation macro sRMSE 选择。",
        "",
        "| 协议 | 路线 | val macro sRMSE ↓ | test macro sRMSE ↓ | test macro raw r ↑ | test macro centered r ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        *overview_rows,
        "",
        "## 完整的结构 × 11 情绪表",
        "",
        "| 协议 | raw r ↑ | standardized RMSE ↓ | within-subject centered r ↑ |",
        "| --- | --- | --- | --- |",
        *[
            f"| `{protocol}` | [11 情绪表]({protocol}_raw_r.md) | "
            f"[11 情绪表]({protocol}_standardized_rmse.md) | "
            f"[11 情绪表]({protocol}_within_subject_centered_r.md) |"
            for protocol in PROTOCOLS
        ],
        "",
        "[逐结构宏指标与 matched-seed 0814 基线差值](condition_summary.csv) · "
        "[逐单元格长表](structure_emotion_long.csv) · "
        "[完整性与选择摘要](summary.json) · [固定输入与 split preflight](preflight.json)",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"completed={len(rows)}/{payload['expected_runs']} missing={len(missing)} out={out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/multiemotion_20260913/structure_matrix_A1"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows, missing = read_runs(args.root, allow_partial=args.allow_partial)
    audit_matched_events(args.root, rows)
    write_tables(args.root, args.out_dir or args.root / "summary", rows, missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
