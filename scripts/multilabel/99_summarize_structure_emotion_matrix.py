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
ROUTE_ID = "A1_Wphysio_no_audio__eeg_eegpt_partial_ft_v1"
METRICS = ("raw_r", "standardized_rmse", "within_subject_centered_r")
BASELINE = "window_attention_regression_full_mean"
PROTOCOLS = ("cross_day", "within_subject_day")
SEEDS = (240800, 240801, 240802)


def condition_order() -> tuple[str, ...]:
    from daily_multimodal.training.structure_emotion import conditions
    return tuple(conditions())


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
                if row.get("route_id") != ROUTE_ID:
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
                f"Fixed input: `{ROUTE_ID}` (`embedding_seed=240800`). ",
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
        "fixed_route": ROUTE_ID,
        "validation_macro_srmse_winner": selection,
        "tables": [f"{protocol}_{metric}.md" for protocol in PROTOCOLS for metric in METRICS],
        "condition_summary": "condition_summary.csv",
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
