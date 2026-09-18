#!/usr/bin/env python3
"""Select validation routes and evaluate the paired H1-vs-H0 multi-emotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    route_rows = load_rows(args.route_root)
    route_selection = select_routes(route_rows)
    output: dict[str, Any] = {"route_selection": route_selection}
    if args.head_root:
        output["head_gate"] = evaluate_head_gate(load_rows(args.head_root), route_selection)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output, args.out_root / "summary.md")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def load_rows(root: Path) -> list[dict[str, Any]]:
    manifest = root / "manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))["results"]
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.rglob("metrics.json"))]


def select_routes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    e0 = [row for row in rows if row["head_variant"] == "E0_existing_11out"]
    result: dict[str, Any] = {}
    for protocol in sorted({row["protocol"] for row in e0}):
        protocol_rows = [row for row in e0 if row["protocol"] == protocol]
        candidates = []
        for experiment in sorted({row["experiment"] for row in protocol_rows}):
            matched = [row for row in protocol_rows if row["experiment"] == experiment]
            values = [float(row["val"]["summary"]["standardized_rmse"]) for row in matched]
            candidates.append(
                {
                    "experiment": experiment,
                    "seed_count": len(matched),
                    "seeds": [int(row["seed"]) for row in matched],
                    "val_macro_standardized_rmse_mean": float(np.mean(values)),
                    "val_macro_standardized_rmse_std": float(np.std(values)),
                }
            )
        if candidates:
            winner = min(candidates, key=lambda row: row["val_macro_standardized_rmse_mean"])
            result[protocol] = {"winner": winner, "candidates": candidates}
    return result


def evaluate_head_gate(rows: list[dict[str, Any]], route_selection: dict[str, Any]) -> dict[str, Any]:
    protocols: dict[str, Any] = {}
    for protocol, selection in route_selection.items():
        route = selection["winner"]["experiment"]
        selected = [row for row in rows if row["protocol"] == protocol and row["experiment"] == route]
        by_key = {(row["head_variant"], int(row["seed"])): row for row in selected}
        seeds = sorted(
            set(seed for head, seed in by_key if head == "H0_shared2_linear11")
            & set(seed for head, seed in by_key if head == "H1_shared2_11xhead2")
        )
        deltas = []
        for seed in seeds:
            h0 = by_key[("H0_shared2_linear11", seed)]
            h1 = by_key[("H1_shared2_11xhead2", seed)]
            delta = {
                "seed": seed,
                "standardized_rmse": metric(h1, "summary", "standardized_rmse") - metric(h0, "summary", "standardized_rmse"),
                "raw_r": metric(h1, "summary", "raw_r") - metric(h0, "summary", "raw_r"),
                "groups": {},
            }
            for group in ("positive_activation", "negative_distress", "fatigue"):
                delta["groups"][group] = {
                    "standardized_rmse": metric(h1, "groups", group, "standardized_rmse")
                    - metric(h0, "groups", group, "standardized_rmse"),
                    "raw_r": metric(h1, "groups", group, "raw_r") - metric(h0, "groups", group, "raw_r"),
                }
            deltas.append(delta)
        mean_srmse = finite_mean([row["standardized_rmse"] for row in deltas])
        mean_raw_r = finite_mean([row["raw_r"] for row in deltas])
        direction_wins = sum(
            row["standardized_rmse"] < 0.0 or row["raw_r"] > 0.0 for row in deltas
        )
        group_guardrail = all(
            not (
                finite_mean([row["groups"][group]["standardized_rmse"] for row in deltas]) > 0.02
                and finite_mean([row["groups"][group]["raw_r"] for row in deltas]) < -0.02
            )
            for group in ("positive_activation", "negative_distress", "fatigue")
        ) if deltas else False
        protocols[protocol] = {
            "route": route,
            "seed_count": len(seeds),
            "deltas_h1_minus_h0": deltas,
            "mean_delta_standardized_rmse": mean_srmse,
            "mean_delta_raw_r": mean_raw_r,
            "direction_wins": direction_wins,
            "noninferiority_pass": bool(mean_srmse <= 0.01) if mean_srmse is not None else False,
            "improvement_pass": bool(mean_srmse <= -0.02 or mean_raw_r >= 0.02) if mean_srmse is not None and mean_raw_r is not None else False,
            "direction_pass": bool(direction_wins >= 2 and len(seeds) >= 3),
            "group_guardrail_pass": bool(group_guardrail),
        }
    enough = len(protocols) == 2 and all(row["seed_count"] >= 3 for row in protocols.values())
    noninferior = enough and all(row["noninferiority_pass"] for row in protocols.values())
    improvement = enough and any(row["improvement_pass"] for row in protocols.values())
    direction = enough and all(row["direction_pass"] for row in protocols.values())
    guardrail = enough and all(row["group_guardrail_pass"] for row in protocols.values())
    passed = bool(enough and noninferior and improvement and direction and guardrail)
    return {
        "protocols": protocols,
        "enough_matched_runs": enough,
        "both_protocol_noninferiority": noninferior,
        "at_least_one_protocol_improvement": improvement,
        "direction_consistency": direction,
        "group_guardrails": guardrail,
        "gate_passed": passed,
        "decision": "promote_multihead" if passed else "do_not_promote_multihead",
    }


def metric(row: dict[str, Any], *path: str) -> float:
    value: Any = row["test"]
    for key in path:
        value = value[key]
    return float(value)


def finite_mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def write_markdown(output: dict[str, Any], path: Path) -> None:
    lines = ["# Multi-emotion Phase 2 Summary", "", "## Validation-only Route Selection", ""]
    for protocol, row in output["route_selection"].items():
        winner = row["winner"]
        lines.append(
            f"- `{protocol}`: `{winner['experiment']}`, val macro sRMSE "
            f"`{winner['val_macro_standardized_rmse_mean']:.4f} +/- {winner['val_macro_standardized_rmse_std']:.4f}`."
        )
    if "head_gate" in output:
        gate = output["head_gate"]
        lines.extend(["", "## H1 - H0 Gate", ""])
        for protocol, row in gate["protocols"].items():
            lines.append(
                f"- `{protocol}` / `{row['route']}`: delta sRMSE `{row['mean_delta_standardized_rmse']:.4f}`, "
                f"delta raw r `{row['mean_delta_raw_r']:.4f}`, wins `{row['direction_wins']}/{row['seed_count']}`."
            )
        lines.extend(["", f"Decision: `{gate['decision']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
