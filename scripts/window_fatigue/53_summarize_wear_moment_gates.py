#!/usr/bin/env python3
"""汇总 Wear × MOMENT fusion matrix 并判定 G1/G2/G3 门槛（全维度泛化版）。

输入：scripts/window_fatigue/32_run_eegpt_centered_loss.py 的 report JSON（一个 seed 组一个文件）。

对每个 (protocol, eeg_branch, video, audio) 配置组，在同 seed 下配对比较四条 wear route：
- G1: wear_moment_frozen_v1 vs wear_deep（raw r 更高或 RMSE 更低）
- G2: wear_moment_partial_ft_v1 vs wear_moment_frozen_v1
- G3（多 seed）: 增益方向一致性 >= 2/3 seeds 且均值方向正确
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

VIDEOS = ("B0", "A1", "A2")
WEARS = ("Wphysio", "Wdeep", "Wmoment_frozen", "Wmoment_ft")
AUDIOS = ("full", "no_audio")
MAIN_PROTOCOLS = ("cross_day", "date_in_order")


def _parse_experiment(name: str) -> dict[str, str] | None:
    for video in VIDEOS:
        for wear in WEARS:
            for audio in AUDIOS:
                if name == f"{video}_{wear}_{audio}":
                    return {"video": video, "wear": wear, "audio": audio}
    return None


def _load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("status") == "failed":
                continue
            test = row.get("test") or {}
            rows.append(
                {
                    "protocol": row["protocol"],
                    "experiment": row["experiment"],
                    "eeg_branch": row.get("eeg_branch", "eeg"),
                    "seed": int(row.get("seed", -1)),
                    "rmse": float(test.get("rmse") or float("nan")),
                    "mae": float(test.get("mae") or float("nan")),
                    "raw_r": float(test.get("raw_r") or float("nan")),
                    "centered_r": float(test.get("within_subject_centered_r") or float("nan")),
                }
            )
    return rows


def _key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    parsed = _parse_experiment(row["experiment"])
    return (row["protocol"], row["eeg_branch"], parsed["video"], parsed["audio"])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protocols = sorted({row["protocol"] for row in rows})
    seeds = sorted({row["seed"] for row in rows})
    eeg_branches = sorted({row["eeg_branch"] for row in rows})
    by_key: dict[tuple[str, str, str, str], dict[str, dict[int, dict[str, Any]]]] = {}
    for row in rows:
        parsed = _parse_experiment(row["experiment"])
        if parsed is None:
            continue
        key = _key(row)
        bucket = by_key.setdefault(key, {})
        bucket.setdefault(parsed["wear"], {})[row["seed"]] = row

    configs: list[dict[str, Any]] = []
    for (protocol, eeg_branch, video, audio) in sorted(by_key):
        bucket = by_key[(protocol, eeg_branch, video, audio)]
        config: dict[str, Any] = {
            "protocol": protocol,
            "eeg_branch": eeg_branch,
            "video": video,
            "audio": audio,
            "rows": bucket,
        }
        configs.append(config)

    decisions: dict[str, Any] = {}
    for gate, metric, better in (
        ("G1_frozen_vs_deep", "raw_r", "higher"),
        ("G1_frozen_vs_deep", "rmse", "lower"),
        ("G1_frozen_vs_physio", "raw_r", "higher"),
        ("G1_frozen_vs_physio", "rmse", "lower"),
        ("G2_ft_vs_frozen", "raw_r", "higher"),
        ("G2_ft_vs_frozen", "rmse", "lower"),
    ):
        gate_rows: dict[str, list[float]] = {}
        for config in configs:
            bucket = config["rows"]
            if gate.startswith("G1_frozen_vs_deep"):
                a, b = "Wmoment_frozen", "Wdeep"
            elif gate.startswith("G1_frozen_vs_physio"):
                a, b = "Wmoment_frozen", "Wphysio"
            else:
                a, b = "Wmoment_ft", "Wmoment_frozen"
            for seed in seeds:
                row_a = bucket.get(a, {}).get(seed)
                row_b = bucket.get(b, {}).get(seed)
                if row_a is None or row_b is None:
                    continue
                delta = row_a[metric] - row_b[metric]
                if better == "lower":
                    delta = -delta
                key = f"{config['protocol']}|{config['eeg_branch']}|{config['video']}|{config['audio']}"
                gate_rows.setdefault(key, []).append(delta)
        summary: dict[str, Any] = {}
        for key, values in sorted(gate_rows.items()):
            mean = statistics.mean(values)
            signs = [1 if v > 0 else 0 for v in values]
            consistent = sum(signs) >= 2 if len(values) >= 3 else bool(values and signs[0])
            summary[key] = {
                "count": len(values),
                "mean_delta": mean,
                "improved_seeds": sum(signs),
                "direction_consistent": consistent,
            }
        decisions[f"{gate}_{metric}"] = summary

    def _any_config_better(gate_prefix: str, metric: str, *, require_consistency: bool) -> bool:
        for protocol in MAIN_PROTOCOLS:
            for key, entry in decisions[f"{gate_prefix}_{metric}"].items():
                if not key.startswith(protocol + "|"):
                    continue
                if require_consistency and entry["count"] >= 3 and not entry["direction_consistent"]:
                    continue
                if entry["mean_delta"] > 0:
                    return True
        return False

    g1_pass = _any_config_better("G1_frozen_vs_deep", "raw_r", require_consistency=True) or _any_config_better(
        "G1_frozen_vs_deep", "rmse", require_consistency=True
    )
    g2_pass = _any_config_better("G2_ft_vs_frozen", "raw_r", require_consistency=False) or _any_config_better(
        "G2_ft_vs_frozen", "rmse", require_consistency=False
    )
    g3_pass = g1_pass and any(seeds) and len(seeds) >= 3
    return {
        "protocols": protocols,
        "seeds": seeds,
        "eeg_branches": eeg_branches,
        "config_count": len(configs),
        "configs": configs,
        "decisions": decisions,
        "gates": {
            "G1_screen_pass": g1_pass,
            "G2_ft_gain_pass": g2_pass,
            "G3_confirm_stable_pass": g3_pass,
        },
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Wear × MOMENT Fusion Gate Summary (full-dimension)",
        "",
        f"- protocols: `{result['protocols']}`",
        f"- seeds: `{result['seeds']}`",
        f"- eeg_branches: `{result['eeg_branches']}`",
        f"- config groups: `{result['config_count']}`",
        f"- gates: `{result['gates']}`",
        "",
        "## Mean delta across seeds per config (raw_r / rmse, positive = better)",
        "",
        "| protocol | eeg | video | audio | G1 frozen-vs-deep | G1 frozen-vs-physio | G2 ft-vs-frozen |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    raw_keys = "G1_frozen_vs_deep_raw_r", "G1_frozen_vs_physio_raw_r", "G2_ft_vs_frozen_raw_r"
    rmse_keys = "G1_frozen_vs_deep_rmse", "G1_frozen_vs_physio_rmse", "G2_ft_vs_frozen_rmse"
    all_keys = sorted({key for key in result["decisions"]["G1_frozen_vs_deep_raw_r"]})
    for key in all_keys:
        protocol, eeg_branch, video, audio = key.split("|")
        cells = []
        for raw_key, rmse_key in zip(raw_keys, rmse_keys):
            raw_entry = result["decisions"][raw_key].get(key)
            rmse_entry = result["decisions"][rmse_key].get(key)
            if raw_entry:
                cells.append(f"raw {raw_entry['mean_delta']:+.4f} ({raw_entry['improved_seeds']}/{raw_entry['count']}) / rmse {rmse_entry['mean_delta']:+.4f}")
            else:
                cells.append("NA")
        lines.append(f"| {protocol} | {eeg_branch} | {video} | {audio} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Full metrics by config", "", "| protocol | eeg | video | audio | seed | route | RMSE | MAE | raw r | centered r |", "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"])
    for config in sorted(result["configs"], key=lambda c: (c["protocol"], c["eeg_branch"], c["video"], c["audio"])):
        for seed in result["seeds"]:
            for wear in ("Wphysio", "Wdeep", "Wmoment_frozen", "Wmoment_ft"):
                row = config["rows"].get(wear, {}).get(seed)
                if row is None:
                    continue
                lines.append(
                    f"| {config['protocol']} | {config['eeg_branch']} | {config['video']} | {config['audio']} | {seed} | {wear} | "
                    f"{row['rmse']:.4f} | {row['mae']:.4f} | {row['raw_r']:.4f} | {row['centered_r']:.4f} |"
                )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", required=True, help="Comma-separated fusion report JSON paths (one per seed group).")
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()
    rows = _load_reports([Path(p) for p in args.reports.split(",") if p.strip()])
    result = summarize(rows)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(result, Path(args.out_md))
    print(f"config_count={result['config_count']}")
    print(f"gates={result['gates']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
